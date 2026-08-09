"""Calendar domain models for Milestone 20.

Calendar events are transient provider views. Local persistence is limited to
account metadata, change proposals, and bounded audit entries. Event title and
description text from the provider are untrusted display data only.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Literal, Mapping, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.config import (
    CALENDAR_PROPOSAL_TTL_SECONDS,
    GOOGLE_CALENDAR_SCOPES,
    MAX_CALENDAR_ACCOUNT_DISPLAY_LABEL_LENGTH,
    MAX_CALENDAR_ERROR_MESSAGE_LENGTH,
    MAX_CALENDAR_EVENT_DESCRIPTION_LENGTH,
    MAX_CALENDAR_EVENT_ID_LENGTH,
    MAX_CALENDAR_EVENT_TITLE_LENGTH,
    MAX_CALENDAR_ID_LENGTH,
)
from src.reminder import (
    LOCAL_WALL_FORMAT,
    local_wall_to_utc_iso,
    parse_local_wall_datetime,
    validate_iana_timezone,
)
from src.tool_common import (
    canonical_json,
    parse_utc_timestamp,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_utc_timestamp,
    validate_uuid,
)

CalendarProviderId = Literal["google"]
CalendarAccountStatus = Literal["connected", "needs_reauth"]
CalendarProposalOperation = Literal["create", "reschedule", "cancel"]
CalendarProposalStatus = Literal[
    "pending",
    "executed",
    "failed",
    "unknown_outcome",
]
CalendarAuditAction = Literal[
    "connected",
    "disconnected",
    "default_calendar_set",
    "create_prepared",
    "created",
    "reschedule_prepared",
    "rescheduled",
    "cancel_prepared",
    "cancelled",
    "proposal_failed",
    "proposal_unknown_outcome",
]
CalendarErrorCategory = Literal[
    "auth_error",
    "not_found",
    "conflict",
    "rate_limited",
    "server_error",
    "network_error",
    "validation_error",
    "secret_store_error",
]

CALENDAR_PROVIDER_IDS: frozenset[str] = frozenset({"google"})
CALENDAR_ACCOUNT_STATUSES: frozenset[str] = frozenset(
    {"connected", "needs_reauth"}
)
CALENDAR_PROPOSAL_OPERATIONS: frozenset[str] = frozenset(
    {"create", "reschedule", "cancel"}
)
CALENDAR_PROPOSAL_STATUSES: frozenset[str] = frozenset(
    {"pending", "executed", "failed", "unknown_outcome"}
)
CALENDAR_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "connected",
        "disconnected",
        "default_calendar_set",
        "create_prepared",
        "created",
        "reschedule_prepared",
        "rescheduled",
        "cancel_prepared",
        "cancelled",
        "proposal_failed",
        "proposal_unknown_outcome",
    }
)
CALENDAR_ERROR_CATEGORIES: frozenset[str] = frozenset(
    {
        "auth_error",
        "not_found",
        "conflict",
        "rate_limited",
        "server_error",
        "network_error",
        "validation_error",
        "secret_store_error",
    }
)

GOOGLE_EVENT_ID_PATTERN = re.compile(r"^[0-9a-v]{5,1024}$")
PRIMARY_CALENDAR_ID = "primary"


class CalendarValidationError(ValueError):
    """Raised when calendar domain validation fails."""


class CalendarError(RuntimeError):
    """Bounded calendar operation failure with a safe user message."""

    def __init__(
        self,
        message: str,
        *,
        category: CalendarErrorCategory,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.user_message = user_message or _default_user_message(category)


class CalendarClock(Protocol):
    """Injectable clock for deterministic calendar proposal tests."""

    def utc_now_iso(self) -> str:
        """Return the current UTC timestamp in ISO 8601 form with trailing Z."""


class SystemCalendarClock:
    """Production clock backed by the system UTC clock."""

    def utc_now_iso(self) -> str:
        return utc_timestamp()


class ManualCalendarClock:
    """Test clock with an explicitly controllable UTC instant."""

    def __init__(self, *, start: datetime | None = None) -> None:
        if start is None:
            self._utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
        else:
            if start.tzinfo is None:
                raise CalendarValidationError("ManualCalendarClock start must be aware.")
            self._utc = start.astimezone(timezone.utc)

    def utc_now_iso(self) -> str:
        return self._utc.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def set_utc(self, value: datetime) -> None:
        """Set the clock to an aware UTC datetime."""
        if value.tzinfo is None:
            raise CalendarValidationError("ManualCalendarClock values must be aware.")
        self._utc = value.astimezone(timezone.utc)

    def advance(self, seconds: float) -> None:
        """Advance the clock by the given number of seconds."""
        self._utc = self._utc + timedelta(seconds=float(seconds))


@dataclass(frozen=True)
class CalendarAccount:
    """Local metadata for one connected Google Calendar account."""

    account_id: str
    provider_id: CalendarProviderId
    display_label: str
    status: CalendarAccountStatus
    default_calendar_id: str
    connected_at: str
    scopes_granted: tuple[str, ...]


@dataclass(frozen=True)
class CalendarInfo:
    """Transient calendar metadata from the provider."""

    calendar_id: str
    summary: str
    timezone: str
    primary: bool
    access_role: str


@dataclass(frozen=True)
class CalendarEvent:
    """Transient normalized calendar event view (not persisted)."""

    event_id: str
    calendar_id: str
    title: str
    description: str | None
    is_all_day: bool
    start: str
    end: str
    status: str
    is_recurring: bool
    recurring_event_id: str | None
    etag: str | None


@dataclass(frozen=True)
class FreeBusyWindow:
    """One busy interval from provider free/busy."""

    calendar_id: str
    start: str
    end: str


@dataclass(frozen=True)
class CalendarChangeProposal:
    """One-shot prepare→confirm write intent."""

    proposal_id: str
    account_id: str
    calendar_id: str
    operation: CalendarProposalOperation
    event_id: str | None
    client_event_id: str | None
    normalized_payload: dict[str, str]
    fingerprint: str
    remote_etag: str | None
    status: CalendarProposalStatus
    created_at: str
    expires_at: str


@dataclass(frozen=True)
class CalendarAuditEntry:
    """Bounded local audit record for calendar actions."""

    audit_id: str
    action: CalendarAuditAction
    occurred_at: str
    account_id: str | None
    proposal_id: str | None
    calendar_id: str | None
    event_id: str | None
    operation: str | None
    from_status: str | None
    to_status: str | None
    error_category: str | None


def _default_user_message(category: CalendarErrorCategory) -> str:
    messages = {
        "auth_error": (
            "Cortana: Calendar authorization failed. "
            "Reconnect with /calendar-connect."
        ),
        "not_found": "Cortana: The requested calendar resource was not found.",
        "conflict": (
            "Cortana: A calendar conflict was detected. "
            "Adjust the time and prepare again."
        ),
        "rate_limited": (
            "Cortana: The calendar provider rate-limited this request. Try again later."
        ),
        "server_error": "Cortana: The calendar provider reported a server error.",
        "network_error": "Cortana: The calendar provider could not be reached.",
        "validation_error": "Cortana: Calendar input validation failed.",
        "secret_store_error": "Cortana: Secure credential storage is unavailable.",
    }
    return messages[category]


def validate_calendar_provider_id(value: str) -> CalendarProviderId:
    """Validate production provider_id (google only)."""
    cleaned = validate_controlled_value(
        value,
        field_name="Calendar provider",
        allowed=CALENDAR_PROVIDER_IDS,
    )
    return cleaned  # type: ignore[return-value]


def validate_calendar_id(value: str, *, field_name: str = "Calendar ID") -> str:
    """Validate a calendar identifier or the primary alias."""
    return require_non_blank_text(
        value,
        field_name=field_name,
        max_length=MAX_CALENDAR_ID_LENGTH,
    )


def validate_event_id(value: str, *, field_name: str = "Event ID") -> str:
    """Validate a provider event identifier."""
    return require_non_blank_text(
        value,
        field_name=field_name,
        max_length=MAX_CALENDAR_EVENT_ID_LENGTH,
    )


def validate_google_event_id(value: str, *, field_name: str = "Client event ID") -> str:
    """Validate a Google Calendar API-compliant event ID."""
    cleaned = require_non_blank_text(
        value,
        field_name=field_name,
        max_length=MAX_CALENDAR_EVENT_ID_LENGTH,
    )
    if not GOOGLE_EVENT_ID_PATTERN.fullmatch(cleaned):
        raise CalendarValidationError(
            f"{field_name} must use Google Calendar base32hex characters "
            "(0-9, a-v) with length 5-1024."
        )
    return cleaned


def generate_google_client_event_id() -> str:
    """Generate one Google-compliant event ID for create proposals."""
    alphabet = "0123456789abcdefghijklmnopqrstuv"
    # 26 chars after prefix keeps length well within Google bounds.
    body = "".join(secrets.choice(alphabet) for _ in range(26))
    return validate_google_event_id(f"c20{body}")


def validate_event_title(value: str) -> str:
    """Validate a bounded event title."""
    return require_non_blank_text(
        value,
        field_name="Event title",
        max_length=MAX_CALENDAR_EVENT_TITLE_LENGTH,
    )


def validate_optional_event_description(value: str | None) -> str | None:
    """Validate optional bounded event description for display paths."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_CALENDAR_EVENT_DESCRIPTION_LENGTH:
        cleaned = cleaned[:MAX_CALENDAR_EVENT_DESCRIPTION_LENGTH]
    return cleaned


def validate_scopes_granted(scopes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Require exactly the M20 Google Calendar scopes, order-insensitive."""
    if not isinstance(scopes, (tuple, list)):
        raise CalendarValidationError("scopes_granted must be a sequence of strings.")
    cleaned = tuple(sorted({str(item).strip() for item in scopes if str(item).strip()}))
    expected = tuple(sorted(GOOGLE_CALENDAR_SCOPES))
    if cleaned != expected:
        raise CalendarValidationError(
            "scopes_granted must be exactly the Milestone 20 Google Calendar scopes."
        )
    return cleaned


def create_calendar_account(
    *,
    display_label: str,
    default_calendar_id: str = PRIMARY_CALENDAR_ID,
    status: str = "connected",
    scopes_granted: tuple[str, ...] | list[str] | None = None,
    connected_at: str | None = None,
    account_id: str | None = None,
    clock: CalendarClock | None = None,
) -> CalendarAccount:
    """Create one validated Google calendar account metadata record."""
    active_clock = clock or SystemCalendarClock()
    return validate_calendar_account(
        CalendarAccount(
            account_id=validate_uuid(
                account_id or str(uuid4()),
                field_name="Account ID",
            ),
            provider_id="google",
            display_label=require_non_blank_text(
                display_label,
                field_name="Display label",
                max_length=MAX_CALENDAR_ACCOUNT_DISPLAY_LABEL_LENGTH,
            ),
            status=validate_controlled_value(  # type: ignore[arg-type]
                status,
                field_name="Account status",
                allowed=CALENDAR_ACCOUNT_STATUSES,
            ),
            default_calendar_id=validate_calendar_id(default_calendar_id),
            connected_at=validate_utc_timestamp(
                connected_at or active_clock.utc_now_iso(),
                field_name="Connected at",
            ),
            scopes_granted=validate_scopes_granted(
                scopes_granted if scopes_granted is not None else GOOGLE_CALENDAR_SCOPES
            ),
        )
    )


def validate_calendar_account(account: CalendarAccount) -> CalendarAccount:
    """Validate one calendar account record."""
    return CalendarAccount(
        account_id=validate_uuid(account.account_id, field_name="Account ID"),
        provider_id=validate_calendar_provider_id(account.provider_id),
        display_label=require_non_blank_text(
            account.display_label,
            field_name="Display label",
            max_length=MAX_CALENDAR_ACCOUNT_DISPLAY_LABEL_LENGTH,
        ),
        status=validate_controlled_value(  # type: ignore[arg-type]
            account.status,
            field_name="Account status",
            allowed=CALENDAR_ACCOUNT_STATUSES,
        ),
        default_calendar_id=validate_calendar_id(account.default_calendar_id),
        connected_at=validate_utc_timestamp(
            account.connected_at,
            field_name="Connected at",
        ),
        scopes_granted=validate_scopes_granted(account.scopes_granted),
    )


def replace_calendar_account(
    account: CalendarAccount,
    *,
    status: str | None = None,
    default_calendar_id: str | None = None,
    display_label: str | None = None,
) -> CalendarAccount:
    """Return a validated copy of an account with selected fields replaced."""
    return validate_calendar_account(
        CalendarAccount(
            account_id=account.account_id,
            provider_id=account.provider_id,
            display_label=(
                display_label if display_label is not None else account.display_label
            ),
            status=status if status is not None else account.status,  # type: ignore[arg-type]
            default_calendar_id=(
                default_calendar_id
                if default_calendar_id is not None
                else account.default_calendar_id
            ),
            connected_at=account.connected_at,
            scopes_granted=account.scopes_granted,
        )
    )


def validate_calendar_info(info: CalendarInfo) -> CalendarInfo:
    """Validate one transient calendar info record."""
    timezone_name = validate_iana_timezone(info.timezone, field_name="Calendar timezone")
    return CalendarInfo(
        calendar_id=validate_calendar_id(info.calendar_id),
        summary=require_non_blank_text(
            info.summary,
            field_name="Calendar summary",
            max_length=MAX_CALENDAR_EVENT_TITLE_LENGTH,
        ),
        timezone=timezone_name,
        primary=bool(info.primary),
        access_role=require_non_blank_text(
            info.access_role,
            field_name="Access role",
            max_length=64,
        ),
    )


def validate_calendar_event(event: CalendarEvent) -> CalendarEvent:
    """Validate one transient calendar event view."""
    if event.is_all_day:
        _validate_all_day_date(event.start, field_name="All-day start")
        _validate_all_day_date(event.end, field_name="All-day end")
    else:
        validate_utc_timestamp(event.start, field_name="Event start")
        validate_utc_timestamp(event.end, field_name="Event end")
        if parse_utc_timestamp(event.start) >= parse_utc_timestamp(event.end):
            raise CalendarValidationError("Event end must be after event start.")

    recurring_event_id = None
    if event.recurring_event_id is not None:
        recurring_event_id = validate_event_id(
            event.recurring_event_id,
            field_name="Recurring event ID",
        )
    is_recurring = bool(event.is_recurring) or recurring_event_id is not None
    return CalendarEvent(
        event_id=validate_event_id(event.event_id),
        calendar_id=validate_calendar_id(event.calendar_id),
        title=validate_event_title(event.title) if event.title.strip() else "(untitled)",
        description=validate_optional_event_description(event.description),
        is_all_day=bool(event.is_all_day),
        start=event.start.strip(),
        end=event.end.strip(),
        status=require_non_blank_text(
            event.status,
            field_name="Event status",
            max_length=64,
        ),
        is_recurring=is_recurring,
        recurring_event_id=recurring_event_id,
        etag=(
            require_non_blank_text(event.etag, field_name="ETag", max_length=200)
            if event.etag is not None and event.etag.strip()
            else None
        ),
    )


def event_blocks_writes(event: CalendarEvent) -> bool:
    """Return True when M20 must refuse write operations on this event."""
    return bool(event.is_recurring) or event.recurring_event_id is not None or event.is_all_day


def validate_freebusy_window(window: FreeBusyWindow) -> FreeBusyWindow:
    """Validate one free/busy busy window."""
    start = validate_utc_timestamp(window.start, field_name="Busy start")
    end = validate_utc_timestamp(window.end, field_name="Busy end")
    if parse_utc_timestamp(start) >= parse_utc_timestamp(end):
        raise CalendarValidationError("Busy window end must be after start.")
    return FreeBusyWindow(
        calendar_id=validate_calendar_id(window.calendar_id),
        start=start,
        end=end,
    )


def intervals_overlap(
    start_a: str,
    end_a: str,
    start_b: str,
    end_b: str,
) -> bool:
    """Return True when half-open intervals [a,b) and [c,d) overlap."""
    a0 = parse_utc_timestamp(start_a)
    a1 = parse_utc_timestamp(end_a)
    b0 = parse_utc_timestamp(start_b)
    b1 = parse_utc_timestamp(end_b)
    return a0 < b1 and b0 < a1


def find_conflicts(
    *,
    start: str,
    end: str,
    busy_windows: list[FreeBusyWindow],
) -> list[FreeBusyWindow]:
    """Return busy windows that overlap the proposed half-open interval."""
    conflicts: list[FreeBusyWindow] = []
    for window in busy_windows:
        if intervals_overlap(start, end, window.start, window.end):
            conflicts.append(window)
    return conflicts


def local_wall_pair_to_utc(
    *,
    start_local: str,
    end_local: str,
    timezone_name: str,
) -> tuple[str, str]:
    """Convert local wall start/end in a calendar timezone to UTC ISO Z."""
    validate_iana_timezone(timezone_name)
    start_wall = parse_local_wall_datetime(start_local, field_name="Start time")
    end_wall = parse_local_wall_datetime(end_local, field_name="End time")
    start_utc = local_wall_to_utc_iso(start_wall, timezone_name)
    end_utc = local_wall_to_utc_iso(end_wall, timezone_name)
    if parse_utc_timestamp(start_utc) >= parse_utc_timestamp(end_utc):
        raise CalendarValidationError("End time must be after start time.")
    return start_utc, end_utc


def compute_proposal_fingerprint(
    *,
    proposal_id: str,
    account_id: str,
    calendar_id: str,
    operation: str,
    event_id: str | None,
    client_event_id: str | None,
    normalized_payload: Mapping[str, str],
) -> str:
    """Return a SHA-256 fingerprint bound to proposal identity and write intent."""
    payload = {
        "proposal_id": proposal_id,
        "account_id": account_id,
        "calendar_id": calendar_id,
        "operation": operation,
        "event_id": event_id,
        "client_event_id": client_event_id,
        "normalized_payload": dict(sorted(normalized_payload.items())),
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def create_change_proposal(
    *,
    account_id: str,
    calendar_id: str,
    operation: str,
    event_id: str | None,
    client_event_id: str | None,
    normalized_payload: Mapping[str, str],
    remote_etag: str | None,
    clock: CalendarClock | None = None,
    ttl_seconds: int = CALENDAR_PROPOSAL_TTL_SECONDS,
) -> CalendarChangeProposal:
    """Create one pending validated change proposal."""
    active_clock = clock or SystemCalendarClock()
    if ttl_seconds < 1:
        raise CalendarValidationError("Proposal TTL must be at least 1 second.")
    proposal_id = str(uuid4())
    created_at = active_clock.utc_now_iso()
    expires_at = (
        parse_utc_timestamp(created_at) + timedelta(seconds=ttl_seconds)
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")
    cleaned_operation = validate_controlled_value(
        operation,
        field_name="Proposal operation",
        allowed=CALENDAR_PROPOSAL_OPERATIONS,
    )
    cleaned_event_id = (
        validate_event_id(event_id) if event_id is not None else None
    )
    cleaned_client_id = (
        validate_google_event_id(client_event_id)
        if client_event_id is not None
        else None
    )
    if cleaned_operation == "create":
        if cleaned_event_id is not None:
            raise CalendarValidationError("Create proposals must not set event_id.")
        if cleaned_client_id is None:
            raise CalendarValidationError(
                "Create proposals require client_event_id."
            )
    else:
        if cleaned_event_id is None:
            raise CalendarValidationError(
                f"{cleaned_operation} proposals require event_id."
            )
        if cleaned_client_id is not None:
            raise CalendarValidationError(
                f"{cleaned_operation} proposals must not set client_event_id."
            )

    cleaned_payload = _validate_normalized_payload(
        cleaned_operation,
        normalized_payload,
    )
    fingerprint = compute_proposal_fingerprint(
        proposal_id=proposal_id,
        account_id=validate_uuid(account_id, field_name="Account ID"),
        calendar_id=validate_calendar_id(calendar_id),
        operation=cleaned_operation,
        event_id=cleaned_event_id,
        client_event_id=cleaned_client_id,
        normalized_payload=cleaned_payload,
    )
    return validate_change_proposal(
        CalendarChangeProposal(
            proposal_id=proposal_id,
            account_id=validate_uuid(account_id, field_name="Account ID"),
            calendar_id=validate_calendar_id(calendar_id),
            operation=cleaned_operation,  # type: ignore[arg-type]
            event_id=cleaned_event_id,
            client_event_id=cleaned_client_id,
            normalized_payload=cleaned_payload,
            fingerprint=fingerprint,
            remote_etag=(
                require_non_blank_text(
                    remote_etag,
                    field_name="Remote ETag",
                    max_length=200,
                )
                if remote_etag is not None and remote_etag.strip()
                else None
            ),
            status="pending",
            created_at=created_at,
            expires_at=expires_at,
        )
    )


def validate_change_proposal(
    proposal: CalendarChangeProposal,
) -> CalendarChangeProposal:
    """Validate one change proposal record."""
    operation = validate_controlled_value(
        proposal.operation,
        field_name="Proposal operation",
        allowed=CALENDAR_PROPOSAL_OPERATIONS,
    )
    status = validate_controlled_value(
        proposal.status,
        field_name="Proposal status",
        allowed=CALENDAR_PROPOSAL_STATUSES,
    )
    event_id = (
        validate_event_id(proposal.event_id)
        if proposal.event_id is not None
        else None
    )
    client_event_id = (
        validate_google_event_id(proposal.client_event_id)
        if proposal.client_event_id is not None
        else None
    )
    payload = _validate_normalized_payload(operation, proposal.normalized_payload)
    expected = compute_proposal_fingerprint(
        proposal_id=validate_uuid(proposal.proposal_id, field_name="Proposal ID"),
        account_id=validate_uuid(proposal.account_id, field_name="Account ID"),
        calendar_id=validate_calendar_id(proposal.calendar_id),
        operation=operation,
        event_id=event_id,
        client_event_id=client_event_id,
        normalized_payload=payload,
    )
    if proposal.fingerprint != expected:
        raise CalendarValidationError("Proposal fingerprint is invalid.")
    return CalendarChangeProposal(
        proposal_id=validate_uuid(proposal.proposal_id, field_name="Proposal ID"),
        account_id=validate_uuid(proposal.account_id, field_name="Account ID"),
        calendar_id=validate_calendar_id(proposal.calendar_id),
        operation=operation,  # type: ignore[arg-type]
        event_id=event_id,
        client_event_id=client_event_id,
        normalized_payload=payload,
        fingerprint=expected,
        remote_etag=(
            require_non_blank_text(
                proposal.remote_etag,
                field_name="Remote ETag",
                max_length=200,
            )
            if proposal.remote_etag is not None
            else None
        ),
        status=status,  # type: ignore[arg-type]
        created_at=validate_utc_timestamp(
            proposal.created_at,
            field_name="Proposal created_at",
        ),
        expires_at=validate_utc_timestamp(
            proposal.expires_at,
            field_name="Proposal expires_at",
        ),
    )


def replace_change_proposal(
    proposal: CalendarChangeProposal,
    *,
    status: str,
) -> CalendarChangeProposal:
    """Return a validated proposal with an updated status."""
    return validate_change_proposal(
        CalendarChangeProposal(
            proposal_id=proposal.proposal_id,
            account_id=proposal.account_id,
            calendar_id=proposal.calendar_id,
            operation=proposal.operation,
            event_id=proposal.event_id,
            client_event_id=proposal.client_event_id,
            normalized_payload=dict(proposal.normalized_payload),
            fingerprint=proposal.fingerprint,
            remote_etag=proposal.remote_etag,
            status=status,  # type: ignore[arg-type]
            created_at=proposal.created_at,
            expires_at=proposal.expires_at,
        )
    )


def is_proposal_expired(
    proposal: CalendarChangeProposal,
    *,
    now_utc_iso: str,
) -> bool:
    """Derive expiry for pending proposals only."""
    if proposal.status != "pending":
        return False
    return parse_utc_timestamp(proposal.expires_at) <= parse_utc_timestamp(now_utc_iso)


def create_calendar_audit_entry(
    *,
    action: str,
    account_id: str | None = None,
    proposal_id: str | None = None,
    calendar_id: str | None = None,
    event_id: str | None = None,
    operation: str | None = None,
    from_status: str | None = None,
    to_status: str | None = None,
    error_category: str | None = None,
    clock: CalendarClock | None = None,
) -> CalendarAuditEntry:
    """Create one validated calendar audit entry."""
    active_clock = clock or SystemCalendarClock()
    return validate_calendar_audit_entry(
        CalendarAuditEntry(
            audit_id=str(uuid4()),
            action=action,  # type: ignore[arg-type]
            occurred_at=active_clock.utc_now_iso(),
            account_id=account_id,
            proposal_id=proposal_id,
            calendar_id=calendar_id,
            event_id=event_id,
            operation=operation,
            from_status=from_status,
            to_status=to_status,
            error_category=error_category,
        )
    )


def validate_calendar_audit_entry(entry: CalendarAuditEntry) -> CalendarAuditEntry:
    """Validate one calendar audit entry."""
    action = validate_controlled_value(
        entry.action,
        field_name="Audit action",
        allowed=CALENDAR_AUDIT_ACTIONS,
    )
    error_category = None
    if entry.error_category is not None:
        error_category = validate_controlled_value(
            entry.error_category,
            field_name="Error category",
            allowed=CALENDAR_ERROR_CATEGORIES,
        )
    if operation := entry.operation:
        if operation not in CALENDAR_PROPOSAL_OPERATIONS:
            raise CalendarValidationError("Audit operation is invalid.")
    return CalendarAuditEntry(
        audit_id=validate_uuid(entry.audit_id, field_name="Audit ID"),
        action=action,  # type: ignore[arg-type]
        occurred_at=validate_utc_timestamp(
            entry.occurred_at,
            field_name="Audit occurred_at",
        ),
        account_id=(
            validate_uuid(entry.account_id, field_name="Account ID")
            if entry.account_id is not None
            else None
        ),
        proposal_id=(
            validate_uuid(entry.proposal_id, field_name="Proposal ID")
            if entry.proposal_id is not None
            else None
        ),
        calendar_id=(
            validate_calendar_id(entry.calendar_id)
            if entry.calendar_id is not None
            else None
        ),
        event_id=(
            validate_event_id(entry.event_id)
            if entry.event_id is not None
            else None
        ),
        operation=operation,
        from_status=entry.from_status,
        to_status=entry.to_status,
        error_category=error_category,
    )


def _validate_normalized_payload(
    operation: str,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise CalendarValidationError("normalized_payload must be an object.")
    cleaned = {str(key): str(value) for key, value in payload.items()}
    if operation == "create":
        required = {"title", "start_utc", "end_utc", "timezone"}
        if set(cleaned) != required:
            raise CalendarValidationError(
                "Create payload must contain only title, start_utc, end_utc, timezone."
            )
        cleaned["title"] = validate_event_title(cleaned["title"])
        cleaned["timezone"] = validate_iana_timezone(cleaned["timezone"])
        cleaned["start_utc"] = validate_utc_timestamp(
            cleaned["start_utc"],
            field_name="Start",
        )
        cleaned["end_utc"] = validate_utc_timestamp(
            cleaned["end_utc"],
            field_name="End",
        )
        if parse_utc_timestamp(cleaned["start_utc"]) >= parse_utc_timestamp(
            cleaned["end_utc"]
        ):
            raise CalendarValidationError("End time must be after start time.")
        return cleaned
    if operation == "reschedule":
        required = {"start_utc", "end_utc", "timezone"}
        if set(cleaned) != required:
            raise CalendarValidationError(
                "Reschedule payload must contain only start_utc, end_utc, timezone."
            )
        cleaned["timezone"] = validate_iana_timezone(cleaned["timezone"])
        cleaned["start_utc"] = validate_utc_timestamp(
            cleaned["start_utc"],
            field_name="Start",
        )
        cleaned["end_utc"] = validate_utc_timestamp(
            cleaned["end_utc"],
            field_name="End",
        )
        if parse_utc_timestamp(cleaned["start_utc"]) >= parse_utc_timestamp(
            cleaned["end_utc"]
        ):
            raise CalendarValidationError("End time must be after start time.")
        return cleaned
    if cleaned:
        raise CalendarValidationError("Cancel payload must be empty.")
    return {}


def _validate_all_day_date(value: str, *, field_name: str) -> str:
    cleaned = require_non_blank_text(value, field_name=field_name, max_length=32)
    try:
        date.fromisoformat(cleaned)
    except ValueError as error:
        raise CalendarValidationError(
            f"{field_name} must be an ISO date YYYY-MM-DD."
        ) from error
    return cleaned


def format_local_wall_from_utc(utc_iso: str, timezone_name: str) -> str:
    """Format a UTC instant as local wall time in the given IANA zone."""
    local = parse_utc_timestamp(utc_iso).astimezone(ZoneInfo(timezone_name))
    return local.strftime(LOCAL_WALL_FORMAT)


# Re-export helpers used by calendar commands/services.
__all__ = [
    "CALENDAR_AUDIT_ACTIONS",
    "CALENDAR_ERROR_CATEGORIES",
    "CALENDAR_PROPOSAL_OPERATIONS",
    "CALENDAR_PROPOSAL_STATUSES",
    "CALENDAR_PROVIDER_IDS",
    "PRIMARY_CALENDAR_ID",
    "CalendarAccount",
    "CalendarAuditEntry",
    "CalendarChangeProposal",
    "CalendarClock",
    "CalendarError",
    "CalendarErrorCategory",
    "CalendarEvent",
    "CalendarInfo",
    "CalendarValidationError",
    "FreeBusyWindow",
    "ManualCalendarClock",
    "SystemCalendarClock",
    "compute_proposal_fingerprint",
    "create_calendar_account",
    "create_calendar_audit_entry",
    "create_change_proposal",
    "event_blocks_writes",
    "find_conflicts",
    "format_local_wall_from_utc",
    "generate_google_client_event_id",
    "intervals_overlap",
    "is_proposal_expired",
    "local_wall_pair_to_utc",
    "replace_calendar_account",
    "replace_change_proposal",
    "validate_calendar_account",
    "validate_calendar_audit_entry",
    "validate_calendar_event",
    "validate_calendar_id",
    "validate_calendar_info",
    "validate_change_proposal",
    "validate_event_id",
    "validate_event_title",
    "validate_freebusy_window",
    "validate_google_event_id",
    "validate_scopes_granted",
]
