"""Calendar service: OAuth lifecycle, reads, prepare/confirm writes.

Google remains authoritative for events. Local state stores account metadata,
proposals, and bounded audits only. Secrets stay in SecretStore.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import NoReturn

from google.oauth2.credentials import Credentials

from src.calendar_google import (
    GoogleCalendarProvider,
    credentials_from_refresh_token,
    revoke_google_refresh_token,
    run_google_oauth_flow,
    validate_google_oauth_client_file,
)
from src.calendar_models import (
    PRIMARY_CALENDAR_ID,
    CalendarAccount,
    CalendarChangeProposal,
    CalendarClock,
    CalendarError,
    CalendarEvent,
    CalendarInfo,
    CalendarValidationError,
    FreeBusyWindow,
    SystemCalendarClock,
    create_calendar_account,
    create_calendar_audit_entry,
    create_change_proposal,
    event_blocks_writes,
    find_conflicts,
    generate_google_client_event_id,
    is_proposal_expired,
    local_wall_pair_to_utc,
    replace_calendar_account,
    replace_change_proposal,
    validate_calendar_id,
    validate_event_id,
    validate_event_title,
)
from src.reminder import ReminderValidationError
from src.calendar_provider import CalendarProvider
from src.calendar_repository import CalendarRepository, CalendarStorageError
from src.config import (
    GOOGLE_CALENDAR_SCOPES,
    MAX_CALENDAR_EVENT_WINDOW_DAYS,
    MAX_CALENDAR_LIST_RESULTS,
)
from src.secret_store import (
    SecretStore,
    SecretStoreError,
    google_refresh_token_secret_key,
)
from src.tool_common import parse_utc_timestamp, validate_uuid

logger = logging.getLogger("ProjectCortana")

ProviderFactory = Callable[[Credentials], CalendarProvider]
OAuthFlowRunner = Callable[[Path], Credentials]


@dataclass(frozen=True)
class CalendarStatusView:
    """Bounded status projection for /status."""

    connection_state: str
    provider_id: str | None
    default_calendar_id: str | None
    pending_proposal_count: int
    unknown_outcome_count: int


class CalendarService:
    """Application service for Milestone 20 calendar integration."""

    def __init__(
        self,
        repository: CalendarRepository,
        secret_store: SecretStore,
        *,
        oauth_client_file: Path | None,
        clock: CalendarClock | None = None,
        provider_factory: ProviderFactory | None = None,
        oauth_flow_runner: OAuthFlowRunner | None = None,
    ) -> None:
        self._repository = repository
        self._secret_store = secret_store
        self._oauth_client_file = oauth_client_file
        self._clock = clock or SystemCalendarClock()
        self._provider_factory = provider_factory or (
            lambda credentials: GoogleCalendarProvider(credentials)
        )
        self._oauth_flow_runner = oauth_flow_runner or run_google_oauth_flow

    @property
    def clock(self) -> CalendarClock:
        return self._clock

    def connect(self) -> CalendarAccount:
        """Authorize one Google account via Desktop OAuth loopback flow."""
        existing = self._repository.get_account()
        if existing is not None and existing.status == "connected":
            raise CalendarError(
                "An account is already connected.",
                category="validation_error",
                user_message=(
                    "Cortana: A Google Calendar account is already connected. "
                    "Disconnect first with /calendar-disconnect."
                ),
            )
        client_file = self._require_oauth_client_file()
        credentials = self._oauth_flow_runner(client_file)
        refresh_token = credentials.refresh_token
        if not refresh_token:
            raise CalendarError(
                "Missing refresh token after OAuth.",
                category="auth_error",
                user_message=(
                    "Cortana: Google did not return a refresh token. "
                    "Revoke prior consent and reconnect."
                ),
            )
        display_label = "Google Calendar"
        account = create_calendar_account(
            display_label=display_label,
            default_calendar_id=PRIMARY_CALENDAR_ID,
            status="connected",
            scopes_granted=GOOGLE_CALENDAR_SCOPES,
            clock=self._clock,
        )
        secret_key = google_refresh_token_secret_key(account.account_id)
        try:
            self._secret_store.set_secret(secret_key, refresh_token)
        except SecretStoreError as error:
            raise CalendarError(
                str(error),
                category="secret_store_error",
                user_message=error.user_message,
            ) from error
        audit = create_calendar_audit_entry(
            action="connected",
            account_id=account.account_id,
            calendar_id=account.default_calendar_id,
            clock=self._clock,
        )
        try:
            saved = self._repository.save_account_with_audits(account, (audit,))
        except CalendarStorageError:
            try:
                self._secret_store.delete_secret(secret_key)
            except SecretStoreError:
                logger.error("Failed to roll back refresh token after save error")
            raise
        assert saved is not None
        return saved

    def disconnect(self) -> None:
        """Revoke (best effort), delete local secret, and remove account metadata."""
        account = self._repository.get_account()
        if account is None:
            raise CalendarError(
                "No calendar account is connected.",
                category="validation_error",
                user_message="Cortana: No Google Calendar account is connected.",
            )
        secret_key = google_refresh_token_secret_key(account.account_id)
        refresh_token: str | None = None
        try:
            refresh_token = self._secret_store.get_secret(secret_key)
        except SecretStoreError:
            logger.error("SecretStore get failed during disconnect")
        if refresh_token:
            try:
                revoke_google_refresh_token(refresh_token)
            except Exception as error:
                logger.error(
                    "Google token revoke raised error_type=%s",
                    type(error).__name__,
                )
        try:
            self._secret_store.delete_secret(secret_key)
        except SecretStoreError as error:
            raise CalendarError(
                str(error),
                category="secret_store_error",
                user_message=error.user_message,
            ) from error
        audit = create_calendar_audit_entry(
            action="disconnected",
            account_id=account.account_id,
            clock=self._clock,
        )
        self._repository.save_account_with_audits(None, (audit,))

    def set_default_calendar(self, calendar_id: str) -> CalendarAccount:
        """Set the stored default calendar after verifying it exists."""
        account = self._require_usable_account()
        cleaned = validate_calendar_id(calendar_id)
        provider = self._provider()
        calendars = provider.list_calendars()
        if cleaned != PRIMARY_CALENDAR_ID and all(
            item.calendar_id != cleaned for item in calendars
        ):
            raise CalendarError(
                "Calendar not found.",
                category="not_found",
                user_message=f"Cortana: No calendar found with ID '{cleaned}'.",
            )
        updated = replace_calendar_account(account, default_calendar_id=cleaned)
        audit = create_calendar_audit_entry(
            action="default_calendar_set",
            account_id=account.account_id,
            calendar_id=cleaned,
            clock=self._clock,
        )
        saved = self._repository.save_account_with_audits(updated, (audit,))
        assert saved is not None
        return saved

    def list_calendars(self) -> list[CalendarInfo]:
        self._require_usable_account()
        return self._provider().list_calendars()

    def list_events(
        self,
        calendar_id: str | None = None,
        *,
        limit: int = MAX_CALENDAR_LIST_RESULTS,
    ) -> list[CalendarEvent]:
        account = self._require_usable_account()
        resolved = self._resolve_calendar_id(calendar_id, account)
        if limit < 1 or limit > MAX_CALENDAR_LIST_RESULTS:
            raise CalendarError(
                "Invalid list limit.",
                category="validation_error",
                user_message=(
                    f"Cortana: Event list limit must be between 1 and "
                    f"{MAX_CALENDAR_LIST_RESULTS}."
                ),
            )
        now = parse_utc_timestamp(self._clock.utc_now_iso())
        time_min = now.isoformat(timespec="microseconds").replace("+00:00", "Z")
        time_max = (
            now + timedelta(days=MAX_CALENDAR_EVENT_WINDOW_DAYS)
        ).isoformat(timespec="microseconds").replace("+00:00", "Z")
        return self._provider().list_events(
            resolved,
            time_min_utc=time_min,
            time_max_utc=time_max,
            limit=limit,
        )

    def get_event(
        self,
        event_id: str,
        calendar_id: str | None = None,
    ) -> CalendarEvent:
        account = self._require_usable_account()
        resolved = self._resolve_calendar_id(calendar_id, account)
        return self._provider().get_event(resolved, validate_event_id(event_id))

    def get_freebusy(
        self,
        *,
        start_local: str,
        end_local: str,
        calendar_id: str | None = None,
    ) -> list[FreeBusyWindow]:
        account = self._require_usable_account()
        resolved = self._resolve_calendar_id(calendar_id, account)
        timezone_name = self._calendar_timezone(resolved)
        start_utc, end_utc = local_wall_pair_to_utc(
            start_local=start_local,
            end_local=end_local,
            timezone_name=timezone_name,
        )
        return self._provider().get_freebusy(
            [resolved],
            time_min_utc=start_utc,
            time_max_utc=end_utc,
        )

    def prepare_create(
        self,
        *,
        title: str,
        start_local: str,
        end_local: str,
        calendar_id: str | None = None,
    ) -> CalendarChangeProposal:
        account = self._require_usable_account()
        resolved = self._resolve_calendar_id(calendar_id, account)
        try:
            timezone_name = self._calendar_timezone(resolved)
            cleaned_title = validate_event_title(title)
            start_utc, end_utc = local_wall_pair_to_utc(
                start_local=start_local,
                end_local=end_local,
                timezone_name=timezone_name,
            )
        except (CalendarValidationError, ReminderValidationError) as error:
            raise CalendarError(
                str(error),
                category="validation_error",
                user_message=f"Cortana: {error}",
            ) from error
        self._assert_no_conflicts(resolved, start_utc, end_utc)
        client_event_id = generate_google_client_event_id()
        proposal = create_change_proposal(
            account_id=account.account_id,
            calendar_id=resolved,
            operation="create",
            event_id=None,
            client_event_id=client_event_id,
            normalized_payload={
                "title": cleaned_title,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "timezone": timezone_name,
            },
            remote_etag=None,
            clock=self._clock,
        )
        audit = create_calendar_audit_entry(
            action="create_prepared",
            account_id=account.account_id,
            proposal_id=proposal.proposal_id,
            calendar_id=resolved,
            event_id=client_event_id,
            operation="create",
            from_status=None,
            to_status="pending",
            clock=self._clock,
        )
        return self._repository.save_proposal_with_audits(
            proposal,
            (audit,),
            is_new=True,
        )

    def prepare_reschedule(
        self,
        *,
        event_id: str,
        start_local: str,
        end_local: str,
        calendar_id: str | None = None,
    ) -> CalendarChangeProposal:
        account = self._require_usable_account()
        resolved = self._resolve_calendar_id(calendar_id, account)
        try:
            timezone_name = self._calendar_timezone(resolved)
            start_utc, end_utc = local_wall_pair_to_utc(
                start_local=start_local,
                end_local=end_local,
                timezone_name=timezone_name,
            )
        except (CalendarValidationError, ReminderValidationError) as error:
            raise CalendarError(
                str(error),
                category="validation_error",
                user_message=f"Cortana: {error}",
            ) from error
        event = self._provider().get_event(resolved, validate_event_id(event_id))
        self._assert_writable_event(event)
        self._assert_no_conflicts(resolved, start_utc, end_utc)
        proposal = create_change_proposal(
            account_id=account.account_id,
            calendar_id=resolved,
            operation="reschedule",
            event_id=event.event_id,
            client_event_id=None,
            normalized_payload={
                "start_utc": start_utc,
                "end_utc": end_utc,
                "timezone": timezone_name,
            },
            remote_etag=event.etag,
            clock=self._clock,
        )
        audit = create_calendar_audit_entry(
            action="reschedule_prepared",
            account_id=account.account_id,
            proposal_id=proposal.proposal_id,
            calendar_id=resolved,
            event_id=event.event_id,
            operation="reschedule",
            from_status=None,
            to_status="pending",
            clock=self._clock,
        )
        return self._repository.save_proposal_with_audits(
            proposal,
            (audit,),
            is_new=True,
        )

    def prepare_cancel(
        self,
        *,
        event_id: str,
        calendar_id: str | None = None,
    ) -> CalendarChangeProposal:
        account = self._require_usable_account()
        resolved = self._resolve_calendar_id(calendar_id, account)
        event = self._provider().get_event(resolved, validate_event_id(event_id))
        self._assert_writable_event(event)
        proposal = create_change_proposal(
            account_id=account.account_id,
            calendar_id=resolved,
            operation="cancel",
            event_id=event.event_id,
            client_event_id=None,
            normalized_payload={},
            remote_etag=event.etag,
            clock=self._clock,
        )
        audit = create_calendar_audit_entry(
            action="cancel_prepared",
            account_id=account.account_id,
            proposal_id=proposal.proposal_id,
            calendar_id=resolved,
            event_id=event.event_id,
            operation="cancel",
            from_status=None,
            to_status="pending",
            clock=self._clock,
        )
        return self._repository.save_proposal_with_audits(
            proposal,
            (audit,),
            is_new=True,
        )

    def confirm(self, proposal_id: str) -> CalendarChangeProposal:
        """Execute or recover one proposal after explicit confirmation."""
        account = self._require_usable_account()
        try:
            cleaned_id = validate_uuid(proposal_id, field_name="Proposal ID")
        except Exception as error:
            raise CalendarError(
                "Invalid proposal ID.",
                category="validation_error",
                user_message="Cortana: Proposal ID must be a valid UUID.",
            ) from error
        proposal = self._repository.get_proposal(cleaned_id)
        if proposal is None:
            raise CalendarError(
                "Proposal not found.",
                category="not_found",
                user_message=f"Cortana: No calendar proposal found with ID '{cleaned_id}'.",
            )
        if proposal.account_id != account.account_id:
            raise CalendarError(
                "Proposal account mismatch.",
                category="validation_error",
                user_message="Cortana: That proposal belongs to a disconnected account.",
            )
        if proposal.status == "executed":
            raise CalendarError(
                "Proposal already executed.",
                category="validation_error",
                user_message="Cortana: That proposal was already executed.",
            )
        if proposal.status == "failed":
            raise CalendarError(
                "Proposal already failed.",
                category="validation_error",
                user_message=(
                    "Cortana: That proposal already failed. Prepare a new change."
                ),
            )
        if proposal.status == "pending" and is_proposal_expired(
            proposal,
            now_utc_iso=self._clock.utc_now_iso(),
        ):
            raise CalendarError(
                "Proposal expired.",
                category="validation_error",
                user_message=(
                    "Cortana: That proposal has expired. Prepare the change again."
                ),
            )
        if proposal.status not in {"pending", "unknown_outcome"}:
            raise CalendarError(
                "Proposal is not confirmable.",
                category="validation_error",
                user_message="Cortana: That proposal cannot be confirmed.",
            )

        # Fingerprint revalidation happens inside validate on replace; ensure payload intact.
        from src.calendar_models import compute_proposal_fingerprint

        expected = compute_proposal_fingerprint(
            proposal_id=proposal.proposal_id,
            account_id=proposal.account_id,
            calendar_id=proposal.calendar_id,
            operation=proposal.operation,
            event_id=proposal.event_id,
            client_event_id=proposal.client_event_id,
            normalized_payload=proposal.normalized_payload,
        )
        if expected != proposal.fingerprint:
            return self._fail_proposal(
                proposal,
                category="validation_error",
                message="Fingerprint mismatch.",
                user_message="Cortana: Proposal integrity check failed.",
            )

        if proposal.operation == "create":
            return self._confirm_create(proposal)
        if proposal.operation == "reschedule":
            return self._confirm_reschedule(proposal)
        return self._confirm_cancel(proposal)

    def status_view(self) -> CalendarStatusView:
        account = self._repository.get_account()
        proposals = self._repository.list_proposals()
        pending = sum(1 for item in proposals if item.status == "pending")
        unknown = sum(1 for item in proposals if item.status == "unknown_outcome")
        if account is None:
            return CalendarStatusView(
                connection_state="not_connected",
                provider_id=None,
                default_calendar_id=None,
                pending_proposal_count=pending,
                unknown_outcome_count=unknown,
            )
        return CalendarStatusView(
            connection_state=account.status,
            provider_id=account.provider_id,
            default_calendar_id=account.default_calendar_id,
            pending_proposal_count=pending,
            unknown_outcome_count=unknown,
        )

    def _confirm_create(
        self,
        proposal: CalendarChangeProposal,
    ) -> CalendarChangeProposal:
        assert proposal.client_event_id is not None
        provider = self._provider()
        payload = proposal.normalized_payload
        start_utc = payload["start_utc"]
        end_utc = payload["end_utc"]
        try:
            self._assert_no_conflicts(proposal.calendar_id, start_utc, end_utc)
        except CalendarError as error:
            if error.category == "conflict":
                return self._fail_proposal(
                    proposal,
                    category="conflict",
                    message=str(error),
                    user_message=error.user_message,
                )
            raise

        if proposal.status == "unknown_outcome":
            existing = self._try_get_event(
                provider,
                proposal.calendar_id,
                proposal.client_event_id,
            )
            if existing is not None and self._event_matches_create(existing, proposal):
                return self._mark_executed(
                    proposal,
                    action="created",
                    event_id=existing.event_id,
                )

        try:
            created = provider.create_event(
                proposal.calendar_id,
                event_id=proposal.client_event_id,
                title=payload["title"],
                start_utc=start_utc,
                end_utc=end_utc,
                timezone_name=payload["timezone"],
            )
        except CalendarError as error:
            if error.category == "conflict":
                # Duplicate ID may mean prior unknown create succeeded.
                existing = self._try_get_event(
                    provider,
                    proposal.calendar_id,
                    proposal.client_event_id,
                )
                if existing is not None and self._event_matches_create(
                    existing, proposal
                ):
                    return self._mark_executed(
                        proposal,
                        action="created",
                        event_id=existing.event_id,
                    )
            if error.category in {"network_error", "server_error"}:
                return self._mark_unknown(proposal, error.category)
            return self._fail_proposal(
                proposal,
                category=error.category,
                message=str(error),
                user_message=error.user_message,
            )
        return self._mark_executed(
            proposal,
            action="created",
            event_id=created.event_id,
        )

    def _confirm_reschedule(
        self,
        proposal: CalendarChangeProposal,
    ) -> CalendarChangeProposal:
        assert proposal.event_id is not None
        provider = self._provider()
        payload = proposal.normalized_payload
        start_utc = payload["start_utc"]
        end_utc = payload["end_utc"]

        current = self._try_get_event(provider, proposal.calendar_id, proposal.event_id)
        if current is None:
            return self._fail_proposal(
                proposal,
                category="not_found",
                message="Event missing before reschedule.",
                user_message=(
                    "Cortana: The event was deleted or moved remotely. "
                    "Prepare again."
                ),
            )
        if event_blocks_writes(current):
            return self._fail_proposal(
                proposal,
                category="validation_error",
                message="Event became non-writable.",
                user_message=(
                    "Cortana: That event cannot be modified by Milestone 20 "
                    "(recurring or all-day)."
                ),
            )
        if proposal.remote_etag and current.etag and current.etag != proposal.remote_etag:
            if self._event_matches_reschedule(current, proposal):
                return self._mark_executed(
                    proposal,
                    action="rescheduled",
                    event_id=current.event_id,
                )
            return self._fail_proposal(
                proposal,
                category="conflict",
                message="ETag changed.",
                user_message=(
                    "Cortana: The calendar event changed remotely. "
                    "Prepare the change again."
                ),
            )
        if self._event_matches_reschedule(current, proposal):
            return self._mark_executed(
                proposal,
                action="rescheduled",
                event_id=current.event_id,
            )

        try:
            self._assert_no_conflicts(proposal.calendar_id, start_utc, end_utc)
        except CalendarError as error:
            if error.category == "conflict":
                return self._fail_proposal(
                    proposal,
                    category="conflict",
                    message=str(error),
                    user_message=error.user_message,
                )
            raise
        try:
            updated = provider.update_event(
                proposal.calendar_id,
                proposal.event_id,
                start_utc=start_utc,
                end_utc=end_utc,
                timezone_name=payload["timezone"],
                etag=proposal.remote_etag,
            )
        except CalendarError as error:
            if error.category in {"network_error", "server_error"}:
                return self._mark_unknown(proposal, error.category)
            return self._fail_proposal(
                proposal,
                category=error.category,
                message=str(error),
                user_message=error.user_message,
            )
        return self._mark_executed(
            proposal,
            action="rescheduled",
            event_id=updated.event_id,
        )

    def _confirm_cancel(
        self,
        proposal: CalendarChangeProposal,
    ) -> CalendarChangeProposal:
        assert proposal.event_id is not None
        provider = self._provider()
        current = self._try_get_event(provider, proposal.calendar_id, proposal.event_id)
        if current is None:
            return self._mark_executed(
                proposal,
                action="cancelled",
                event_id=proposal.event_id,
            )
        if event_blocks_writes(current):
            return self._fail_proposal(
                proposal,
                category="validation_error",
                message="Event became non-writable.",
                user_message=(
                    "Cortana: That event cannot be modified by Milestone 20 "
                    "(recurring or all-day)."
                ),
            )
        if proposal.remote_etag and current.etag and current.etag != proposal.remote_etag:
            return self._fail_proposal(
                proposal,
                category="conflict",
                message="ETag changed.",
                user_message=(
                    "Cortana: The calendar event changed remotely. "
                    "Prepare the change again."
                ),
            )
        try:
            provider.delete_event(
                proposal.calendar_id,
                proposal.event_id,
                etag=proposal.remote_etag,
            )
        except CalendarError as error:
            if error.category == "not_found":
                return self._mark_executed(
                    proposal,
                    action="cancelled",
                    event_id=proposal.event_id,
                )
            if error.category in {"network_error", "server_error"}:
                return self._mark_unknown(proposal, error.category)
            return self._fail_proposal(
                proposal,
                category=error.category,
                message=str(error),
                user_message=error.user_message,
            )
        return self._mark_executed(
            proposal,
            action="cancelled",
            event_id=proposal.event_id,
        )

    def _mark_executed(
        self,
        proposal: CalendarChangeProposal,
        *,
        action: str,
        event_id: str,
    ) -> CalendarChangeProposal:
        updated = replace_change_proposal(proposal, status="executed")
        audit = create_calendar_audit_entry(
            action=action,
            account_id=proposal.account_id,
            proposal_id=proposal.proposal_id,
            calendar_id=proposal.calendar_id,
            event_id=event_id,
            operation=proposal.operation,
            from_status=proposal.status,
            to_status="executed",
            clock=self._clock,
        )
        return self._repository.save_proposal_with_audits(updated, (audit,))

    def _mark_unknown(
        self,
        proposal: CalendarChangeProposal,
        category: str,
    ) -> NoReturn:
        if proposal.status != "unknown_outcome":
            updated = replace_change_proposal(proposal, status="unknown_outcome")
            audit = create_calendar_audit_entry(
                action="proposal_unknown_outcome",
                account_id=proposal.account_id,
                proposal_id=proposal.proposal_id,
                calendar_id=proposal.calendar_id,
                event_id=proposal.event_id or proposal.client_event_id,
                operation=proposal.operation,
                from_status=proposal.status,
                to_status="unknown_outcome",
                error_category=category,
                clock=self._clock,
            )
            self._repository.save_proposal_with_audits(updated, (audit,))
        raise CalendarError(
            "Provider outcome uncertain.",
            category=category,  # type: ignore[arg-type]
            user_message=(
                "Cortana: The calendar provider outcome is uncertain. "
                f"Confirm again with /calendar-confirm {proposal.proposal_id}."
            ),
        )

    def _fail_proposal(
        self,
        proposal: CalendarChangeProposal,
        *,
        category: str,
        message: str,
        user_message: str,
    ) -> NoReturn:
        updated = replace_change_proposal(proposal, status="failed")
        audit = create_calendar_audit_entry(
            action="proposal_failed",
            account_id=proposal.account_id,
            proposal_id=proposal.proposal_id,
            calendar_id=proposal.calendar_id,
            event_id=proposal.event_id or proposal.client_event_id,
            operation=proposal.operation,
            from_status=proposal.status,
            to_status="failed",
            error_category=category,
            clock=self._clock,
        )
        self._repository.save_proposal_with_audits(updated, (audit,))
        raise CalendarError(
            message,
            category=category,  # type: ignore[arg-type]
            user_message=user_message,
        )

    def _event_matches_create(
        self,
        event: CalendarEvent,
        proposal: CalendarChangeProposal,
    ) -> bool:
        payload = proposal.normalized_payload
        return (
            not event.is_all_day
            and event.title == payload["title"]
            and event.start == payload["start_utc"]
            and event.end == payload["end_utc"]
        )

    def _event_matches_reschedule(
        self,
        event: CalendarEvent,
        proposal: CalendarChangeProposal,
    ) -> bool:
        payload = proposal.normalized_payload
        return (
            not event.is_all_day
            and event.start == payload["start_utc"]
            and event.end == payload["end_utc"]
        )

    def _try_get_event(
        self,
        provider: CalendarProvider,
        calendar_id: str,
        event_id: str,
    ) -> CalendarEvent | None:
        try:
            return provider.get_event(calendar_id, event_id)
        except CalendarError as error:
            if error.category == "not_found":
                return None
            raise

    def _assert_no_conflicts(
        self,
        calendar_id: str,
        start_utc: str,
        end_utc: str,
    ) -> None:
        busy = self._provider().get_freebusy(
            [calendar_id],
            time_min_utc=start_utc,
            time_max_utc=end_utc,
        )
        conflicts = find_conflicts(
            start=start_utc,
            end=end_utc,
            busy_windows=busy,
        )
        if conflicts:
            raise CalendarError(
                "Conflict detected.",
                category="conflict",
            )

    def _assert_writable_event(self, event: CalendarEvent) -> None:
        if event_blocks_writes(event):
            raise CalendarError(
                "Recurring or all-day writes are not supported.",
                category="validation_error",
                user_message=(
                    "Cortana: Milestone 20 supports timed non-recurring writes only. "
                    "Recurring and all-day changes are deferred."
                ),
            )

    def _calendar_timezone(self, calendar_id: str) -> str:
        calendars = self._provider().list_calendars()
        for item in calendars:
            if item.calendar_id == calendar_id or (
                calendar_id == PRIMARY_CALENDAR_ID and item.primary
            ):
                return item.timezone
        if calendar_id == PRIMARY_CALENDAR_ID:
            # Fall back to fetching via list using primary alias match.
            for item in calendars:
                if item.primary:
                    return item.timezone
        raise CalendarError(
            "Calendar timezone unavailable.",
            category="validation_error",
            user_message=(
                "Cortana: The selected calendar timezone is unavailable or invalid."
            ),
        )

    def _resolve_calendar_id(
        self,
        calendar_id: str | None,
        account: CalendarAccount,
    ) -> str:
        if calendar_id is None or calendar_id.strip() == "" or calendar_id.strip() == "-":
            return account.default_calendar_id
        return validate_calendar_id(calendar_id)

    def _require_usable_account(self) -> CalendarAccount:
        account = self._repository.get_account()
        if account is None:
            raise CalendarError(
                "No calendar account connected.",
                category="validation_error",
                user_message=(
                    "Cortana: No Google Calendar account is connected. "
                    "Use /calendar-connect."
                ),
            )
        if account.status == "needs_reauth":
            raise CalendarError(
                "Account needs reauthorization.",
                category="auth_error",
                user_message=(
                    "Cortana: Calendar authorization must be renewed. "
                    "Use /calendar-disconnect then /calendar-connect."
                ),
            )
        return account

    def _require_oauth_client_file(self) -> Path:
        if self._oauth_client_file is None:
            raise CalendarError(
                "OAuth client file is not configured.",
                category="validation_error",
                user_message=(
                    "Cortana: Set CORTANA_GOOGLE_OAUTH_CLIENT_FILE to your Google "
                    "Desktop OAuth client JSON path."
                ),
            )
        return validate_google_oauth_client_file(self._oauth_client_file)

    def _provider(self) -> CalendarProvider:
        account = self._require_usable_account()
        client_file = self._require_oauth_client_file()
        secret_key = google_refresh_token_secret_key(account.account_id)
        try:
            refresh_token = self._secret_store.get_secret(secret_key)
        except SecretStoreError as error:
            self._mark_needs_reauth(account)
            raise CalendarError(
                str(error),
                category="secret_store_error",
                user_message=error.user_message,
            ) from error
        if not refresh_token:
            self._mark_needs_reauth(account)
            raise CalendarError(
                "Refresh token missing.",
                category="auth_error",
            )
        try:
            credentials = credentials_from_refresh_token(
                client_secrets_file=client_file,
                refresh_token=refresh_token,
            )
        except CalendarError as error:
            if error.category == "auth_error":
                self._mark_needs_reauth(account)
            raise
        return self._provider_factory(credentials)

    def _mark_needs_reauth(self, account: CalendarAccount) -> None:
        updated = replace_calendar_account(account, status="needs_reauth")
        try:
            self._repository.save_account_with_audits(updated, ())
        except CalendarStorageError:
            logger.error("Failed to persist needs_reauth status")
