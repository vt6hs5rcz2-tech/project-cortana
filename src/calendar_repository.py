"""Calendar control repository with atomic JSON persistence.

Stores one optional account, proposals, and bounded audit entries.
Never stores OAuth tokens or provider event catalogs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.calendar_models import (
    CalendarAccount,
    CalendarAuditEntry,
    CalendarChangeProposal,
    CalendarValidationError,
    validate_calendar_account,
    validate_calendar_audit_entry,
    validate_change_proposal,
)
from src.config import (
    CALENDAR_REPOSITORY_SCHEMA_VERSION,
    MAX_CALENDAR_AUDIT_ENTRIES,
    MAX_CALENDAR_PROPOSALS,
)

logger = logging.getLogger("ProjectCortana")

CALENDAR_LOAD_ERROR_MESSAGE = (
    "Cortana: Calendar repository data could not be loaded safely. "
    "The existing repository file was left unchanged for inspection."
)
CALENDAR_SAVE_ERROR_MESSAGE = (
    "Cortana: Calendar repository data could not be updated safely."
)
CALENDAR_CAPACITY_ERROR_MESSAGE = (
    "Cortana: Calendar proposal storage limit reached. "
    "Confirm, wait for expiry, or disconnect before preparing more changes."
)

PERSISTED_ROOT_KEYS: frozenset[str] = frozenset(
    {"version", "account", "proposals", "audit_entries"}
)
PERSISTED_ACCOUNT_KEYS: frozenset[str] = frozenset(
    {
        "account_id",
        "provider_id",
        "display_label",
        "status",
        "default_calendar_id",
        "connected_at",
        "scopes_granted",
    }
)
PERSISTED_PROPOSAL_KEYS: frozenset[str] = frozenset(
    {
        "proposal_id",
        "account_id",
        "calendar_id",
        "operation",
        "event_id",
        "client_event_id",
        "normalized_payload",
        "fingerprint",
        "remote_etag",
        "status",
        "created_at",
        "expires_at",
    }
)
PERSISTED_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "audit_id",
        "action",
        "occurred_at",
        "account_id",
        "proposal_id",
        "calendar_id",
        "event_id",
        "operation",
        "from_status",
        "to_status",
        "error_category",
    }
)
FORBIDDEN_PERSISTED_KEYS: frozenset[str] = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "authorization",
        "password",
        "secret",
        "api_key",
    }
)


class CalendarStorageError(RuntimeError):
    """Raised when calendar repository data cannot be loaded or saved safely."""

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or CALENDAR_LOAD_ERROR_MESSAGE


class CalendarRepository(Protocol):
    """Persistence boundary for calendar control state."""

    def get_account(self) -> CalendarAccount | None:
        """Return the connected account, or None."""

    def get_proposal(self, proposal_id: str) -> CalendarChangeProposal | None:
        """Return one proposal by ID, or None."""

    def list_proposals(self) -> list[CalendarChangeProposal]:
        """Return proposals in deterministic storage order."""

    def list_audit_entries(self) -> list[CalendarAuditEntry]:
        """Return audit entries in append order."""

    def save_account_with_audits(
        self,
        account: CalendarAccount | None,
        audit_entries: Sequence[CalendarAuditEntry],
    ) -> CalendarAccount | None:
        """Atomically replace the account and append audits."""

    def save_proposal_with_audits(
        self,
        proposal: CalendarChangeProposal,
        audit_entries: Sequence[CalendarAuditEntry],
        *,
        is_new: bool = False,
    ) -> CalendarChangeProposal:
        """Atomically persist one proposal update plus audits."""


@dataclass
class _CalendarRepositoryState:
    """In-memory snapshot of the calendar control repository."""

    account: CalendarAccount | None = None
    proposals: dict[str, CalendarChangeProposal] = field(default_factory=dict)
    proposal_order: list[str] = field(default_factory=list)
    audit_entries: list[CalendarAuditEntry] = field(default_factory=list)


class JsonCalendarRepository:
    """JSON-backed calendar control store with atomic writes."""

    def __init__(
        self,
        file_path: Path,
        *,
        max_proposals: int = MAX_CALENDAR_PROPOSALS,
        max_audit_entries: int = MAX_CALENDAR_AUDIT_ENTRIES,
    ) -> None:
        if max_proposals < 1:
            raise CalendarValidationError("max_proposals must be at least 1.")
        if max_audit_entries < 1:
            raise CalendarValidationError("max_audit_entries must be at least 1.")
        self._file_path = file_path
        self._max_proposals = max_proposals
        self._max_audit_entries = max_audit_entries
        self._state: _CalendarRepositoryState | None = None
        self._load_error: CalendarStorageError | None = None
        self.atomic_replace_count = 0

    def get_account(self) -> CalendarAccount | None:
        return self._ensure_state().account

    def get_proposal(self, proposal_id: str) -> CalendarChangeProposal | None:
        return self._ensure_state().proposals.get(proposal_id.strip())

    def list_proposals(self) -> list[CalendarChangeProposal]:
        state = self._ensure_state()
        return [
            state.proposals[proposal_id]
            for proposal_id in state.proposal_order
            if proposal_id in state.proposals
        ]

    def list_audit_entries(self) -> list[CalendarAuditEntry]:
        return list(self._ensure_state().audit_entries)

    def save_account_with_audits(
        self,
        account: CalendarAccount | None,
        audit_entries: Sequence[CalendarAuditEntry],
    ) -> CalendarAccount | None:
        state = self._clone_state(self._ensure_state())
        validated_account = (
            validate_calendar_account(account) if account is not None else None
        )
        state.account = validated_account
        for entry in audit_entries:
            state.audit_entries.append(validate_calendar_audit_entry(entry))
        self._enforce_audit_retention(state)
        self._persist(state)
        return validated_account

    def save_proposal_with_audits(
        self,
        proposal: CalendarChangeProposal,
        audit_entries: Sequence[CalendarAuditEntry],
        *,
        is_new: bool = False,
    ) -> CalendarChangeProposal:
        state = self._clone_state(self._ensure_state())
        validated = validate_change_proposal(proposal)
        exists = validated.proposal_id in state.proposals
        if is_new and exists:
            raise CalendarStorageError(
                "Proposal already exists.",
                user_message=CALENDAR_SAVE_ERROR_MESSAGE,
            )
        if not is_new and not exists:
            raise CalendarStorageError(
                "Proposal does not exist.",
                user_message=CALENDAR_SAVE_ERROR_MESSAGE,
            )
        if is_new:
            active_count = sum(
                1
                for item in state.proposals.values()
                if item.status in {"pending", "unknown_outcome"}
            )
            if active_count >= self._max_proposals:
                raise CalendarStorageError(
                    "Proposal capacity reached.",
                    user_message=CALENDAR_CAPACITY_ERROR_MESSAGE,
                )
            # Bound total stored proposals by pruning oldest terminal ones first.
            self._enforce_proposal_retention(state, reserving=1)
            state.proposals[validated.proposal_id] = validated
            state.proposal_order.append(validated.proposal_id)
        else:
            state.proposals[validated.proposal_id] = validated
        for entry in audit_entries:
            state.audit_entries.append(validate_calendar_audit_entry(entry))
        self._enforce_audit_retention(state)
        self._persist(state)
        return validated

    def _ensure_state(self) -> _CalendarRepositoryState:
        if self._load_error is not None:
            raise self._load_error
        if self._state is not None:
            return self._state
        if not self._file_path.exists():
            self._state = _CalendarRepositoryState()
            return self._state
        try:
            raw = self._file_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self._load_error = CalendarStorageError(
                f"Calendar repository load failed: {type(error).__name__}",
                user_message=CALENDAR_LOAD_ERROR_MESSAGE,
            )
            logger.error(
                "Calendar repository load failed error_type=%s",
                type(error).__name__,
            )
            raise self._load_error from error
        try:
            self._state = self._parse_payload(payload)
        except (CalendarValidationError, CalendarStorageError, TypeError, ValueError) as error:
            self._load_error = CalendarStorageError(
                f"Calendar repository parse failed: {type(error).__name__}",
                user_message=CALENDAR_LOAD_ERROR_MESSAGE,
            )
            logger.error(
                "Calendar repository parse failed error_type=%s",
                type(error).__name__,
            )
            raise self._load_error from error
        return self._state

    def _parse_payload(self, payload: object) -> _CalendarRepositoryState:
        if not isinstance(payload, dict):
            raise CalendarStorageError("Root payload must be an object.")
        self._assert_no_forbidden_keys(payload)
        unknown = set(payload) - PERSISTED_ROOT_KEYS
        if unknown:
            raise CalendarStorageError(f"Unexpected root keys: {sorted(unknown)}")
        version = payload.get("version")
        if version != CALENDAR_REPOSITORY_SCHEMA_VERSION:
            raise CalendarStorageError("Unsupported calendar repository version.")
        account_raw = payload.get("account")
        account = None
        if account_raw is not None:
            if not isinstance(account_raw, dict):
                raise CalendarStorageError("Account must be an object or null.")
            self._assert_exact_keys(account_raw, PERSISTED_ACCOUNT_KEYS, "account")
            scopes = account_raw.get("scopes_granted")
            if not isinstance(scopes, list):
                raise CalendarStorageError("scopes_granted must be a list.")
            account = validate_calendar_account(
                CalendarAccount(
                    account_id=str(account_raw["account_id"]),
                    provider_id=str(account_raw["provider_id"]),  # type: ignore[arg-type]
                    display_label=str(account_raw["display_label"]),
                    status=str(account_raw["status"]),  # type: ignore[arg-type]
                    default_calendar_id=str(account_raw["default_calendar_id"]),
                    connected_at=str(account_raw["connected_at"]),
                    scopes_granted=tuple(str(item) for item in scopes),
                )
            )
        proposals_raw = payload.get("proposals")
        if not isinstance(proposals_raw, list):
            raise CalendarStorageError("proposals must be a list.")
        proposals: dict[str, CalendarChangeProposal] = {}
        order: list[str] = []
        for item in proposals_raw:
            if not isinstance(item, dict):
                raise CalendarStorageError("Each proposal must be an object.")
            self._assert_exact_keys(item, PERSISTED_PROPOSAL_KEYS, "proposal")
            normalized = item.get("normalized_payload")
            if not isinstance(normalized, dict):
                raise CalendarStorageError("normalized_payload must be an object.")
            proposal = validate_change_proposal(
                CalendarChangeProposal(
                    proposal_id=str(item["proposal_id"]),
                    account_id=str(item["account_id"]),
                    calendar_id=str(item["calendar_id"]),
                    operation=str(item["operation"]),  # type: ignore[arg-type]
                    event_id=(
                        None if item["event_id"] is None else str(item["event_id"])
                    ),
                    client_event_id=(
                        None
                        if item["client_event_id"] is None
                        else str(item["client_event_id"])
                    ),
                    normalized_payload={
                        str(key): str(value) for key, value in normalized.items()
                    },
                    fingerprint=str(item["fingerprint"]),
                    remote_etag=(
                        None
                        if item["remote_etag"] is None
                        else str(item["remote_etag"])
                    ),
                    status=str(item["status"]),  # type: ignore[arg-type]
                    created_at=str(item["created_at"]),
                    expires_at=str(item["expires_at"]),
                )
            )
            proposals[proposal.proposal_id] = proposal
            order.append(proposal.proposal_id)
        audits_raw = payload.get("audit_entries")
        if not isinstance(audits_raw, list):
            raise CalendarStorageError("audit_entries must be a list.")
        audits: list[CalendarAuditEntry] = []
        for item in audits_raw:
            if not isinstance(item, dict):
                raise CalendarStorageError("Each audit entry must be an object.")
            self._assert_exact_keys(item, PERSISTED_AUDIT_KEYS, "audit")
            audits.append(
                validate_calendar_audit_entry(
                    CalendarAuditEntry(
                        audit_id=str(item["audit_id"]),
                        action=str(item["action"]),  # type: ignore[arg-type]
                        occurred_at=str(item["occurred_at"]),
                        account_id=(
                            None
                            if item["account_id"] is None
                            else str(item["account_id"])
                        ),
                        proposal_id=(
                            None
                            if item["proposal_id"] is None
                            else str(item["proposal_id"])
                        ),
                        calendar_id=(
                            None
                            if item["calendar_id"] is None
                            else str(item["calendar_id"])
                        ),
                        event_id=(
                            None if item["event_id"] is None else str(item["event_id"])
                        ),
                        operation=(
                            None
                            if item["operation"] is None
                            else str(item["operation"])
                        ),
                        from_status=(
                            None
                            if item["from_status"] is None
                            else str(item["from_status"])
                        ),
                        to_status=(
                            None if item["to_status"] is None else str(item["to_status"])
                        ),
                        error_category=(
                            None
                            if item["error_category"] is None
                            else str(item["error_category"])
                        ),
                    )
                )
            )
        return _CalendarRepositoryState(
            account=account,
            proposals=proposals,
            proposal_order=order,
            audit_entries=audits,
        )

    def _persist(self, state: _CalendarRepositoryState) -> None:
        payload = {
            "version": CALENDAR_REPOSITORY_SCHEMA_VERSION,
            "account": (
                self._serialize_account(state.account)
                if state.account is not None
                else None
            ),
            "proposals": [
                self._serialize_proposal(state.proposals[proposal_id])
                for proposal_id in state.proposal_order
                if proposal_id in state.proposals
            ],
            "audit_entries": [
                self._serialize_audit(entry) for entry in state.audit_entries
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._state = state

    def _serialize_account(self, account: CalendarAccount) -> dict[str, object]:
        return {
            "account_id": account.account_id,
            "provider_id": account.provider_id,
            "display_label": account.display_label,
            "status": account.status,
            "default_calendar_id": account.default_calendar_id,
            "connected_at": account.connected_at,
            "scopes_granted": list(account.scopes_granted),
        }

    def _serialize_proposal(
        self,
        proposal: CalendarChangeProposal,
    ) -> dict[str, object]:
        return {
            "proposal_id": proposal.proposal_id,
            "account_id": proposal.account_id,
            "calendar_id": proposal.calendar_id,
            "operation": proposal.operation,
            "event_id": proposal.event_id,
            "client_event_id": proposal.client_event_id,
            "normalized_payload": dict(proposal.normalized_payload),
            "fingerprint": proposal.fingerprint,
            "remote_etag": proposal.remote_etag,
            "status": proposal.status,
            "created_at": proposal.created_at,
            "expires_at": proposal.expires_at,
        }

    def _serialize_audit(self, entry: CalendarAuditEntry) -> dict[str, object]:
        return {
            "audit_id": entry.audit_id,
            "action": entry.action,
            "occurred_at": entry.occurred_at,
            "account_id": entry.account_id,
            "proposal_id": entry.proposal_id,
            "calendar_id": entry.calendar_id,
            "event_id": entry.event_id,
            "operation": entry.operation,
            "from_status": entry.from_status,
            "to_status": entry.to_status,
            "error_category": entry.error_category,
        }

    def _atomic_write(self, content: str) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".calendar-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._file_path)
            self.atomic_replace_count += 1
            temporary_path = None
        except OSError as error:
            logger.error(
                "Calendar repository save failed error_type=%s",
                type(error).__name__,
            )
            raise CalendarStorageError(
                f"Calendar repository save failed: {type(error).__name__}",
                user_message=CALENDAR_SAVE_ERROR_MESSAGE,
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _clone_state(
        self,
        state: _CalendarRepositoryState,
    ) -> _CalendarRepositoryState:
        return _CalendarRepositoryState(
            account=state.account,
            proposals=dict(state.proposals),
            proposal_order=list(state.proposal_order),
            audit_entries=list(state.audit_entries),
        )

    def _enforce_audit_retention(self, state: _CalendarRepositoryState) -> None:
        overflow = len(state.audit_entries) - self._max_audit_entries
        if overflow > 0:
            del state.audit_entries[:overflow]

    def _enforce_proposal_retention(
        self,
        state: _CalendarRepositoryState,
        *,
        reserving: int,
    ) -> None:
        while len(state.proposals) + reserving > self._max_proposals:
            terminal_id = next(
                (
                    proposal_id
                    for proposal_id in state.proposal_order
                    if state.proposals[proposal_id].status in {"executed", "failed"}
                ),
                None,
            )
            if terminal_id is None:
                raise CalendarStorageError(
                    "Proposal capacity reached.",
                    user_message=CALENDAR_CAPACITY_ERROR_MESSAGE,
                )
            del state.proposals[terminal_id]
            state.proposal_order.remove(terminal_id)

    def _assert_exact_keys(
        self,
        payload: dict[str, object],
        allowed: frozenset[str],
        label: str,
    ) -> None:
        self._assert_no_forbidden_keys(payload)
        if set(payload) != set(allowed):
            raise CalendarStorageError(f"Unexpected {label} keys.")

    def _assert_no_forbidden_keys(self, payload: dict[str, object]) -> None:
        lowered = {str(key).strip().lower() for key in payload}
        overlap = lowered & FORBIDDEN_PERSISTED_KEYS
        if overlap:
            raise CalendarStorageError(
                f"Forbidden credential keys present: {sorted(overlap)}"
            )
