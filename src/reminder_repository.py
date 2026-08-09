"""Reminder repository abstractions and atomic JSON persistence.

One application instance should access a reminder repository file at a time.
Atomic writes protect against partial-write corruption within a single writer.
They do not provide full concurrent multi-process coordination.
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

from src.config import (
    MAX_REMINDER_AUDIT_ENTRIES,
    MAX_STORED_REMINDERS,
    REMINDER_REPOSITORY_SCHEMA_VERSION,
)
from src.reminder import (
    Reminder,
    ReminderAuditEntry,
    ReminderValidationError,
    validate_reminder,
    validate_reminder_audit_entry,
)
from src.tool_common import validate_uuid

logger = logging.getLogger("ProjectCortana")

REMINDER_LOAD_ERROR_MESSAGE = (
    "Cortana: Reminder repository data could not be loaded safely. "
    "The existing repository file was left unchanged for inspection."
)
REMINDER_SAVE_ERROR_MESSAGE = (
    "Cortana: Reminder repository data could not be updated safely."
)
REMINDER_CAPACITY_ERROR_MESSAGE = (
    "Cortana: Reminder storage limit reached. "
    "Delete or cancel existing reminders before creating new ones."
)

PERSISTED_ROOT_KEYS: frozenset[str] = frozenset(
    {"version", "reminders", "audit_entries"}
)
PERSISTED_REMINDER_KEYS: frozenset[str] = frozenset(
    {
        "reminder_id",
        "title",
        "message",
        "status",
        "due_at",
        "timezone",
        "recurrence_type",
        "recurrence_interval",
        "recurrence_weekdays",
        "recurrence_anchor_at",
        "created_at",
        "updated_at",
    }
)
PERSISTED_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "audit_id",
        "reminder_id",
        "action",
        "occurred_at",
        "from_status",
        "to_status",
        "old_due_at",
        "new_due_at",
        "skipped_occurrences",
    }
)


class ReminderStorageError(RuntimeError):
    """Raised when reminder repository data cannot be loaded or saved safely."""

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or REMINDER_LOAD_ERROR_MESSAGE


class ReminderRepository(Protocol):
    """Persistence boundary for reminders and reminder audit entries."""

    def get_reminder(self, reminder_id: str) -> Reminder | None:
        """Return one reminder by ID, or None when missing/invalid."""

    def list_reminders(self) -> list[Reminder]:
        """Return reminders in deterministic storage order."""

    def list_audit_entries(
        self,
        *,
        reminder_id: str | None = None,
    ) -> list[ReminderAuditEntry]:
        """Return audit entries in append order, optionally filtered."""

    def save_reminder_with_audits(
        self,
        reminder: Reminder,
        audit_entries: Sequence[ReminderAuditEntry],
        *,
        is_new: bool = False,
    ) -> Reminder:
        """Atomically persist one reminder update plus one or more audits."""


@dataclass
class _ReminderRepositoryState:
    """In-memory snapshot of the reminder repository."""

    reminders: dict[str, Reminder] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    audit_entries: list[ReminderAuditEntry] = field(default_factory=list)


class JsonReminderRepository:
    """JSON-backed reminder store with atomic writes and sticky load errors."""

    def __init__(
        self,
        file_path: Path,
        *,
        max_reminders: int = MAX_STORED_REMINDERS,
        max_audit_entries: int = MAX_REMINDER_AUDIT_ENTRIES,
    ) -> None:
        if max_reminders < 1:
            raise ReminderValidationError("max_reminders must be at least 1.")
        if max_audit_entries < 1:
            raise ReminderValidationError("max_audit_entries must be at least 1.")
        self._file_path = file_path
        self._max_reminders = max_reminders
        self._max_audit_entries = max_audit_entries
        self._state: _ReminderRepositoryState | None = None
        self._load_error: ReminderStorageError | None = None
        self.atomic_replace_count = 0

    @property
    def file_path(self) -> Path:
        """Return the reminder repository JSON file path."""
        return self._file_path

    def get_reminder(self, reminder_id: str) -> Reminder | None:
        """Return one reminder by ID, or None when missing/invalid."""
        try:
            cleaned = validate_uuid(reminder_id, field_name="Reminder ID")
        except Exception:
            return None
        return self._ensure_loaded().reminders.get(cleaned)

    def list_reminders(self) -> list[Reminder]:
        """Return reminders in deterministic storage order."""
        state = self._ensure_loaded()
        return [
            state.reminders[reminder_id]
            for reminder_id in state.order
            if reminder_id in state.reminders
        ]

    def list_audit_entries(
        self,
        *,
        reminder_id: str | None = None,
    ) -> list[ReminderAuditEntry]:
        """Return audit entries in append order, optionally filtered."""
        entries = list(self._ensure_loaded().audit_entries)
        if reminder_id is None:
            return entries
        try:
            cleaned = validate_uuid(reminder_id, field_name="Reminder ID")
        except Exception:
            return []
        return [entry for entry in entries if entry.reminder_id == cleaned]

    def save_reminder_with_audits(
        self,
        reminder: Reminder,
        audit_entries: Sequence[ReminderAuditEntry],
        *,
        is_new: bool = False,
    ) -> Reminder:
        """Atomically persist one reminder mutation with its audit entries."""
        if not audit_entries:
            raise ReminderStorageError(
                "Lifecycle mutations require at least one audit entry.",
                user_message=REMINDER_SAVE_ERROR_MESSAGE,
            )

        state = self._clone_state(self._ensure_loaded())
        validated = validate_reminder(reminder)
        existing = state.reminders.get(validated.reminder_id)

        if is_new:
            if existing is not None:
                raise ReminderStorageError(
                    "Duplicate reminder ID rejected.",
                    user_message=REMINDER_SAVE_ERROR_MESSAGE,
                )
            if len(state.order) >= self._max_reminders:
                raise ReminderStorageError(
                    "Reminder storage capacity reached.",
                    user_message=REMINDER_CAPACITY_ERROR_MESSAGE,
                )
            state.reminders[validated.reminder_id] = validated
            state.order.append(validated.reminder_id)
        else:
            if existing is None:
                raise ReminderStorageError(
                    "Reminder not found for update.",
                    user_message=REMINDER_SAVE_ERROR_MESSAGE,
                )
            state.reminders[validated.reminder_id] = validated

        for entry in audit_entries:
            validated_entry = validate_reminder_audit_entry(entry)
            if validated_entry.reminder_id != validated.reminder_id:
                raise ReminderStorageError(
                    "Audit reminder_id must match the mutated reminder.",
                    user_message=REMINDER_SAVE_ERROR_MESSAGE,
                )
            if any(
                item.audit_id == validated_entry.audit_id
                for item in state.audit_entries
            ):
                raise ReminderStorageError(
                    "Duplicate reminder audit ID rejected.",
                    user_message=REMINDER_SAVE_ERROR_MESSAGE,
                )
            state.audit_entries.append(validated_entry)

        self._enforce_audit_retention(state)
        self._persist(state)
        return validated

    def _ensure_loaded(self) -> _ReminderRepositoryState:
        if self._load_error is not None:
            raise self._load_error
        if self._state is not None:
            return self._state

        if not self._file_path.exists():
            self._state = _ReminderRepositoryState()
            return self._state

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                raise ReminderStorageError(
                    "Reminder repository file is empty.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise ReminderStorageError(
                    "Reminder repository root must be an object.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )
            self._state = self._deserialize_state(payload)
            return self._state
        except ReminderStorageError as error:
            self._load_error = ReminderStorageError(
                str(error),
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
            raise self._load_error from error
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            ReminderValidationError,
        ) as error:
            logger.error(
                "Reminder repository load failed with error_type=%s",
                type(error).__name__,
            )
            self._load_error = ReminderStorageError(
                f"Reminder repository load failed: {type(error).__name__}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
            raise self._load_error from None

    def _deserialize_state(self, payload: dict[str, object]) -> _ReminderRepositoryState:
        unexpected_root = set(payload) - PERSISTED_ROOT_KEYS
        if unexpected_root:
            raise ReminderStorageError(
                f"Unexpected reminder repository keys: {sorted(unexpected_root)}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        missing_root = PERSISTED_ROOT_KEYS - set(payload)
        if missing_root:
            raise ReminderStorageError(
                f"Missing reminder repository keys: {sorted(missing_root)}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        version = payload.get("version")
        if version != REMINDER_REPOSITORY_SCHEMA_VERSION:
            raise ReminderStorageError(
                f"Unsupported reminder repository schema version: {version!r}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        reminders_payload = payload.get("reminders")
        audits_payload = payload.get("audit_entries")
        if not isinstance(reminders_payload, list) or not isinstance(
            audits_payload, list
        ):
            raise ReminderStorageError(
                "Reminder repository collections must be lists.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        state = _ReminderRepositoryState()
        for item in reminders_payload:
            reminder = self._deserialize_reminder(item)
            if reminder.reminder_id in state.reminders:
                raise ReminderStorageError(
                    "Duplicate reminder ID in repository file.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )
            state.reminders[reminder.reminder_id] = reminder
            state.order.append(reminder.reminder_id)

        if len(state.order) > self._max_reminders:
            raise ReminderStorageError(
                "Reminder repository exceeds configured capacity.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        seen_audit_ids: set[str] = set()
        for item in audits_payload:
            entry = self._deserialize_audit(item)
            if entry.audit_id in seen_audit_ids:
                raise ReminderStorageError(
                    "Duplicate reminder audit ID in repository file.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )
            seen_audit_ids.add(entry.audit_id)
            state.audit_entries.append(entry)

        self._enforce_audit_retention(state)
        return state

    def _deserialize_reminder(self, item: object) -> Reminder:
        if not isinstance(item, dict):
            raise ReminderStorageError(
                "Reminder record must be an object.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_REMINDER_KEYS
        if unexpected:
            raise ReminderStorageError(
                f"Unexpected reminder keys: {sorted(unexpected)}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_REMINDER_KEYS - set(item)
        if missing:
            raise ReminderStorageError(
                f"Missing reminder keys: {sorted(missing)}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        weekdays_raw = item["recurrence_weekdays"]
        if not isinstance(weekdays_raw, list):
            raise ReminderStorageError(
                "recurrence_weekdays must be a list.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        weekdays: list[int] = []
        for day in weekdays_raw:
            if not isinstance(day, int) or isinstance(day, bool):
                raise ReminderStorageError(
                    "recurrence_weekdays values must be integers.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )
            weekdays.append(day)

        interval = item["recurrence_interval"]
        if not isinstance(interval, int) or isinstance(interval, bool):
            raise ReminderStorageError(
                "recurrence_interval must be an integer.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        message = item["message"]
        if message is not None and not isinstance(message, str):
            raise ReminderStorageError(
                "message must be a string or null.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        anchor = item["recurrence_anchor_at"]
        if anchor is not None and not isinstance(anchor, str):
            raise ReminderStorageError(
                "recurrence_anchor_at must be a string or null.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        for key in (
            "reminder_id",
            "title",
            "status",
            "due_at",
            "timezone",
            "recurrence_type",
            "created_at",
            "updated_at",
        ):
            if not isinstance(item[key], str):
                raise ReminderStorageError(
                    f"{key} must be a string.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )

        return validate_reminder(
            Reminder(
                reminder_id=str(item["reminder_id"]),
                title=str(item["title"]),
                message=message,
                status=str(item["status"]),  # type: ignore[arg-type]
                due_at=str(item["due_at"]),
                timezone=str(item["timezone"]),
                recurrence_type=str(item["recurrence_type"]),  # type: ignore[arg-type]
                recurrence_interval=interval,
                recurrence_weekdays=tuple(weekdays),
                recurrence_anchor_at=anchor,
                created_at=str(item["created_at"]),
                updated_at=str(item["updated_at"]),
            )
        )

    def _deserialize_audit(self, item: object) -> ReminderAuditEntry:
        if not isinstance(item, dict):
            raise ReminderStorageError(
                "Audit record must be an object.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_AUDIT_KEYS
        if unexpected:
            raise ReminderStorageError(
                f"Unexpected audit keys: {sorted(unexpected)}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_AUDIT_KEYS - set(item)
        if missing:
            raise ReminderStorageError(
                f"Missing audit keys: {sorted(missing)}",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        skipped = item["skipped_occurrences"]
        if skipped is not None and (
            not isinstance(skipped, int) or isinstance(skipped, bool)
        ):
            raise ReminderStorageError(
                "skipped_occurrences must be an integer or null.",
                user_message=REMINDER_LOAD_ERROR_MESSAGE,
            )

        for key in ("audit_id", "reminder_id", "action", "occurred_at"):
            if not isinstance(item[key], str):
                raise ReminderStorageError(
                    f"{key} must be a string.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )
        for key in ("from_status", "to_status", "old_due_at", "new_due_at"):
            value = item[key]
            if value is not None and not isinstance(value, str):
                raise ReminderStorageError(
                    f"{key} must be a string or null.",
                    user_message=REMINDER_LOAD_ERROR_MESSAGE,
                )

        from_status = item["from_status"]
        to_status = item["to_status"]
        old_due_at = item["old_due_at"]
        new_due_at = item["new_due_at"]
        return validate_reminder_audit_entry(
            ReminderAuditEntry(
                audit_id=str(item["audit_id"]),
                reminder_id=str(item["reminder_id"]),
                action=str(item["action"]),  # type: ignore[arg-type]
                occurred_at=str(item["occurred_at"]),
                from_status=None if from_status is None else str(from_status),
                to_status=None if to_status is None else str(to_status),
                old_due_at=None if old_due_at is None else str(old_due_at),
                new_due_at=None if new_due_at is None else str(new_due_at),
                skipped_occurrences=skipped if isinstance(skipped, int) else None,
            )
        )

    def _persist(self, state: _ReminderRepositoryState) -> None:
        payload = {
            "version": REMINDER_REPOSITORY_SCHEMA_VERSION,
            "reminders": [
                self._serialize_reminder(state.reminders[reminder_id])
                for reminder_id in state.order
                if reminder_id in state.reminders
            ],
            "audit_entries": [
                self._serialize_audit(entry) for entry in state.audit_entries
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._state = state

    def _serialize_reminder(self, reminder: Reminder) -> dict[str, object]:
        return {
            "reminder_id": reminder.reminder_id,
            "title": reminder.title,
            "message": reminder.message,
            "status": reminder.status,
            "due_at": reminder.due_at,
            "timezone": reminder.timezone,
            "recurrence_type": reminder.recurrence_type,
            "recurrence_interval": reminder.recurrence_interval,
            "recurrence_weekdays": list(reminder.recurrence_weekdays),
            "recurrence_anchor_at": reminder.recurrence_anchor_at,
            "created_at": reminder.created_at,
            "updated_at": reminder.updated_at,
        }

    def _serialize_audit(self, entry: ReminderAuditEntry) -> dict[str, object]:
        return {
            "audit_id": entry.audit_id,
            "reminder_id": entry.reminder_id,
            "action": entry.action,
            "occurred_at": entry.occurred_at,
            "from_status": entry.from_status,
            "to_status": entry.to_status,
            "old_due_at": entry.old_due_at,
            "new_due_at": entry.new_due_at,
            "skipped_occurrences": entry.skipped_occurrences,
        }

    def _atomic_write(self, content: str) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".reminders-",
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
                "Reminder repository save failed with error_type=%s",
                type(error).__name__,
            )
            raise ReminderStorageError(
                f"Reminder repository save failed: {type(error).__name__}",
                user_message=REMINDER_SAVE_ERROR_MESSAGE,
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _clone_state(
        self,
        state: _ReminderRepositoryState,
    ) -> _ReminderRepositoryState:
        return _ReminderRepositoryState(
            reminders=dict(state.reminders),
            order=list(state.order),
            audit_entries=list(state.audit_entries),
        )

    def _enforce_audit_retention(self, state: _ReminderRepositoryState) -> None:
        overflow = len(state.audit_entries) - self._max_audit_entries
        if overflow > 0:
            del state.audit_entries[:overflow]
