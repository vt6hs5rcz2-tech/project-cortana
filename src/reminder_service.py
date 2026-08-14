"""Business logic for Milestone 19 persistent reminders.

This service owns lifecycle transitions, overdue derivation, recurrence
advancement, and audit generation. It does not deliver notifications and does
not interact with memory, AI, security, tools, or workflows.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import MAX_REMINDER_LIST_RESULTS
from src.user_facing import cortana_domain_message
from src.reminder import (
    REMINDER_TERMINAL_STATUSES,
    ManualReminderClock,
    Reminder,
    ReminderAuditEntry,
    ReminderClock,
    ReminderSchedulingError,
    ReminderValidationError,
    SystemReminderClock,
    compute_next_occurrence_after,
    count_skipped_occurrences,
    create_reminder,
    create_reminder_audit_entry,
    is_overdue,
    local_wall_to_utc_iso,
    parse_local_wall_datetime,
    replace_reminder,
    validate_iana_timezone,
)
from src.reminder_repository import ReminderRepository, ReminderStorageError
from src.tool_common import validate_uuid


class ReminderServiceError(RuntimeError):
    """Raised when a reminder lifecycle operation cannot complete."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


@dataclass(frozen=True)
class ReminderView:
    """Reminder plus derived overdue indicator for display."""

    reminder: Reminder
    overdue: bool


class ReminderService:
    """Scheduling service over a ReminderRepository with an injectable clock."""

    def __init__(
        self,
        repository: ReminderRepository,
        *,
        clock: ReminderClock | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or SystemReminderClock()

    @property
    def clock(self) -> ReminderClock:
        """Return the injected clock."""
        return self._clock

    def create(
        self,
        *,
        title: str,
        local_due: str,
        timezone_name: str,
        recurrence_type: str = "none",
        recurrence_interval: int = 1,
        recurrence_weekdays: tuple[int, ...] = (),
        message: str | None = None,
    ) -> Reminder:
        """Create and persist one scheduled reminder with a created audit."""
        reminder = create_reminder(
            title=title,
            local_due=local_due,
            timezone_name=timezone_name,
            message=message,
            recurrence_type=recurrence_type,
            recurrence_interval=recurrence_interval,
            recurrence_weekdays=recurrence_weekdays,
            clock=self._clock,
        )
        audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="created",
            from_status=None,
            to_status=reminder.status,
            old_due_at=None,
            new_due_at=reminder.due_at,
            clock=self._clock,
        )
        try:
            return self._repository.save_reminder_with_audits(
                reminder,
                (audit,),
                is_new=True,
            )
        except ReminderStorageError:
            raise
        except ReminderValidationError as error:
            raise ReminderServiceError(
                str(error),
                user_message=cortana_domain_message(
                    error,
                    fallback="Cortana: I couldn't save that reminder.",
                ),
            ) from error

    def list_scheduled(self, *, limit: int = MAX_REMINDER_LIST_RESULTS) -> list[ReminderView]:
        """List scheduled reminders ordered by due_at then reminder_id."""
        if limit < 1:
            raise ReminderServiceError("List limit must be at least 1.")
        now = self._clock.utc_now_iso()
        scheduled = [
            reminder
            for reminder in self._repository.list_reminders()
            if reminder.status == "scheduled"
        ]
        scheduled.sort(key=lambda item: (item.due_at, item.reminder_id))
        views = [
            ReminderView(reminder=item, overdue=is_overdue(item, now_utc_iso=now))
            for item in scheduled[:limit]
        ]
        return views

    def count_all(self) -> int:
        """Return the total number of stored reminders."""
        return len(self._repository.list_reminders())

    def count_scheduled(self) -> int:
        """Return the number of reminders with status scheduled."""
        return sum(
            1
            for reminder in self._repository.list_reminders()
            if reminder.status == "scheduled"
        )

    def get(self, reminder_id: str) -> ReminderView | None:
        """Return one reminder view, or None when missing."""
        try:
            cleaned = validate_uuid(reminder_id, field_name="Reminder ID")
        except Exception as error:
            raise ReminderServiceError(
                "Invalid reminder ID.",
                user_message="Cortana: Reminder ID must be a valid UUID.",
            ) from error
        reminder = self._repository.get_reminder(cleaned)
        if reminder is None:
            return None
        return ReminderView(
            reminder=reminder,
            overdue=is_overdue(reminder, now_utc_iso=self._clock.utc_now_iso()),
        )

    def list_audits(
        self,
        reminder_id: str,
        *,
        limit: int,
    ) -> list[ReminderAuditEntry]:
        """Return the most recent audit entries for one reminder."""
        entries = self._repository.list_audit_entries(reminder_id=reminder_id)
        if limit < 1:
            return []
        return entries[-limit:]

    def complete(self, reminder_id: str) -> Reminder:
        """Complete a one-time reminder or advance a recurring series."""
        reminder = self._require_mutable(reminder_id)
        now = self._clock.utc_now_iso()

        if reminder.recurrence_type == "none":
            updated = replace_reminder(
                reminder,
                status="completed",
                clock=self._clock,
            )
            audit = create_reminder_audit_entry(
                reminder_id=reminder.reminder_id,
                action="completed",
                from_status=reminder.status,
                to_status="completed",
                old_due_at=reminder.due_at,
                new_due_at=reminder.due_at,
                clock=self._clock,
            )
            return self._repository.save_reminder_with_audits(updated, (audit,))

        try:
            next_due = compute_next_occurrence_after(
                reminder,
                after_utc_iso=now,
            )
        except ReminderSchedulingError as error:
            raise ReminderServiceError(
                str(error),
                user_message=error.user_message,
            ) from error

        skipped = count_skipped_occurrences(
            reminder,
            after_due_utc_iso=reminder.due_at,
            until_utc_iso=now,
        )
        updated = replace_reminder(
            reminder,
            status="scheduled",
            due_at=next_due,
            clock=self._clock,
        )
        completed_audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="completed",
            from_status=reminder.status,
            to_status="scheduled",
            old_due_at=reminder.due_at,
            new_due_at=next_due,
            skipped_occurrences=skipped if skipped > 0 else None,
            clock=self._clock,
        )
        advanced_audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="occurrence_advanced",
            from_status="scheduled",
            to_status="scheduled",
            old_due_at=reminder.due_at,
            new_due_at=next_due,
            skipped_occurrences=skipped if skipped > 0 else None,
            clock=self._clock,
        )
        return self._repository.save_reminder_with_audits(
            updated,
            (completed_audit, advanced_audit),
        )

    def cancel(self, reminder_id: str) -> Reminder:
        """Cancel a scheduled reminder."""
        reminder = self._require_mutable(reminder_id)
        updated = replace_reminder(
            reminder,
            status="cancelled",
            clock=self._clock,
        )
        audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="cancelled",
            from_status=reminder.status,
            to_status="cancelled",
            old_due_at=reminder.due_at,
            new_due_at=reminder.due_at,
            clock=self._clock,
        )
        return self._repository.save_reminder_with_audits(updated, (audit,))

    def snooze(self, reminder_id: str, *, local_until: str) -> Reminder:
        """Move only the current occurrence; preserve recurrence series anchor."""
        reminder = self._require_mutable(reminder_id)
        wall = parse_local_wall_datetime(local_until, field_name="Snooze until")
        new_due = local_wall_to_utc_iso(wall, reminder.timezone)
        updated = replace_reminder(
            reminder,
            due_at=new_due,
            # Keep recurrence_anchor_at unchanged via default ...
            clock=self._clock,
        )
        audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="snoozed",
            from_status=reminder.status,
            to_status=updated.status,
            old_due_at=reminder.due_at,
            new_due_at=new_due,
            clock=self._clock,
        )
        return self._repository.save_reminder_with_audits(updated, (audit,))

    def reschedule(
        self,
        reminder_id: str,
        *,
        local_due: str,
        timezone_name: str | None,
    ) -> Reminder:
        """Reschedule due time; reset recurrence anchor for recurring reminders."""
        reminder = self._require_mutable(reminder_id)
        zone = (
            reminder.timezone
            if timezone_name is None
            else validate_iana_timezone(timezone_name)
        )
        wall = parse_local_wall_datetime(local_due, field_name="Reschedule time")
        new_due = local_wall_to_utc_iso(wall, zone)
        if reminder.recurrence_type == "none":
            updated = replace_reminder(
                reminder,
                due_at=new_due,
                timezone_name=zone,
                recurrence_anchor_at=None,
                clock=self._clock,
            )
        else:
            updated = replace_reminder(
                reminder,
                due_at=new_due,
                timezone_name=zone,
                recurrence_anchor_at=new_due,
                clock=self._clock,
            )
        audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="rescheduled",
            from_status=reminder.status,
            to_status=updated.status,
            old_due_at=reminder.due_at,
            new_due_at=new_due,
            clock=self._clock,
        )
        return self._repository.save_reminder_with_audits(updated, (audit,))

    def _require_mutable(self, reminder_id: str) -> Reminder:
        try:
            cleaned = validate_uuid(reminder_id, field_name="Reminder ID")
        except Exception as error:
            raise ReminderServiceError(
                "Invalid reminder ID.",
                user_message="Cortana: Reminder ID must be a valid UUID.",
            ) from error
        reminder = self._repository.get_reminder(cleaned)
        if reminder is None:
            raise ReminderServiceError(
                "Reminder not found.",
                user_message=(
                    f"Cortana: No reminder found with ID '{reminder_id}'."
                ),
            )
        if reminder.status in REMINDER_TERMINAL_STATUSES:
            raise ReminderServiceError(
                f"Reminder is terminal ({reminder.status}).",
                user_message=(
                    f"Cortana: Reminder '{reminder_id}' is {reminder.status} "
                    "and cannot be modified."
                ),
            )
        return reminder


def create_reminder_service(
    repository: ReminderRepository,
    *,
    clock: ReminderClock | None = None,
) -> ReminderService:
    """Factory helper for reminder service construction."""
    return ReminderService(repository, clock=clock)


__all__ = [
    "ManualReminderClock",
    "ReminderService",
    "ReminderServiceError",
    "ReminderView",
    "SystemReminderClock",
    "create_reminder_service",
]
