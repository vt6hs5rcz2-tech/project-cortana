"""Local slash-command handlers for Milestone 19 reminder scheduling."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    MAX_REMINDER_LIST_PREVIEW_CHARS,
    MAX_REMINDER_LIST_RESULTS,
    MAX_REMINDER_SHOW_AUDIT_ENTRIES,
    get_default_reminder_repository_file_path,
)
from src.reminder import (
    Reminder,
    ReminderValidationError,
    format_local_wall,
    format_recurrence_spec,
    parse_recurrence_spec,
    utc_iso_to_local,
)
from src.reminder_repository import (
    JsonReminderRepository,
    ReminderRepository,
    ReminderStorageError,
)
from src.reminder_service import (
    ReminderService,
    ReminderServiceError,
    ReminderView,
)
from src.command_argument_utils import extract_command_argument

logger = logging.getLogger("ProjectCortana")

FIELD_DELIMITER = " | "
KEEP_TIMEZONE_SENTINEL = "-"
NO_MESSAGE_SENTINEL = "-"

COMMAND_REMINDER_ADD = "reminder-add"
COMMAND_REMINDERS = "reminders"
COMMAND_REMINDER_SHOW = "reminder-show"
COMMAND_REMINDER_COMPLETE = "reminder-complete"
COMMAND_REMINDER_CANCEL = "reminder-cancel"
COMMAND_REMINDER_SNOOZE = "reminder-snooze"
COMMAND_REMINDER_RESCHEDULE = "reminder-reschedule"

REMINDER_COMMAND_NAMES = frozenset(
    {
        COMMAND_REMINDER_ADD,
        COMMAND_REMINDERS,
        COMMAND_REMINDER_SHOW,
        COMMAND_REMINDER_COMPLETE,
        COMMAND_REMINDER_CANCEL,
        COMMAND_REMINDER_SNOOZE,
        COMMAND_REMINDER_RESCHEDULE,
    }
)

REMINDER_ADD_USAGE = (
    "Cortana: Usage: /reminder-add <title> | <YYYY-MM-DD HH:MM> | "
    "<IANA-timezone> | <recurrence> | <message>"
)
REMINDER_SHOW_USAGE = "Cortana: Usage: /reminder-show <reminder-id>"
REMINDER_COMPLETE_USAGE = "Cortana: Usage: /reminder-complete <reminder-id>"
REMINDER_CANCEL_USAGE = "Cortana: Usage: /reminder-cancel <reminder-id>"
REMINDER_SNOOZE_USAGE = (
    "Cortana: Usage: /reminder-snooze <reminder-id> | <YYYY-MM-DD HH:MM>"
)
REMINDER_RESCHEDULE_USAGE = (
    "Cortana: Usage: /reminder-reschedule <reminder-id> | "
    "<YYYY-MM-DD HH:MM> | <IANA-timezone-or->"
)
REMINDERS_EMPTY = "Cortana: No scheduled reminders."


@dataclass(frozen=True)
class ReminderCommandContext:
    """Inputs available to reminder slash-command handlers."""

    message: str
    reminder_service: ReminderService


@dataclass(frozen=True)
class ReminderCommandResult:
    """User-facing result from a reminder command."""

    message: str


ReminderCommandHandler = Callable[
    [ReminderCommandContext],
    ReminderCommandResult,
]


def create_default_reminder_service(
    *,
    repository_file_path: Path | None = None,
) -> ReminderService:
    """Create the default JSON-backed reminder service."""
    path = repository_file_path or get_default_reminder_repository_file_path()
    repository: ReminderRepository = JsonReminderRepository(path)
    return ReminderService(repository)


def split_delimited_fields(argument: str, expected_count: int) -> list[str] | None:
    """Split a pipe-delimited argument into an exact field count."""
    parts = argument.split(FIELD_DELIMITER)
    if len(parts) != expected_count:
        return None
    return [part.strip() for part in parts]


def bounded_preview(text: str, *, max_chars: int = MAX_REMINDER_LIST_PREVIEW_CHARS) -> str:
    """Return a bounded single-line preview."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3] + "..."


def handle_reminder_command(
    command_name: str,
    context: ReminderCommandContext,
) -> ReminderCommandResult | None:
    """Dispatch one reminder command, or return None when unknown."""
    handler = REMINDER_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    try:
        return handler(context)
    except ReminderServiceError as error:
        return ReminderCommandResult(message=error.user_message)
    except ReminderStorageError as error:
        return ReminderCommandResult(message=error.user_message)
    except ReminderValidationError as error:
        return ReminderCommandResult(message=f"Cortana: {error}")


def _handle_reminder_add(context: ReminderCommandContext) -> ReminderCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 5)
    if fields is None:
        return ReminderCommandResult(message=REMINDER_ADD_USAGE)

    title, local_due, timezone_name, recurrence_raw, message_raw = fields
    if not title or not local_due or not timezone_name or not recurrence_raw:
        return ReminderCommandResult(message=REMINDER_ADD_USAGE)

    recurrence_type, interval, weekdays = parse_recurrence_spec(recurrence_raw)
    message = None if message_raw == NO_MESSAGE_SENTINEL else message_raw
    reminder = context.reminder_service.create(
        title=title,
        local_due=local_due,
        timezone_name=timezone_name,
        recurrence_type=recurrence_type,
        recurrence_interval=interval,
        recurrence_weekdays=weekdays,
        message=message,
    )
    return ReminderCommandResult(
        message=(
            f"Cortana: Reminder created ({reminder.reminder_id}). "
            f"Due {_format_due(reminder)} [{reminder.timezone}] "
            f"recurrence={format_recurrence_spec(reminder)}"
        )
    )


def _handle_reminders(context: ReminderCommandContext) -> ReminderCommandResult:
    argument = extract_command_argument(context.message).strip()
    if argument:
        return ReminderCommandResult(
            message="Cortana: Usage: /reminders"
        )
    views = context.reminder_service.list_scheduled(limit=MAX_REMINDER_LIST_RESULTS)
    if not views:
        return ReminderCommandResult(message=REMINDERS_EMPTY)

    lines = ["Cortana: Scheduled reminders:"]
    for view in views:
        lines.append(_format_list_line(view))
    return ReminderCommandResult(message="\n".join(lines))


def _handle_reminder_show(context: ReminderCommandContext) -> ReminderCommandResult:
    reminder_id = extract_command_argument(context.message).strip()
    if not reminder_id or FIELD_DELIMITER in reminder_id:
        return ReminderCommandResult(message=REMINDER_SHOW_USAGE)

    view = context.reminder_service.get(reminder_id)
    if view is None:
        return ReminderCommandResult(
            message=f"Cortana: No reminder found with ID '{reminder_id}'."
        )

    reminder = view.reminder
    lines = [
        f"Cortana: Reminder {reminder.reminder_id}",
        f"  title: {reminder.title}",
        f"  status: {reminder.status}",
        f"  overdue: {'yes' if view.overdue else 'no'}",
        f"  due_at_utc: {reminder.due_at}",
        f"  due_local: {_format_due(reminder)}",
        f"  timezone: {reminder.timezone}",
        f"  recurrence: {format_recurrence_spec(reminder)}",
        f"  message: {reminder.message if reminder.message is not None else '-'}",
        f"  created_at: {reminder.created_at}",
        f"  updated_at: {reminder.updated_at}",
    ]
    if reminder.recurrence_anchor_at is not None:
        lines.append(f"  recurrence_anchor_at: {reminder.recurrence_anchor_at}")

    audits = context.reminder_service.list_audits(
        reminder.reminder_id,
        limit=MAX_REMINDER_SHOW_AUDIT_ENTRIES,
    )
    if audits:
        lines.append("  recent_audit:")
        for entry in audits:
            detail = (
                f"    - {entry.occurred_at} {entry.action}"
                f" {entry.from_status or '-'}->{entry.to_status or '-'}"
            )
            if entry.old_due_at or entry.new_due_at:
                detail += f" due:{entry.old_due_at or '-'}->{entry.new_due_at or '-'}"
            if entry.skipped_occurrences is not None:
                detail += f" skipped={entry.skipped_occurrences}"
            lines.append(detail)
    return ReminderCommandResult(message="\n".join(lines))


def _handle_reminder_complete(context: ReminderCommandContext) -> ReminderCommandResult:
    reminder_id = extract_command_argument(context.message).strip()
    if not reminder_id or FIELD_DELIMITER in reminder_id:
        return ReminderCommandResult(message=REMINDER_COMPLETE_USAGE)
    reminder = context.reminder_service.complete(reminder_id)
    if reminder.status == "completed":
        return ReminderCommandResult(
            message=f"Cortana: Reminder completed ({reminder.reminder_id})."
        )
    return ReminderCommandResult(
        message=(
            f"Cortana: Reminder occurrence completed ({reminder.reminder_id}). "
            f"Next due {_format_due(reminder)} [{reminder.timezone}]"
        )
    )


def _handle_reminder_cancel(context: ReminderCommandContext) -> ReminderCommandResult:
    reminder_id = extract_command_argument(context.message).strip()
    if not reminder_id or FIELD_DELIMITER in reminder_id:
        return ReminderCommandResult(message=REMINDER_CANCEL_USAGE)
    reminder = context.reminder_service.cancel(reminder_id)
    return ReminderCommandResult(
        message=f"Cortana: Reminder cancelled ({reminder.reminder_id})."
    )


def _handle_reminder_snooze(context: ReminderCommandContext) -> ReminderCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 2)
    if fields is None:
        return ReminderCommandResult(message=REMINDER_SNOOZE_USAGE)
    reminder_id, local_until = fields
    if not reminder_id or not local_until:
        return ReminderCommandResult(message=REMINDER_SNOOZE_USAGE)
    reminder = context.reminder_service.snooze(
        reminder_id,
        local_until=local_until,
    )
    return ReminderCommandResult(
        message=(
            f"Cortana: Reminder snoozed ({reminder.reminder_id}). "
            f"Due {_format_due(reminder)} [{reminder.timezone}]"
        )
    )


def _handle_reminder_reschedule(
    context: ReminderCommandContext,
) -> ReminderCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return ReminderCommandResult(message=REMINDER_RESCHEDULE_USAGE)
    reminder_id, local_due, timezone_raw = fields
    if not reminder_id or not local_due or not timezone_raw:
        return ReminderCommandResult(message=REMINDER_RESCHEDULE_USAGE)
    timezone_name = None if timezone_raw == KEEP_TIMEZONE_SENTINEL else timezone_raw
    reminder = context.reminder_service.reschedule(
        reminder_id,
        local_due=local_due,
        timezone_name=timezone_name,
    )
    return ReminderCommandResult(
        message=(
            f"Cortana: Reminder rescheduled ({reminder.reminder_id}). "
            f"Due {_format_due(reminder)} [{reminder.timezone}]"
        )
    )


def _format_due(reminder: Reminder) -> str:
    local = utc_iso_to_local(reminder.due_at, reminder.timezone)
    return format_local_wall(local)


def _format_list_line(view: ReminderView) -> str:
    reminder = view.reminder
    overdue_marker = " overdue" if view.overdue else ""
    title = bounded_preview(reminder.title)
    return (
        f"  [{reminder.reminder_id}] {_format_due(reminder)} "
        f"[{reminder.timezone}]{overdue_marker} "
        f"{title} ({format_recurrence_spec(reminder)})"
    )


REMINDER_COMMAND_HANDLERS: dict[str, ReminderCommandHandler] = {
    COMMAND_REMINDER_ADD: _handle_reminder_add,
    COMMAND_REMINDERS: _handle_reminders,
    COMMAND_REMINDER_SHOW: _handle_reminder_show,
    COMMAND_REMINDER_COMPLETE: _handle_reminder_complete,
    COMMAND_REMINDER_CANCEL: _handle_reminder_cancel,
    COMMAND_REMINDER_SNOOZE: _handle_reminder_snooze,
    COMMAND_REMINDER_RESCHEDULE: _handle_reminder_reschedule,
}
