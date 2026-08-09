"""Local slash-command handlers for Milestone 20 calendar integration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.calendar_models import (
    CalendarChangeProposal,
    CalendarError,
    CalendarEvent,
    CalendarValidationError,
    format_local_wall_from_utc,
)
from src.calendar_repository import (
    CalendarStorageError,
    JsonCalendarRepository,
)
from src.calendar_service import CalendarService
from src.config import (
    MAX_CALENDAR_LIST_PREVIEW_CHARS,
    MAX_CALENDAR_LIST_RESULTS,
    get_default_calendar_repository_file_path,
)
from src.reminder_commands import split_delimited_fields
from src.secret_store import KeyringSecretStore, SecretStore, SecretStoreError
from src.command_argument_utils import extract_command_argument

logger = logging.getLogger("ProjectCortana")

COMMAND_CALENDAR_CONNECT = "calendar-connect"
COMMAND_CALENDAR_DISCONNECT = "calendar-disconnect"
COMMAND_CALENDARS = "calendars"
COMMAND_CALENDAR_USE = "calendar-use"
COMMAND_CALENDAR_EVENTS = "calendar-events"
COMMAND_CALENDAR_EVENT = "calendar-event"
COMMAND_CALENDAR_FREEBUSY = "calendar-freebusy"
COMMAND_CALENDAR_CREATE = "calendar-create"
COMMAND_CALENDAR_RESCHEDULE = "calendar-reschedule"
COMMAND_CALENDAR_CANCEL = "calendar-cancel"
COMMAND_CALENDAR_CONFIRM = "calendar-confirm"

CALENDAR_COMMAND_NAMES = frozenset(
    {
        COMMAND_CALENDAR_CONNECT,
        COMMAND_CALENDAR_DISCONNECT,
        COMMAND_CALENDARS,
        COMMAND_CALENDAR_USE,
        COMMAND_CALENDAR_EVENTS,
        COMMAND_CALENDAR_EVENT,
        COMMAND_CALENDAR_FREEBUSY,
        COMMAND_CALENDAR_CREATE,
        COMMAND_CALENDAR_RESCHEDULE,
        COMMAND_CALENDAR_CANCEL,
        COMMAND_CALENDAR_CONFIRM,
    }
)

CALENDAR_USE_USAGE = "Cortana: Usage: /calendar-use <calendar-id>"
CALENDAR_EVENT_USAGE = (
    "Cortana: Usage: /calendar-event <calendar-id-or-> | <event-id>"
)
CALENDAR_FREEBUSY_USAGE = (
    "Cortana: Usage: /calendar-freebusy <calendar-id-or-> | "
    "<YYYY-MM-DD HH:MM> | <YYYY-MM-DD HH:MM>"
)
CALENDAR_CREATE_USAGE = (
    "Cortana: Usage: /calendar-create <calendar-id-or-> | <title> | "
    "<YYYY-MM-DD HH:MM> | <YYYY-MM-DD HH:MM>"
)
CALENDAR_RESCHEDULE_USAGE = (
    "Cortana: Usage: /calendar-reschedule <calendar-id-or-> | <event-id> | "
    "<YYYY-MM-DD HH:MM> | <YYYY-MM-DD HH:MM>"
)
CALENDAR_CANCEL_USAGE = (
    "Cortana: Usage: /calendar-cancel <calendar-id-or-> | <event-id>"
)
CALENDAR_CONFIRM_USAGE = "Cortana: Usage: /calendar-confirm <proposal-id>"
CALENDARS_EMPTY = "Cortana: No calendars were returned for the connected account."
EVENTS_EMPTY = "Cortana: No upcoming events were found in the selected window."


@dataclass(frozen=True)
class CalendarCommandContext:
    """Inputs available to calendar slash-command handlers."""

    message: str
    calendar_service: CalendarService


@dataclass(frozen=True)
class CalendarCommandResult:
    """User-facing result from a calendar command."""

    message: str


CalendarCommandHandler = Callable[
    [CalendarCommandContext],
    CalendarCommandResult,
]


def create_default_calendar_service(
    *,
    repository_file_path: Path | None = None,
    secret_store: SecretStore | None = None,
    oauth_client_file: Path | None = None,
) -> CalendarService:
    """Create the default JSON-backed calendar service with OS secret storage."""
    path = repository_file_path or get_default_calendar_repository_file_path()
    repository = JsonCalendarRepository(path)
    store = secret_store or KeyringSecretStore()
    return CalendarService(
        repository,
        store,
        oauth_client_file=oauth_client_file,
    )


def handle_calendar_command(
    command_name: str,
    context: CalendarCommandContext,
) -> CalendarCommandResult | None:
    """Dispatch one calendar command, or return None when unknown."""
    handler = CALENDAR_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    try:
        return handler(context)
    except CalendarError as error:
        return CalendarCommandResult(message=error.user_message)
    except CalendarStorageError as error:
        return CalendarCommandResult(message=error.user_message)
    except SecretStoreError as error:
        return CalendarCommandResult(message=error.user_message)
    except CalendarValidationError as error:
        return CalendarCommandResult(message=f"Cortana: {error}")


def _handle_calendar_connect(context: CalendarCommandContext) -> CalendarCommandResult:
    argument = extract_command_argument(context.message).strip()
    if argument:
        return CalendarCommandResult(
            message="Cortana: Usage: /calendar-connect"
        )
    account = context.calendar_service.connect()
    return CalendarCommandResult(
        message=(
            f"Cortana: Google Calendar connected ({account.account_id}). "
            f"Default calendar: {account.default_calendar_id}."
        )
    )


def _handle_calendar_disconnect(
    context: CalendarCommandContext,
) -> CalendarCommandResult:
    argument = extract_command_argument(context.message).strip()
    if argument:
        return CalendarCommandResult(
            message="Cortana: Usage: /calendar-disconnect"
        )
    context.calendar_service.disconnect()
    return CalendarCommandResult(
        message="Cortana: Google Calendar disconnected. Local credentials removed."
    )


def _handle_calendars(context: CalendarCommandContext) -> CalendarCommandResult:
    argument = extract_command_argument(context.message).strip()
    if argument:
        return CalendarCommandResult(message="Cortana: Usage: /calendars")
    calendars = context.calendar_service.list_calendars()
    if not calendars:
        return CalendarCommandResult(message=CALENDARS_EMPTY)
    lines = ["Cortana: Calendars:"]
    for item in calendars:
        primary = " primary" if item.primary else ""
        lines.append(
            f"  [{item.calendar_id}] {_preview(item.summary)} "
            f"tz={item.timezone} role={item.access_role}{primary}"
        )
    return CalendarCommandResult(message="\n".join(lines))


def _handle_calendar_use(context: CalendarCommandContext) -> CalendarCommandResult:
    calendar_id = extract_command_argument(context.message).strip()
    if not calendar_id or " | " in calendar_id:
        return CalendarCommandResult(message=CALENDAR_USE_USAGE)
    account = context.calendar_service.set_default_calendar(calendar_id)
    return CalendarCommandResult(
        message=(
            f"Cortana: Default calendar set to {account.default_calendar_id}."
        )
    )


def _handle_calendar_events(context: CalendarCommandContext) -> CalendarCommandResult:
    argument = extract_command_argument(context.message).strip()
    calendar_id: str | None = None
    if argument:
        if " | " in argument:
            return CalendarCommandResult(
                message="Cortana: Usage: /calendar-events [<calendar-id>]"
            )
        calendar_id = argument
    events = context.calendar_service.list_events(
        calendar_id,
        limit=MAX_CALENDAR_LIST_RESULTS,
    )
    if not events:
        return CalendarCommandResult(message=EVENTS_EMPTY)
    lines = ["Cortana: Upcoming events:"]
    for event in events:
        lines.append(_format_event_line(event))
    return CalendarCommandResult(message="\n".join(lines))


def _handle_calendar_event(context: CalendarCommandContext) -> CalendarCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 2)
    if fields is None:
        return CalendarCommandResult(message=CALENDAR_EVENT_USAGE)
    calendar_token, event_id = fields
    if not event_id:
        return CalendarCommandResult(message=CALENDAR_EVENT_USAGE)
    calendar_id = None if calendar_token == "-" else calendar_token
    event = context.calendar_service.get_event(event_id, calendar_id)
    lines = [
        f"Cortana: Event {event.event_id}",
        f"  calendar_id: {event.calendar_id}",
        f"  title: {event.title}",
        f"  status: {event.status}",
        f"  all_day: {'yes' if event.is_all_day else 'no'}",
        f"  start: {event.start}",
        f"  end: {event.end}",
        f"  recurring: {'yes' if event.is_recurring else 'no'}",
        f"  recurring_event_id: {event.recurring_event_id or '-'}",
        f"  etag: {event.etag or '-'}",
    ]
    if event.description:
        lines.append(f"  description: {_preview(event.description)}")
    return CalendarCommandResult(message="\n".join(lines))


def _handle_calendar_freebusy(context: CalendarCommandContext) -> CalendarCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return CalendarCommandResult(message=CALENDAR_FREEBUSY_USAGE)
    calendar_token, start_local, end_local = fields
    if not start_local or not end_local:
        return CalendarCommandResult(message=CALENDAR_FREEBUSY_USAGE)
    calendar_id = None if calendar_token == "-" else calendar_token
    windows = context.calendar_service.get_freebusy(
        start_local=start_local,
        end_local=end_local,
        calendar_id=calendar_id,
    )
    if not windows:
        return CalendarCommandResult(
            message="Cortana: No busy intervals in the requested window."
        )
    lines = ["Cortana: Busy intervals:"]
    for window in windows:
        lines.append(f"  [{window.calendar_id}] {window.start} -> {window.end}")
    return CalendarCommandResult(message="\n".join(lines))


def _handle_calendar_create(context: CalendarCommandContext) -> CalendarCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 4)
    if fields is None:
        return CalendarCommandResult(message=CALENDAR_CREATE_USAGE)
    calendar_token, title, start_local, end_local = fields
    if not title or not start_local or not end_local:
        return CalendarCommandResult(message=CALENDAR_CREATE_USAGE)
    calendar_id = None if calendar_token == "-" else calendar_token
    proposal = context.calendar_service.prepare_create(
        title=title,
        start_local=start_local,
        end_local=end_local,
        calendar_id=calendar_id,
    )
    return CalendarCommandResult(message=_format_proposal_summary(proposal))


def _handle_calendar_reschedule(
    context: CalendarCommandContext,
) -> CalendarCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 4)
    if fields is None:
        return CalendarCommandResult(message=CALENDAR_RESCHEDULE_USAGE)
    calendar_token, event_id, start_local, end_local = fields
    if not event_id or not start_local or not end_local:
        return CalendarCommandResult(message=CALENDAR_RESCHEDULE_USAGE)
    calendar_id = None if calendar_token == "-" else calendar_token
    proposal = context.calendar_service.prepare_reschedule(
        event_id=event_id,
        start_local=start_local,
        end_local=end_local,
        calendar_id=calendar_id,
    )
    return CalendarCommandResult(message=_format_proposal_summary(proposal))


def _handle_calendar_cancel(context: CalendarCommandContext) -> CalendarCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 2)
    if fields is None:
        return CalendarCommandResult(message=CALENDAR_CANCEL_USAGE)
    calendar_token, event_id = fields
    if not event_id:
        return CalendarCommandResult(message=CALENDAR_CANCEL_USAGE)
    calendar_id = None if calendar_token == "-" else calendar_token
    proposal = context.calendar_service.prepare_cancel(
        event_id=event_id,
        calendar_id=calendar_id,
    )
    return CalendarCommandResult(message=_format_proposal_summary(proposal))


def _handle_calendar_confirm(context: CalendarCommandContext) -> CalendarCommandResult:
    proposal_id = extract_command_argument(context.message).strip()
    if not proposal_id or " | " in proposal_id:
        return CalendarCommandResult(message=CALENDAR_CONFIRM_USAGE)
    proposal = context.calendar_service.confirm(proposal_id)
    return CalendarCommandResult(
        message=(
            f"Cortana: Calendar proposal executed ({proposal.proposal_id}). "
            f"operation={proposal.operation} status={proposal.status}"
        )
    )


def _format_proposal_summary(proposal: CalendarChangeProposal) -> str:
    payload = proposal.normalized_payload
    lines = [
        f"Cortana: Calendar change prepared ({proposal.proposal_id}).",
        f"  operation: {proposal.operation}",
        f"  calendar_id: {proposal.calendar_id}",
        f"  status: {proposal.status}",
        f"  expires_at: {proposal.expires_at}",
        "  Confirm with: "
        f"/calendar-confirm {proposal.proposal_id}",
    ]
    if proposal.operation == "create":
        tz = payload["timezone"]
        lines.extend(
            [
                f"  client_event_id: {proposal.client_event_id}",
                f"  title: {_preview(payload['title'])}",
                f"  start_local: {format_local_wall_from_utc(payload['start_utc'], tz)}",
                f"  end_local: {format_local_wall_from_utc(payload['end_utc'], tz)}",
                f"  timezone: {tz}",
            ]
        )
    elif proposal.operation == "reschedule":
        tz = payload["timezone"]
        lines.extend(
            [
                f"  event_id: {proposal.event_id}",
                f"  start_local: {format_local_wall_from_utc(payload['start_utc'], tz)}",
                f"  end_local: {format_local_wall_from_utc(payload['end_utc'], tz)}",
                f"  timezone: {tz}",
            ]
        )
    else:
        lines.append(f"  event_id: {proposal.event_id}")
    return "\n".join(lines)


def _format_event_line(event: CalendarEvent) -> str:
    kind = "all-day" if event.is_all_day else "timed"
    recurring = " recurring" if event.is_recurring else ""
    return (
        f"  [{event.event_id}] {_preview(event.title)} "
        f"{event.start} -> {event.end} ({kind}{recurring})"
    )


def _preview(text: str, *, max_chars: int = MAX_CALENDAR_LIST_PREVIEW_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3] + "..."


CALENDAR_COMMAND_HANDLERS: dict[str, CalendarCommandHandler] = {
    COMMAND_CALENDAR_CONNECT: _handle_calendar_connect,
    COMMAND_CALENDAR_DISCONNECT: _handle_calendar_disconnect,
    COMMAND_CALENDARS: _handle_calendars,
    COMMAND_CALENDAR_USE: _handle_calendar_use,
    COMMAND_CALENDAR_EVENTS: _handle_calendar_events,
    COMMAND_CALENDAR_EVENT: _handle_calendar_event,
    COMMAND_CALENDAR_FREEBUSY: _handle_calendar_freebusy,
    COMMAND_CALENDAR_CREATE: _handle_calendar_create,
    COMMAND_CALENDAR_RESCHEDULE: _handle_calendar_reschedule,
    COMMAND_CALENDAR_CANCEL: _handle_calendar_cancel,
    COMMAND_CALENDAR_CONFIRM: _handle_calendar_confirm,
}
