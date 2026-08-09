"""Persistent reminder/scheduling domain models for Milestone 19.

Reminders are scheduled obligations, not memories. Title and message text are
inert user-controlled display data and must never be executed or treated as
trusted instructions.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config import (
    MAX_REMINDER_DAILY_INTERVAL,
    MAX_REMINDER_MESSAGE_LENGTH,
    MAX_REMINDER_MONTHLY_INTERVAL,
    MAX_REMINDER_RECURRENCE_SEARCH_STEPS,
    MAX_REMINDER_TITLE_LENGTH,
    MAX_REMINDER_WEEKLY_INTERVAL,
)
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolEnumError,
    InvalidToolIdError,
    InvalidToolTimestampError,
    ToolFieldTooLongError,
    ToolValidationError,
    parse_utc_timestamp,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_utc_timestamp,
    validate_utc_timestamp,
    validate_uuid,
)

ReminderStatus = Literal["scheduled", "completed", "cancelled"]
RecurrenceType = Literal["none", "daily", "weekly", "monthly"]
ReminderAuditAction = Literal[
    "created",
    "rescheduled",
    "snoozed",
    "completed",
    "occurrence_advanced",
    "cancelled",
]

REMINDER_STATUSES: frozenset[str] = frozenset(
    {"scheduled", "completed", "cancelled"}
)
RECURRENCE_TYPES: frozenset[str] = frozenset(
    {"none", "daily", "weekly", "monthly"}
)
REMINDER_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "created",
        "rescheduled",
        "snoozed",
        "completed",
        "occurrence_advanced",
        "cancelled",
    }
)
REMINDER_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "cancelled"}
)

WEEKDAY_TOKEN_TO_INT: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
WEEKDAY_INT_TO_TOKEN: dict[int, str] = {
    value: key for key, value in WEEKDAY_TOKEN_TO_INT.items()
}

LOCAL_WALL_FORMAT = "%Y-%m-%d %H:%M"


class ReminderValidationError(ValueError):
    """Raised when reminder domain validation fails."""


class ReminderSchedulingError(RuntimeError):
    """Raised when a bounded recurrence search cannot find a valid occurrence."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or (
            "Cortana: Reminder scheduling could not compute a valid next occurrence."
        )


class ReminderClock(Protocol):
    """Injectable clock for deterministic reminder scheduling tests."""

    def utc_now_iso(self) -> str:
        """Return the current UTC timestamp in ISO 8601 form with trailing Z."""


class SystemReminderClock:
    """Production clock backed by the system UTC clock."""

    def utc_now_iso(self) -> str:
        return utc_timestamp()


class ManualReminderClock:
    """Test clock with an explicitly controllable UTC instant."""

    def __init__(self, *, start: datetime | None = None) -> None:
        if start is None:
            self._utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
        else:
            if start.tzinfo is None:
                raise ReminderValidationError("ManualReminderClock start must be aware.")
            self._utc = start.astimezone(timezone.utc)

    def utc_now_iso(self) -> str:
        return self._utc.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def set_utc(self, value: datetime) -> None:
        """Set the clock to an aware UTC datetime."""
        if value.tzinfo is None:
            raise ReminderValidationError("ManualReminderClock values must be aware.")
        self._utc = value.astimezone(timezone.utc)

    def advance(self, seconds: float) -> None:
        """Advance the clock by the given number of seconds."""
        self._utc = self._utc + timedelta(seconds=float(seconds))


@dataclass(frozen=True)
class Reminder:
    """Immutable scheduled obligation record."""

    reminder_id: str
    title: str
    message: str | None
    status: ReminderStatus
    due_at: str
    timezone: str
    recurrence_type: RecurrenceType
    recurrence_interval: int
    recurrence_weekdays: tuple[int, ...]
    recurrence_anchor_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ReminderAuditEntry:
    """Bounded lifecycle audit record for one reminder mutation."""

    audit_id: str
    reminder_id: str
    action: ReminderAuditAction
    occurred_at: str
    from_status: str | None
    to_status: str | None
    old_due_at: str | None
    new_due_at: str | None
    skipped_occurrences: int | None


def validate_iana_timezone(value: str, *, field_name: str = "Timezone") -> str:
    """Validate and return an IANA timezone identifier resolvable by ZoneInfo."""
    cleaned = require_non_blank_text(
        value,
        field_name=field_name,
        max_length=100,
    )
    try:
        ZoneInfo(cleaned)
    except ZoneInfoNotFoundError as error:
        raise ReminderValidationError(
            f"{field_name} must be a valid IANA timezone identifier."
        ) from error
    except Exception as error:
        # ZoneInfo may raise other errors for malformed keys on some platforms.
        raise ReminderValidationError(
            f"{field_name} must be a valid IANA timezone identifier."
        ) from error
    return cleaned


def parse_local_wall_datetime(value: str, *, field_name: str = "Local time") -> datetime:
    """Parse a local wall clock string as naive datetime (YYYY-MM-DD HH:MM)."""
    cleaned = require_non_blank_text(value, field_name=field_name, max_length=32)
    if "Z" in cleaned.upper() or "+" in cleaned:
        raise ReminderValidationError(
            f"{field_name} must be local wall time YYYY-MM-DD HH:MM without "
            "timezone offset."
        )
    try:
        return datetime.strptime(cleaned, LOCAL_WALL_FORMAT)
    except ValueError as error:
        raise ReminderValidationError(
            f"{field_name} must use YYYY-MM-DD HH:MM format."
        ) from error


def local_wall_to_utc_iso(
    local_wall: datetime,
    timezone_name: str,
) -> str:
    """Convert a naive local wall datetime in an IANA zone to UTC ISO Z."""
    if local_wall.tzinfo is not None:
        raise ReminderValidationError("Local wall datetime must be naive.")
    zone = ZoneInfo(validate_iana_timezone(timezone_name))
    # fold=0 selects the earlier offset during ambiguous fall-back periods.
    aware_local = local_wall.replace(tzinfo=zone, fold=0)
    utc_value = aware_local.astimezone(timezone.utc)
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def utc_iso_to_local(utc_iso: str, timezone_name: str) -> datetime:
    """Convert a UTC ISO Z timestamp into an aware datetime in the given zone."""
    utc_value = parse_utc_timestamp(utc_iso)
    zone = ZoneInfo(validate_iana_timezone(timezone_name))
    return utc_value.astimezone(zone)


def format_local_wall(local_aware: datetime) -> str:
    """Format an aware local datetime as YYYY-MM-DD HH:MM."""
    return local_aware.strftime(LOCAL_WALL_FORMAT)


def validate_recurrence_weekdays(
    weekdays: tuple[int, ...] | list[int],
    *,
    recurrence_type: str,
) -> tuple[int, ...]:
    """Validate weekday integers for weekly recurrence."""
    if recurrence_type != "weekly":
        if weekdays:
            raise ReminderValidationError(
                "Recurrence weekdays are only valid for weekly recurrence."
            )
        return ()

    seen: set[int] = set()
    normalized: list[int] = []
    for day in weekdays:
        if not isinstance(day, int) or isinstance(day, bool):
            raise ReminderValidationError(
                "Recurrence weekdays must be integers from 0 (Monday) to 6 (Sunday)."
            )
        if day < 0 or day > 6:
            raise ReminderValidationError(
                "Recurrence weekdays must be integers from 0 (Monday) to 6 (Sunday)."
            )
        if day in seen:
            raise ReminderValidationError(
                "Recurrence weekdays must be unique."
            )
        seen.add(day)
        normalized.append(day)
    return tuple(normalized)


def validate_recurrence_interval(interval: int, *, recurrence_type: str) -> int:
    """Validate recurrence interval bounds for the given recurrence type."""
    if not isinstance(interval, int) or isinstance(interval, bool):
        raise ReminderValidationError("Recurrence interval must be an integer.")
    if recurrence_type == "none":
        if interval != 1:
            raise ReminderValidationError(
                "Non-recurring reminders must use recurrence_interval 1."
            )
        return 1
    if interval < 1:
        raise ReminderValidationError("Recurrence interval must be at least 1.")
    if recurrence_type == "daily" and interval > MAX_REMINDER_DAILY_INTERVAL:
        raise ReminderValidationError(
            f"Daily recurrence interval cannot exceed {MAX_REMINDER_DAILY_INTERVAL}."
        )
    if recurrence_type == "weekly" and interval > MAX_REMINDER_WEEKLY_INTERVAL:
        raise ReminderValidationError(
            f"Weekly recurrence interval cannot exceed {MAX_REMINDER_WEEKLY_INTERVAL}."
        )
    if recurrence_type == "monthly" and interval > MAX_REMINDER_MONTHLY_INTERVAL:
        raise ReminderValidationError(
            f"Monthly recurrence interval cannot exceed {MAX_REMINDER_MONTHLY_INTERVAL}."
        )
    return interval


def parse_recurrence_spec(value: str) -> tuple[RecurrenceType, int, tuple[int, ...]]:
    """Parse a command recurrence token into type, interval, and weekdays."""
    cleaned = require_non_blank_text(
        value,
        field_name="Recurrence",
        max_length=100,
    ).lower()
    parts = cleaned.split(":")
    recurrence_type = validate_controlled_value(
        parts[0],
        field_name="recurrence type",
        allowed=RECURRENCE_TYPES,
    )
    if recurrence_type == "none":
        if len(parts) != 1:
            raise ReminderValidationError(
                "Recurrence 'none' does not accept interval or weekdays."
            )
        return "none", 1, ()

    interval = 1
    weekdays: tuple[int, ...] = ()
    if recurrence_type == "daily":
        if len(parts) > 2:
            raise ReminderValidationError(
                "Daily recurrence accepts at most daily[:interval]."
            )
        if len(parts) == 2:
            interval = _parse_positive_int(parts[1], field_name="Daily interval")
    elif recurrence_type == "monthly":
        if len(parts) > 2:
            raise ReminderValidationError(
                "Monthly recurrence accepts at most monthly[:interval]."
            )
        if len(parts) == 2:
            interval = _parse_positive_int(parts[1], field_name="Monthly interval")
    elif recurrence_type == "weekly":
        if len(parts) > 3:
            raise ReminderValidationError(
                "Weekly recurrence accepts weekly[:interval][:weekdays]."
            )
        if len(parts) >= 2 and parts[1]:
            interval = _parse_positive_int(parts[1], field_name="Weekly interval")
        if len(parts) == 3:
            weekdays = _parse_weekday_tokens(parts[2])
    interval = validate_recurrence_interval(interval, recurrence_type=recurrence_type)
    weekdays = validate_recurrence_weekdays(weekdays, recurrence_type=recurrence_type)
    return recurrence_type, interval, weekdays  # type: ignore[return-value]


def format_recurrence_spec(reminder: Reminder) -> str:
    """Format a reminder's recurrence fields as a command-style token."""
    if reminder.recurrence_type == "none":
        return "none"
    if reminder.recurrence_type == "daily":
        if reminder.recurrence_interval == 1:
            return "daily"
        return f"daily:{reminder.recurrence_interval}"
    if reminder.recurrence_type == "monthly":
        if reminder.recurrence_interval == 1:
            return "monthly"
        return f"monthly:{reminder.recurrence_interval}"
    # weekly
    tokens = [WEEKDAY_INT_TO_TOKEN[day] for day in reminder.recurrence_weekdays]
    if reminder.recurrence_interval == 1 and not tokens:
        return "weekly"
    if tokens:
        return (
            f"weekly:{reminder.recurrence_interval}:{','.join(tokens)}"
        )
    return f"weekly:{reminder.recurrence_interval}"


def validate_reminder(reminder: Reminder) -> Reminder:
    """Validate and return a normalized Reminder."""
    try:
        reminder_id = validate_uuid(reminder.reminder_id, field_name="Reminder ID")
        title = require_non_blank_text(
            reminder.title,
            field_name="Title",
            max_length=MAX_REMINDER_TITLE_LENGTH,
        )
        message = _validate_optional_message(reminder.message)
        status = validate_controlled_value(
            reminder.status,
            field_name="reminder status",
            allowed=REMINDER_STATUSES,
        )
        due_at = validate_utc_timestamp(reminder.due_at, field_name="Due at")
        timezone_name = validate_iana_timezone(reminder.timezone)
        recurrence_type = validate_controlled_value(
            reminder.recurrence_type,
            field_name="recurrence type",
            allowed=RECURRENCE_TYPES,
        )
        recurrence_interval = validate_recurrence_interval(
            reminder.recurrence_interval,
            recurrence_type=recurrence_type,
        )
        recurrence_weekdays = validate_recurrence_weekdays(
            reminder.recurrence_weekdays,
            recurrence_type=recurrence_type,
        )
        created_at = validate_utc_timestamp(
            reminder.created_at,
            field_name="Created at",
        )
        updated_at = validate_utc_timestamp(
            reminder.updated_at,
            field_name="Updated at",
        )
        if recurrence_type == "none":
            if reminder.recurrence_anchor_at is not None:
                raise ReminderValidationError(
                    "Non-recurring reminders must not set recurrence_anchor_at."
                )
            recurrence_anchor_at = None
        else:
            if reminder.recurrence_anchor_at is None:
                raise ReminderValidationError(
                    "Recurring reminders require recurrence_anchor_at."
                )
            recurrence_anchor_at = validate_utc_timestamp(
                reminder.recurrence_anchor_at,
                field_name="Recurrence anchor",
            )
    except (
        BlankToolFieldError,
        ToolFieldTooLongError,
        InvalidToolEnumError,
        InvalidToolIdError,
        InvalidToolTimestampError,
        ToolValidationError,
    ) as error:
        raise ReminderValidationError(str(error)) from error

    return Reminder(
        reminder_id=reminder_id,
        title=title,
        message=message,
        status=status,  # type: ignore[arg-type]
        due_at=due_at,
        timezone=timezone_name,
        recurrence_type=recurrence_type,  # type: ignore[arg-type]
        recurrence_interval=recurrence_interval,
        recurrence_weekdays=recurrence_weekdays,
        recurrence_anchor_at=recurrence_anchor_at,
        created_at=created_at,
        updated_at=updated_at,
    )


def create_reminder(
    *,
    title: str,
    local_due: datetime | str,
    timezone_name: str,
    message: str | None = None,
    recurrence_type: str = "none",
    recurrence_interval: int = 1,
    recurrence_weekdays: tuple[int, ...] = (),
    clock: ReminderClock | None = None,
) -> Reminder:
    """Create a validated scheduled reminder from local wall time + IANA zone."""
    active_clock = clock or SystemReminderClock()
    now = active_clock.utc_now_iso()
    zone = validate_iana_timezone(timezone_name)
    if isinstance(local_due, str):
        wall = parse_local_wall_datetime(local_due)
    else:
        wall = local_due
    due_at = local_wall_to_utc_iso(wall, zone)
    recurrence = validate_controlled_value(
        recurrence_type,
        field_name="recurrence type",
        allowed=RECURRENCE_TYPES,
    )
    interval = validate_recurrence_interval(
        recurrence_interval,
        recurrence_type=recurrence,
    )
    weekdays = validate_recurrence_weekdays(
        recurrence_weekdays,
        recurrence_type=recurrence,
    )
    anchor = due_at if recurrence != "none" else None
    return validate_reminder(
        Reminder(
            reminder_id=str(uuid4()),
            title=title,
            message=message,
            status="scheduled",
            due_at=due_at,
            timezone=zone,
            recurrence_type=recurrence,  # type: ignore[arg-type]
            recurrence_interval=interval,
            recurrence_weekdays=weekdays,
            recurrence_anchor_at=anchor,
            created_at=now,
            updated_at=now,
        )
    )


def replace_reminder(
    reminder: Reminder,
    *,
    title: str | None = None,
    message: str | None | object = ...,
    status: str | None = None,
    due_at: str | None = None,
    timezone_name: str | None = None,
    recurrence_type: str | None = None,
    recurrence_interval: int | None = None,
    recurrence_weekdays: tuple[int, ...] | None = None,
    recurrence_anchor_at: str | None | object = ...,
    updated_at: str | None = None,
    clock: ReminderClock | None = None,
) -> Reminder:
    """Return a validated replacement reminder with selected fields updated."""
    active_clock = clock or SystemReminderClock()
    next_message: str | None
    if message is ...:
        next_message = reminder.message
    else:
        next_message = message  # type: ignore[assignment]

    next_anchor: str | None
    if recurrence_anchor_at is ...:
        next_anchor = reminder.recurrence_anchor_at
    else:
        next_anchor = recurrence_anchor_at  # type: ignore[assignment]

    return validate_reminder(
        Reminder(
            reminder_id=reminder.reminder_id,
            title=reminder.title if title is None else title,
            message=next_message,
            status=(
                reminder.status
                if status is None
                else status  # type: ignore[arg-type]
            ),
            due_at=reminder.due_at if due_at is None else due_at,
            timezone=reminder.timezone if timezone_name is None else timezone_name,
            recurrence_type=(
                reminder.recurrence_type
                if recurrence_type is None
                else recurrence_type  # type: ignore[arg-type]
            ),
            recurrence_interval=(
                reminder.recurrence_interval
                if recurrence_interval is None
                else recurrence_interval
            ),
            recurrence_weekdays=(
                reminder.recurrence_weekdays
                if recurrence_weekdays is None
                else recurrence_weekdays
            ),
            recurrence_anchor_at=next_anchor,
            created_at=reminder.created_at,
            updated_at=updated_at or active_clock.utc_now_iso(),
        )
    )


def create_reminder_audit_entry(
    *,
    reminder_id: str,
    action: str,
    occurred_at: str | None = None,
    from_status: str | None = None,
    to_status: str | None = None,
    old_due_at: str | None = None,
    new_due_at: str | None = None,
    skipped_occurrences: int | None = None,
    clock: ReminderClock | None = None,
) -> ReminderAuditEntry:
    """Create one validated reminder audit entry."""
    active_clock = clock or SystemReminderClock()
    return validate_reminder_audit_entry(
        ReminderAuditEntry(
            audit_id=str(uuid4()),
            reminder_id=reminder_id,
            action=action,  # type: ignore[arg-type]
            occurred_at=occurred_at or active_clock.utc_now_iso(),
            from_status=from_status,
            to_status=to_status,
            old_due_at=old_due_at,
            new_due_at=new_due_at,
            skipped_occurrences=skipped_occurrences,
        )
    )


def validate_reminder_audit_entry(entry: ReminderAuditEntry) -> ReminderAuditEntry:
    """Validate and return a normalized ReminderAuditEntry."""
    try:
        audit_id = validate_uuid(entry.audit_id, field_name="Audit ID")
        reminder_id = validate_uuid(entry.reminder_id, field_name="Reminder ID")
        action = validate_controlled_value(
            entry.action,
            field_name="reminder audit action",
            allowed=REMINDER_AUDIT_ACTIONS,
        )
        occurred_at = validate_utc_timestamp(
            entry.occurred_at,
            field_name="Occurred at",
        )
        from_status = _validate_optional_status(entry.from_status, "from_status")
        to_status = _validate_optional_status(entry.to_status, "to_status")
        old_due_at = validate_optional_utc_timestamp(
            entry.old_due_at,
            field_name="Old due at",
        )
        new_due_at = validate_optional_utc_timestamp(
            entry.new_due_at,
            field_name="New due at",
        )
    except (
        BlankToolFieldError,
        ToolFieldTooLongError,
        InvalidToolEnumError,
        InvalidToolIdError,
        InvalidToolTimestampError,
        ToolValidationError,
    ) as error:
        raise ReminderValidationError(str(error)) from error

    skipped = entry.skipped_occurrences
    if skipped is not None:
        if not isinstance(skipped, int) or isinstance(skipped, bool) or skipped < 0:
            raise ReminderValidationError(
                "skipped_occurrences must be a non-negative integer or None."
            )

    return ReminderAuditEntry(
        audit_id=audit_id,
        reminder_id=reminder_id,
        action=action,  # type: ignore[arg-type]
        occurred_at=occurred_at,
        from_status=from_status,
        to_status=to_status,
        old_due_at=old_due_at,
        new_due_at=new_due_at,
        skipped_occurrences=skipped,
    )


def is_overdue(reminder: Reminder, *, now_utc_iso: str) -> bool:
    """Return True when a scheduled reminder's due_at is at or before now."""
    if reminder.status != "scheduled":
        return False
    return parse_utc_timestamp(reminder.due_at) <= parse_utc_timestamp(now_utc_iso)


def monday_of_week(day: date) -> date:
    """Return the Monday that starts the ISO-like week containing day."""
    return day - timedelta(days=day.weekday())


def weeks_between(start_monday: date, end_monday: date) -> int:
    """Return whole weeks between two Monday week-start dates."""
    return (end_monday - start_monday).days // 7


def add_months(year: int, month: int, months: int) -> tuple[int, int]:
    """Add months to a year/month pair without day clamping."""
    total = (year * 12 + (month - 1)) + months
    return total // 12, (total % 12) + 1


def compute_next_occurrence_after(
    reminder: Reminder,
    *,
    after_utc_iso: str,
    max_steps: int = MAX_REMINDER_RECURRENCE_SEARCH_STEPS,
) -> str:
    """Return the first recurrence UTC due time strictly after after_utc_iso.

    Calculations use the series recurrence_anchor_at and preserve local wall
    clock time in Reminder.timezone. The search is hard-bounded.
    """
    if reminder.recurrence_type == "none":
        raise ReminderSchedulingError(
            "Cannot compute next occurrence for a non-recurring reminder."
        )
    if reminder.recurrence_anchor_at is None:
        raise ReminderSchedulingError(
            "Recurring reminder is missing recurrence_anchor_at."
        )
    if max_steps < 1:
        raise ReminderSchedulingError("Recurrence search bound must be at least 1.")

    after_utc = parse_utc_timestamp(after_utc_iso)
    zone_name = reminder.timezone
    anchor_local = utc_iso_to_local(reminder.recurrence_anchor_at, zone_name)
    wall_time = time(
        hour=anchor_local.hour,
        minute=anchor_local.minute,
        second=anchor_local.second,
        microsecond=anchor_local.microsecond,
    )

    if reminder.recurrence_type == "daily":
        return _next_daily_after(
            after_utc=after_utc,
            anchor_local=anchor_local,
            wall_time=wall_time,
            timezone_name=zone_name,
            interval=reminder.recurrence_interval,
            max_steps=max_steps,
        )
    if reminder.recurrence_type == "weekly":
        return _next_weekly_after(
            after_utc=after_utc,
            anchor_local=anchor_local,
            wall_time=wall_time,
            timezone_name=zone_name,
            interval=reminder.recurrence_interval,
            weekdays=reminder.recurrence_weekdays,
            max_steps=max_steps,
        )
    return _next_monthly_after(
        after_utc=after_utc,
        anchor_local=anchor_local,
        wall_time=wall_time,
        timezone_name=zone_name,
        interval=reminder.recurrence_interval,
        max_steps=max_steps,
    )


def count_skipped_occurrences(
    reminder: Reminder,
    *,
    after_due_utc_iso: str,
    until_utc_iso: str,
    max_steps: int = MAX_REMINDER_RECURRENCE_SEARCH_STEPS,
) -> int:
    """Count recurrence slots in (after_due, until] for bounded skipped audit."""
    if reminder.recurrence_type == "none":
        return 0
    count = 0
    cursor = after_due_utc_iso
    until = parse_utc_timestamp(until_utc_iso)
    for _ in range(max_steps):
        try:
            nxt = compute_next_occurrence_after(
                reminder,
                after_utc_iso=cursor,
                max_steps=max_steps,
            )
        except ReminderSchedulingError:
            break
        nxt_dt = parse_utc_timestamp(nxt)
        if nxt_dt > until:
            break
        count += 1
        cursor = nxt
    return count


def _next_daily_after(
    *,
    after_utc: datetime,
    anchor_local: datetime,
    wall_time: time,
    timezone_name: str,
    interval: int,
    max_steps: int,
) -> str:
    after_local = after_utc.astimezone(ZoneInfo(timezone_name))
    anchor_date = anchor_local.date()
    start_date = after_local.date()
    for step in range(max_steps):
        candidate_date = start_date + timedelta(days=step)
        day_delta = (candidate_date - anchor_date).days
        if day_delta < 0:
            continue
        if day_delta % interval != 0:
            continue
        candidate_utc = _local_date_wall_to_utc(
            candidate_date,
            wall_time,
            timezone_name,
        )
        if candidate_utc > after_utc:
            return candidate_utc.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            )
    raise ReminderSchedulingError(
        "Bounded daily recurrence search found no valid next occurrence.",
    )


def _next_weekly_after(
    *,
    after_utc: datetime,
    anchor_local: datetime,
    wall_time: time,
    timezone_name: str,
    interval: int,
    weekdays: tuple[int, ...],
    max_steps: int,
) -> str:
    selected = weekdays or (anchor_local.weekday(),)
    anchor_week_start = monday_of_week(anchor_local.date())
    after_local = after_utc.astimezone(ZoneInfo(timezone_name))
    start_date = after_local.date()
    for step in range(max_steps):
        candidate_date = start_date + timedelta(days=step)
        if candidate_date.weekday() not in selected:
            continue
        candidate_week_start = monday_of_week(candidate_date)
        week_delta = weeks_between(anchor_week_start, candidate_week_start)
        if week_delta < 0:
            continue
        if week_delta % interval != 0:
            continue
        candidate_utc = _local_date_wall_to_utc(
            candidate_date,
            wall_time,
            timezone_name,
        )
        if candidate_utc > after_utc:
            return candidate_utc.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            )
    raise ReminderSchedulingError(
        "Bounded weekly recurrence search found no valid next occurrence.",
    )


def _next_monthly_after(
    *,
    after_utc: datetime,
    anchor_local: datetime,
    wall_time: time,
    timezone_name: str,
    interval: int,
    max_steps: int,
) -> str:
    anchor_day = anchor_local.day
    # Start N near the after month relative to anchor, then search forward.
    after_local = after_utc.astimezone(ZoneInfo(timezone_name))
    months_after_anchor = (
        (after_local.year - anchor_local.year) * 12
        + (after_local.month - anchor_local.month)
    )
    start_n = max(0, months_after_anchor // interval - 1)
    for step in range(max_steps):
        n = start_n + step
        year, month = add_months(anchor_local.year, anchor_local.month, n * interval)
        day = min(anchor_day, calendar.monthrange(year, month)[1])
        candidate_utc = _local_date_wall_to_utc(
            date(year, month, day),
            wall_time,
            timezone_name,
        )
        if candidate_utc > after_utc:
            return candidate_utc.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            )
    raise ReminderSchedulingError(
        "Bounded monthly recurrence search found no valid next occurrence.",
    )


def _local_date_wall_to_utc(
    day: date,
    wall_time: time,
    timezone_name: str,
) -> datetime:
    local_naive = datetime(
        day.year,
        day.month,
        day.day,
        wall_time.hour,
        wall_time.minute,
        wall_time.second,
        wall_time.microsecond,
    )
    iso = local_wall_to_utc_iso(local_naive, timezone_name)
    return parse_utc_timestamp(iso)


def _validate_optional_message(message: str | None) -> str | None:
    if message is None:
        return None
    if not isinstance(message, str):
        raise ReminderValidationError("Message must be a string or None.")
    cleaned = message.strip()
    if not cleaned:
        raise ReminderValidationError("Message cannot be blank when provided.")
    if len(cleaned) > MAX_REMINDER_MESSAGE_LENGTH:
        raise ReminderValidationError(
            f"Message exceeds the maximum length of {MAX_REMINDER_MESSAGE_LENGTH} "
            "characters."
        )
    return cleaned


def _validate_optional_status(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_controlled_value(
        value,
        field_name=field_name,
        allowed=REMINDER_STATUSES,
    )


def _parse_positive_int(value: str, *, field_name: str) -> int:
    cleaned = value.strip()
    if not cleaned.isdigit():
        raise ReminderValidationError(f"{field_name} must be a positive integer.")
    parsed = int(cleaned)
    if parsed < 1:
        raise ReminderValidationError(f"{field_name} must be a positive integer.")
    return parsed


def _parse_weekday_tokens(value: str) -> tuple[int, ...]:
    tokens = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not tokens:
        raise ReminderValidationError("Weekly weekday list cannot be empty.")
    weekdays: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if token not in WEEKDAY_TOKEN_TO_INT:
            raise ReminderValidationError(
                "Weekdays must be mon,tue,wed,thu,fri,sat,sun."
            )
        day = WEEKDAY_TOKEN_TO_INT[token]
        if day in seen:
            raise ReminderValidationError("Recurrence weekdays must be unique.")
        seen.add(day)
        weekdays.append(day)
    return tuple(weekdays)
