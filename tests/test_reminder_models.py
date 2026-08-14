"""Tests for Milestone 19 reminder domain models and timezone/recurrence math."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.reminder import (
    ManualReminderClock,
    ReminderSchedulingError,
    ReminderValidationError,
    compute_next_occurrence_after,
    count_skipped_occurrences,
    create_reminder,
    is_overdue,
    local_wall_to_utc_iso,
    parse_local_wall_datetime,
    parse_recurrence_spec,
    replace_reminder,
    validate_iana_timezone,
    validate_reminder,
)


def test_america_new_york_resolves_with_tzdata() -> None:
    zone = ZoneInfo("America/New_York")
    assert zone.key == "America/New_York"


def test_invalid_timezone_rejected() -> None:
    with pytest.raises(ReminderValidationError):
        validate_iana_timezone("Not/A_Real_Zone")


def test_local_to_utc_and_z_persistence() -> None:
    utc_iso = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-01-15 09:00"),
        "America/New_York",
    )
    assert utc_iso.endswith("Z")
    parsed = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0
    assert parsed == datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)


def test_local_wall_rejects_offset_and_z() -> None:
    with pytest.raises(ReminderValidationError):
        parse_local_wall_datetime("2026-01-15T09:00:00Z")
    with pytest.raises(ReminderValidationError):
        parse_local_wall_datetime("2026-01-15 09:00+00:00")


def test_spring_forward_nonexistent_time_is_rejected() -> None:
    with pytest.raises(ReminderValidationError, match="does not exist"):
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-03-08 02:30"),
            "America/New_York",
        )


def test_fall_back_ambiguous_time_is_rejected() -> None:
    with pytest.raises(ReminderValidationError, match="ambiguous"):
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-11-01 01:30"),
            "America/New_York",
        )


def test_recurring_wall_clock_preserved_across_dst() -> None:
    reminder = create_reminder(
        title="DST wall clock",
        local_due="2026-03-07 09:00",
        timezone_name="America/New_York",
        recurrence_type="daily",
        clock=ManualReminderClock(
            start=datetime(2026, 3, 1, tzinfo=timezone.utc)
        ),
    )
    next_due = compute_next_occurrence_after(
        reminder,
        after_utc_iso="2026-03-07T14:00:00.000000Z",
    )
    local = datetime.fromisoformat(next_due.replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    )
    assert local.hour == 9
    assert local.minute == 0
    # After spring-forward the UTC offset should be -4 hours for 09:00 EDT.
    assert next_due == "2026-03-08T13:00:00.000000Z"


def test_create_one_time_reminder_fields() -> None:
    clock = ManualReminderClock(start=datetime(2026, 8, 1, tzinfo=timezone.utc))
    reminder = create_reminder(
        title="Call dentist",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        message=None,
        recurrence_type="none",
        clock=clock,
    )
    assert reminder.status == "scheduled"
    assert reminder.recurrence_type == "none"
    assert reminder.recurrence_anchor_at is None
    assert reminder.message is None
    assert reminder.due_at.endswith("Z")
    # 09:00 America/New_York in August is 13:00Z (EDT).
    assert is_overdue(reminder, now_utc_iso="2026-08-10T12:59:59.000000Z") is False
    assert is_overdue(reminder, now_utc_iso="2026-08-10T13:00:00.000000Z") is True


def test_recurring_requires_anchor_and_none_forbids_anchor() -> None:
    reminder = create_reminder(
        title="Daily",
        local_due="2026-08-10 08:00",
        timezone_name="America/New_York",
        recurrence_type="daily",
    )
    assert reminder.recurrence_anchor_at == reminder.due_at

    with pytest.raises(ReminderValidationError):
        validate_reminder(
            replace_reminder(
                reminder,
                recurrence_type="none",
                recurrence_anchor_at=reminder.due_at,
            )
        )


def test_parse_recurrence_specs() -> None:
    assert parse_recurrence_spec("none") == ("none", 1, ())
    assert parse_recurrence_spec("daily:2") == ("daily", 2, ())
    assert parse_recurrence_spec("weekly:1:mon,wed") == ("weekly", 1, (0, 2))
    assert parse_recurrence_spec("monthly:3") == ("monthly", 3, ())


def test_monthly_anchor_jan31_feb28_mar31() -> None:
    reminder = create_reminder(
        title="Month end",
        local_due="2026-01-31 09:00",
        timezone_name="America/New_York",
        recurrence_type="monthly",
        clock=ManualReminderClock(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )
    feb = compute_next_occurrence_after(
        reminder,
        after_utc_iso=reminder.due_at,
    )
    feb_local = datetime.fromisoformat(feb.replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    )
    assert (feb_local.year, feb_local.month, feb_local.day) == (2026, 2, 28)

    after_feb = replace_reminder(reminder, due_at=feb)
    mar = compute_next_occurrence_after(after_feb, after_utc_iso=feb)
    mar_local = datetime.fromisoformat(mar.replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    )
    assert (mar_local.year, mar_local.month, mar_local.day) == (2026, 3, 31)


def test_weekly_interval_phase_with_selected_weekdays() -> None:
    # Anchor Monday 2026-08-10; interval 2; weekdays mon,wed.
    reminder = create_reminder(
        title="Biweekly",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="weekly",
        recurrence_interval=2,
        recurrence_weekdays=(0, 2),
        clock=ManualReminderClock(
            start=datetime(2026, 8, 1, tzinfo=timezone.utc)
        ),
    )
    first = compute_next_occurrence_after(reminder, after_utc_iso=reminder.due_at)
    first_local = datetime.fromisoformat(first.replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    )
    assert first_local.weekday() == 2  # Wednesday same week
    assert first_local.day == 12

    second = compute_next_occurrence_after(reminder, after_utc_iso=first)
    second_local = datetime.fromisoformat(second.replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    )
    # Skip the intervening week; next phase Monday is 2026-08-24.
    assert (second_local.year, second_local.month, second_local.day) == (
        2026,
        8,
        24,
    )


def test_bounded_recurrence_search_fails_closed() -> None:
    reminder = create_reminder(
        title="Bound",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="daily",
    )
    with pytest.raises(ReminderSchedulingError):
        compute_next_occurrence_after(
            reminder,
            after_utc_iso=reminder.due_at,
            max_steps=1,
        )


def test_count_skipped_occurrences_bounded() -> None:
    reminder = create_reminder(
        title="Daily meds",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="daily",
        clock=ManualReminderClock(
            start=datetime(2026, 8, 1, tzinfo=timezone.utc)
        ),
    )
    skipped = count_skipped_occurrences(
        reminder,
        after_due_utc_iso=reminder.due_at,
        until_utc_iso="2026-08-13T14:00:00.000000Z",
    )
    assert skipped == 3


def test_message_blank_rejected_when_provided() -> None:
    with pytest.raises(ReminderValidationError):
        create_reminder(
            title="x",
            local_due="2026-08-10 09:00",
            timezone_name="America/New_York",
            message="   ",
        )
