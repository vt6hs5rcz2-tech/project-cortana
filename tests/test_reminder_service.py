"""Tests for Milestone 19 ReminderService lifecycle and recurrence behavior."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.config import PROJECT_ROOT
from src.reminder import ManualReminderClock, utc_iso_to_local
from src.reminder_repository import JsonReminderRepository
from src.reminder_service import ReminderService, ReminderServiceError

REMINDER_MODULES = (
    PROJECT_ROOT / "src" / "reminder.py",
    PROJECT_ROOT / "src" / "reminder_repository.py",
    PROJECT_ROOT / "src" / "reminder_service.py",
    PROJECT_ROOT / "src" / "reminder_commands.py",
)


def _service(tmp_path: Path, *, when: datetime | None = None) -> ReminderService:
    clock = ManualReminderClock(
        start=when or datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    )
    repo = JsonReminderRepository(tmp_path / "reminders.json")
    return ReminderService(repo, clock=clock)


def test_one_time_create_overdue_complete_cancel_snooze_reschedule(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    reminder = service.create(
        title="Call dentist",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="none",
        message=None,
    )
    assert reminder.status == "scheduled"
    view = service.get(reminder.reminder_id)
    assert view is not None
    assert view.overdue is False

    cast_clock = service.clock
    assert isinstance(cast_clock, ManualReminderClock)
    cast_clock.set_utc(datetime(2026, 8, 10, 14, 0, tzinfo=timezone.utc))
    overdue_view = service.get(reminder.reminder_id)
    assert overdue_view is not None
    assert overdue_view.overdue is True

    snoozed = service.snooze(reminder.reminder_id, local_until="2026-08-10 11:00")
    assert snoozed.due_at == "2026-08-10T15:00:00.000000Z"
    assert snoozed.status == "scheduled"

    rescheduled = service.reschedule(
        reminder.reminder_id,
        local_due="2026-08-11 09:00",
        timezone_name=None,
    )
    assert rescheduled.timezone == "America/New_York"
    assert rescheduled.due_at.endswith("Z")

    completed = service.complete(reminder.reminder_id)
    assert completed.status == "completed"
    with pytest.raises(ReminderServiceError):
        service.complete(reminder.reminder_id)
    with pytest.raises(ReminderServiceError):
        service.cancel(reminder.reminder_id)
    with pytest.raises(ReminderServiceError):
        service.snooze(reminder.reminder_id, local_until="2026-08-12 09:00")
    with pytest.raises(ReminderServiceError):
        service.reschedule(
            reminder.reminder_id,
            local_due="2026-08-12 09:00",
            timezone_name=None,
        )

    other = service.create(
        title="Cancel me",
        local_due="2026-08-12 09:00",
        timezone_name="America/New_York",
        recurrence_type="none",
    )
    cancelled = service.cancel(other.reminder_id)
    assert cancelled.status == "cancelled"
    with pytest.raises(ReminderServiceError):
        service.complete(other.reminder_id)


def test_daily_recurrence_completion_jumps_to_first_future(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        when=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
    )
    reminder = service.create(
        title="Meds",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="daily",
    )
    # Jump clock to Friday after several missed days.
    clock = service.clock
    assert isinstance(clock, ManualReminderClock)
    clock.set_utc(datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc))  # Friday 11:00 EDT

    advanced = service.complete(reminder.reminder_id)
    assert advanced.status == "scheduled"
    local = utc_iso_to_local(advanced.due_at, advanced.timezone)
    assert (local.year, local.month, local.day, local.hour) == (2026, 8, 15, 9)

    audits = service.list_audits(reminder.reminder_id, limit=10)
    actions = [entry.action for entry in audits]
    assert "completed" in actions
    assert "occurrence_advanced" in actions
    completed_audit = next(entry for entry in audits if entry.action == "completed")
    assert completed_audit.skipped_occurrences is not None
    assert completed_audit.skipped_occurrences >= 1


def test_snooze_preserves_recurrence_anchor(tmp_path: Path) -> None:
    service = _service(tmp_path)
    reminder = service.create(
        title="Weekly report",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="weekly",
        recurrence_interval=1,
        recurrence_weekdays=(0,),
    )
    anchor = reminder.recurrence_anchor_at
    snoozed = service.snooze(reminder.reminder_id, local_until="2026-08-10 11:00")
    assert snoozed.recurrence_anchor_at == anchor
    assert snoozed.due_at != reminder.due_at
    assert snoozed.recurrence_type == "weekly"

    clock = service.clock
    assert isinstance(clock, ManualReminderClock)
    clock.set_utc(datetime(2026, 8, 10, 16, 0, tzinfo=timezone.utc))
    advanced = service.complete(snoozed.reminder_id)
    next_local = utc_iso_to_local(advanced.due_at, advanced.timezone)
    assert next_local.weekday() == 0
    assert next_local.hour == 9
    assert advanced.recurrence_anchor_at == anchor


def test_reschedule_resets_anchor_and_optional_timezone(tmp_path: Path) -> None:
    service = _service(tmp_path)
    reminder = service.create(
        title="Weekly",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="weekly",
        recurrence_weekdays=(0,),
    )
    moved = service.reschedule(
        reminder.reminder_id,
        local_due="2026-08-17 10:00",
        timezone_name="America/Chicago",
    )
    assert moved.timezone == "America/Chicago"
    assert moved.recurrence_anchor_at == moved.due_at
    assert moved.due_at != reminder.due_at

    kept = service.reschedule(
        moved.reminder_id,
        local_due="2026-08-24 10:00",
        timezone_name=None,
    )
    assert kept.timezone == "America/Chicago"
    assert kept.recurrence_anchor_at == kept.due_at


def test_list_ordered_by_due_at_then_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    later = service.create(
        title="Later",
        local_due="2026-08-12 09:00",
        timezone_name="America/New_York",
        recurrence_type="none",
    )
    earlier = service.create(
        title="Earlier",
        local_due="2026-08-11 09:00",
        timezone_name="America/New_York",
        recurrence_type="none",
    )
    listed = service.list_scheduled()
    assert [item.reminder.reminder_id for item in listed] == [
        earlier.reminder_id,
        later.reminder_id,
    ]


def test_reminder_modules_forbid_operational_imports() -> None:
    forbidden_modules = {
        "src.ai_service",
        "src.openai_client",
        "src.incident_analysis_service",
        "src.tool_executor",
        "src.workflow_executor",
        "src.memory_store",
        "src.incident_repository",
        "src.evidence_store",
    }
    forbidden_names = {
        "MemoryStore",
        "JsonMemoryStore",
        "OpenAIClient",
        "DefensiveToolExecutor",
        "WorkflowExecutor",
        "IncidentAnalysisService",
        "generate_response",
    }
    for path in REMINDER_MODULES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                pytest.fail(f"{path} imports {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        pytest.fail(f"{path} imports {alias.name}")
        for name in forbidden_names:
            assert name not in source, f"{path} references {name}"


def test_reminder_storage_does_not_touch_memory_files(tmp_path: Path) -> None:
    memory_file = tmp_path / "memories.json"
    memory_file.write_text('{"memories": []}\n', encoding="utf-8")
    before = memory_file.read_text(encoding="utf-8")
    service = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(
            start=datetime(2026, 8, 1, tzinfo=timezone.utc)
        ),
    )
    service.create(
        title="Isolate",
        local_due="2026-08-10 09:00",
        timezone_name="America/New_York",
        recurrence_type="none",
    )
    assert memory_file.read_text(encoding="utf-8") == before
    assert (tmp_path / "reminders.json").exists()


def test_monthly_anchor_not_chained_from_last_due(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    reminder = service.create(
        title="Month end",
        local_due="2026-01-31 09:00",
        timezone_name="UTC",
        recurrence_type="monthly",
    )
    clock = service.clock
    assert isinstance(clock, ManualReminderClock)
    clock.set_utc(datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc))
    after_feb = service.complete(reminder.reminder_id)
    feb_local = utc_iso_to_local(after_feb.due_at, "UTC")
    assert (feb_local.month, feb_local.day) == (2, 28)

    clock.set_utc(datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
    after_mar = service.complete(after_feb.reminder_id)
    mar_local = utc_iso_to_local(after_mar.due_at, "UTC")
    assert (mar_local.month, mar_local.day) == (3, 31)


def test_dst_wall_clock_via_service_daily(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        when=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    reminder = service.create(
        title="DST",
        local_due="2026-03-07 09:00",
        timezone_name="America/New_York",
        recurrence_type="daily",
    )
    clock = service.clock
    assert isinstance(clock, ManualReminderClock)
    clock.set_utc(datetime(2026, 3, 7, 14, 0, tzinfo=timezone.utc))
    advanced = service.complete(reminder.reminder_id)
    local = datetime.fromisoformat(advanced.due_at.replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    )
    assert local.hour == 9
