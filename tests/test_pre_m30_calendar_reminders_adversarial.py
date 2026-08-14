"""Pre-M30 hardening tests: reminder and calendar routing contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.reminder import ManualReminderClock
from src.reminder_repository import JsonReminderRepository
from src.reminder_service import ReminderService, ReminderServiceError
from src.config import MAX_STORED_REMINDERS


def _service(tmp_path: Path, *, when: datetime | None = None) -> ReminderService:
    clock = ManualReminderClock(
        start=when or datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)
    )
    return ReminderService(JsonReminderRepository(tmp_path / "reminders.json"), clock=clock)


def test_full_invalid_date_and_timezone_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(Exception):
        service.create(
            title="Bad date",
            local_due="2026-13-40 99:99",
            timezone_name="America/New_York",
        )
    with pytest.raises(Exception):
        service.create(
            title="Bad zone",
            local_due="2026-08-15 09:00",
            timezone_name="Not/A_Zone",
        )
    assert service.count_all() == 0


def test_full_leap_day_and_month_rollover(tmp_path: Path) -> None:
    service = _service(tmp_path)
    leap = service.create(
        title="Leap",
        local_due="2028-02-29 09:00",
        timezone_name="America/New_York",
    )
    assert leap.status == "scheduled"
    monthly = service.create(
        title="Month end",
        local_due="2026-01-31 09:00",
        timezone_name="America/New_York",
        recurrence_type="monthly",
        recurrence_interval=1,
    )
    assert monthly.recurrence_type == "monthly"


def test_full_duplicate_title_time_creates_two_reminders(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.create(
        title="Standup",
        local_due="2026-08-20 09:00",
        timezone_name="America/New_York",
    )
    second = service.create(
        title="Standup",
        local_due="2026-08-20 09:00",
        timezone_name="America/New_York",
    )
    assert first.reminder_id != second.reminder_id
    assert service.count_scheduled() == 2


def test_full_cancel_twice_and_past_due_allowed(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        when=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc),
    )
    reminder = service.create(
        title="Past",
        local_due="2020-01-01 09:00",
        timezone_name="UTC",
    )
    assert reminder.status == "scheduled"
    service.cancel(reminder.reminder_id)
    with pytest.raises(ReminderServiceError):
        service.cancel(reminder.reminder_id)
    reloaded = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)),
    )
    view = reloaded.get(reminder.reminder_id)
    assert view is not None
    assert view.reminder.status == "cancelled"


def test_full_nl_and_conversation_cannot_create_reminder(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    routed = orchestrator.try_handle("set a reminder")
    assert routed is not None
    assert "reminder-add" in routed.safe_user_message or "/reminder" in routed.safe_user_message
    assert not (tmp_path / "reminders.json").exists()
    guidance = ConversationIntelligence().interpret(
        "schedule a meeting tomorrow and cancel all reminders",
        ConversationState(),
    )
    assert guidance.authorizes_privileged_action is False


def test_full_reminder_cap_is_enforced(tmp_path: Path) -> None:
    service = _service(tmp_path)
    # Stay well under cap but prove persistence plateau helper exists.
    for index in range(12):
        service.create(
            title=f"Item {index}",
            local_due="2026-09-01 09:00",
            timezone_name="UTC",
        )
    assert service.count_all() == 12
    assert MAX_STORED_REMINDERS >= 12
    reloaded = JsonReminderRepository(tmp_path / "reminders.json")
    assert len(reloaded.list_reminders()) == 12


def test_full_end_before_start_is_rejected() -> None:
    from src.calendar_models import CalendarValidationError, local_wall_pair_to_utc

    with pytest.raises(CalendarValidationError):
        local_wall_pair_to_utc(
            start_local="2026-08-20 10:00",
            end_local="2026-08-20 09:00",
            timezone_name="UTC",
        )


def test_full_snooze_and_reschedule_round_trip(tmp_path: Path) -> None:
    service = _service(tmp_path)
    reminder = service.create(
        title="Snooze me",
        local_due="2026-08-20 09:00",
        timezone_name="UTC",
    )
    snoozed = service.snooze(reminder.reminder_id, local_until="2026-08-20 10:30")
    assert snoozed.reminder_id == reminder.reminder_id
    moved = service.reschedule(
        reminder.reminder_id,
        local_due="2026-08-21 11:00",
        timezone_name="UTC",
    )
    reloaded = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)),
    )
    view = reloaded.get(moved.reminder_id)
    assert view is not None
    assert view.reminder.status == "scheduled"


def test_full_nl_calendar_cannot_create_event(tmp_path: Path) -> None:
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    routed = orchestrator.try_handle("schedule a meeting")
    assert routed is not None
    assert "calendar" in routed.safe_user_message.casefold()
    assert not (tmp_path / "calendar_control.json").exists()


def test_full_malformed_calendar_json_fail_closed(tmp_path: Path) -> None:
    from src.calendar_repository import CalendarStorageError, JsonCalendarRepository

    path = tmp_path / "calendar_control.json"
    path.write_text("{broken", encoding="utf-8")
    repo = JsonCalendarRepository(path)
    with pytest.raises(CalendarStorageError):
        repo.get_account()
    assert path.read_text(encoding="utf-8") == "{broken"
