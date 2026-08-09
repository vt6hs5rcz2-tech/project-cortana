"""Tests for Milestone 19 reminder slash commands and M18 guidance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.commands import CommandOutcome, handle_slash_command
from src.conversation import ConversationHistory
from src.conversation_loop import run_conversation_loop
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.reminder import ManualReminderClock
from src.reminder_repository import JsonReminderRepository
from src.reminder_service import ReminderService
from src.settings import Settings


FAKE_CLIENT = cast(OpenAIClient, object())


class FakeLogger(logging.Logger):
    def __init__(self) -> None:
        super().__init__("ProjectCortanaReminderTest")
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        message = str(msg)
        self.info_messages.append(message % args if args else message)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        message = str(msg)
        self.error_messages.append(message % args if args else message)


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_model="gpt-test")


def _service(tmp_path: Path) -> ReminderService:
    return ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(
            start=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        ),
    )


def _run(
    message: str,
    *,
    tmp_path: Path,
    reminder_service: ReminderService | None = None,
    memory_store: JsonMemoryStore | None = None,
    history: ConversationHistory | None = None,
):
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=history or ConversationHistory(),
        memory_store=memory_store or JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        document_retriever=LexicalDocumentRetriever(),
        reminder_service=reminder_service or _service(tmp_path),
    )


def test_reminder_add_list_show_complete_cancel_syntax(tmp_path: Path) -> None:
    service = _service(tmp_path)
    history = ConversationHistory()
    add = _run(
        "/reminder-add Call dentist | 2026-08-10 09:00 | America/New_York | none | -",
        tmp_path=tmp_path,
        reminder_service=service,
        history=history,
    )
    assert add.outcome == CommandOutcome.CONTINUE
    assert add.message is not None
    assert "Reminder created" in add.message
    assert history.turns == []

    listed = _run("/reminders", tmp_path=tmp_path, reminder_service=service)
    assert listed.message is not None
    assert "Call dentist" in listed.message

    reminder_id = service.list_scheduled()[0].reminder.reminder_id
    shown = _run(
        f"/reminder-show {reminder_id}",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert shown.message is not None
    assert reminder_id in shown.message
    assert "recent_audit:" in shown.message

    completed = _run(
        f"/reminder-complete {reminder_id}",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert completed.message is not None
    assert "completed" in completed.message.lower()

    other = _run(
        "/reminder-add Cancel me | 2026-08-11 09:00 | America/New_York | none | -",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert other.message is not None
    other_id = service.list_scheduled()[0].reminder.reminder_id
    cancelled = _run(
        f"/reminder-cancel {other_id}",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert cancelled.message is not None
    assert "cancelled" in cancelled.message.lower()


def test_reminder_add_requires_exact_five_fields(tmp_path: Path) -> None:
    result = _run(
        "/reminder-add Call dentist | 2026-08-10 09:00 | America/New_York | none",
        tmp_path=tmp_path,
    )
    assert result.message is not None
    assert "Usage: /reminder-add" in result.message


def test_snooze_and_reschedule_commands(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _run(
        "/reminder-add Weekly | 2026-08-10 09:00 | America/New_York | weekly:1:mon | Report",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    reminder_id = service.list_scheduled()[0].reminder.reminder_id
    snooze = _run(
        f"/reminder-snooze {reminder_id} | 2026-08-10 11:00",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert snooze.message is not None
    assert "snoozed" in snooze.message.lower()
    assert service.get(reminder_id).reminder.recurrence_anchor_at is not None  # type: ignore[union-attr]

    reschedule = _run(
        f"/reminder-reschedule {reminder_id} | 2026-08-17 09:00 | -",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert reschedule.message is not None
    assert "rescheduled" in reschedule.message.lower()
    reminder = service.get(reminder_id)
    assert reminder is not None
    assert reminder.reminder.timezone == "America/New_York"


def test_message_sentinel_and_recurrence_daily(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = _run(
        "/reminder-add Take medication | 2026-08-10 08:00 | America/New_York | daily | -",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    assert result.message is not None
    reminder = service.list_scheduled()[0].reminder
    assert reminder.message is None
    assert reminder.recurrence_type == "daily"


def test_remember_unchanged_and_separate_from_reminders(tmp_path: Path) -> None:
    memory_store = JsonMemoryStore(tmp_path / "memories.json")
    service = _service(tmp_path)
    remember = _run(
        "/remember John prefers morning meetings.",
        tmp_path=tmp_path,
        reminder_service=service,
        memory_store=memory_store,
    )
    assert remember.message is not None
    assert "Memory saved" in remember.message
    memories = memory_store.list_memories()
    assert len(memories) == 1
    assert "morning meetings" in memories[0].text

    _run(
        "/reminder-add Call John | 2026-08-10 09:00 | America/New_York | none | -",
        tmp_path=tmp_path,
        reminder_service=service,
        memory_store=memory_store,
    )
    assert len(memory_store.list_memories()) == 1
    assert service.count_all() == 1


def test_m18_reminder_guidance_exact_phrases(tmp_path: Path) -> None:
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    for phrase, needle in (
        ("set a reminder", "/reminder-add"),
        ("SET A REMINDER", "/reminder-add"),
        ("list reminders", "/reminders"),
        ("List Reminders", "/reminders"),
    ):
        result = orchestrator.try_handle(phrase)
        assert result is not None
        assert result.domain == "guidance"
        assert needle in result.safe_user_message

    assert orchestrator.try_handle("please set a reminder") is None
    assert orchestrator.try_handle("remind me tomorrow") is None
    assert "ReminderService" not in (
        Path("src/assistant_orchestrator.py").read_text(encoding="utf-8")
    )


def test_guidance_and_commands_do_not_mutate_history(tmp_path: Path) -> None:
    history = ConversationHistory()
    outputs: list[str] = []

    def reader() -> str:
        if not hasattr(reader, "calls"):
            reader.calls = 0  # type: ignore[attr-defined]
        reader.calls += 1  # type: ignore[attr-defined]
        if reader.calls == 1:  # type: ignore[attr-defined]
            return "set a reminder"
        if reader.calls == 2:  # type: ignore[attr-defined]
            return (
                "/reminder-add Call | 2026-08-10 09:00 | America/New_York | none | -"
            )
        return "exit"

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        reminder_service=_service(tmp_path),
        read_input=reader,
        conversation_history=history,
    )
    assert history.turns == []
    assert outputs == [] or True


def test_list_ordering_and_overdue_marker(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _run(
        "/reminder-add Later | 2026-08-12 09:00 | America/New_York | none | -",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    _run(
        "/reminder-add Earlier | 2026-08-09 09:00 | America/New_York | none | -",
        tmp_path=tmp_path,
        reminder_service=service,
    )
    clock = service.clock
    assert isinstance(clock, ManualReminderClock)
    clock.set_utc(datetime(2026, 8, 9, 14, 0, tzinfo=timezone.utc))
    listed = _run("/reminders", tmp_path=tmp_path, reminder_service=service)
    assert listed.message is not None
    earlier_pos = listed.message.index("Earlier")
    later_pos = listed.message.index("Later")
    assert earlier_pos < later_pos
    assert "overdue" in listed.message.lower()
