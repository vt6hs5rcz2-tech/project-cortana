"""Pre-M30 hardening tests: cross-system interaction contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.active_memory import ActiveMemoryContext
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.commands import handle_slash_command
from src.conversation import ConversationHistory
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_loop import process_conversation_turn
from src.conversation_state import ConversationState
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.reminder import ManualReminderClock
from src.reminder_repository import JsonReminderRepository
from src.reminder_service import ReminderService
from src.settings import Settings
from src.speech_delivery import SpeechDeliveryState
from tests.test_study_service import _add, _service


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_model="test-model")


def _slash(message: str, tmp_path: Path, **kwargs: Any) -> Any:
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=kwargs.get("history") or ConversationHistory(),
        memory_store=kwargs.get("memory") or JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=kwargs.get("active") or ActiveMemoryContext(),
        document_vault=kwargs.get("vault") or JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        conversation_state=kwargs.get("state"),
        speech_delivery_state=kwargs.get("delivery"),
        reminder_service=kwargs.get("reminders"),
        study_service=kwargs.get("study"),
    )


def test_full_remember_then_voice_turn_uses_active_memory(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    active = ActiveMemoryContext()
    remembered = _slash("/remember Badge color is blue", tmp_path, memory=store)
    assert store.list_memories()
    memory_id = store.list_memories()[0].id
    _slash(f"/recall {memory_id}", tmp_path, memory=store, active=active)
    history = ConversationHistory()
    client = MagicMock()
    client.responses.create.return_value = MagicMock(output_text="The badge is blue.")
    answer = process_conversation_turn(
        client=cast(Any, client),
        settings=_settings(),
        user_message="What color is the badge?",
        logger=MagicMock(),
        conversation_history=history,
        active_memory_context=active,
        conversation_state=ConversationState(),
        interaction_mode="voice",
    )
    assert answer is not None
    assert active.list_active()
    create_kwargs = client.responses.create.call_args.kwargs
    payload = str(create_kwargs)
    assert "blue" in payload.casefold() or "badge" in payload.casefold()


def test_full_study_survives_clear_conversation(tmp_path: Path) -> None:
    study, vault, _fake, repo = _service(tmp_path)
    doc_id = _add(vault, "guide.md", "Isolate the host.")
    session = study.start_session(doc_id)
    history = ConversationHistory()
    history.add_user_message("hello")
    history.add_assistant_message("hi")
    state = ConversationState()
    state.set_active_goal("chat goal")
    delivery = SpeechDeliveryState()
    delivery.record_completed_chunk("hi")
    _slash(
        "/clear",
        tmp_path,
        history=history,
        state=state,
        delivery=delivery,
        vault=vault,
        study=study,
    )
    assert history.turns == []
    assert state.is_empty
    assert repo.get_session(session.session_id) is not None


def test_full_forget_that_vs_forget_command(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("Keep this fact")
    intel = ConversationIntelligence()
    state = ConversationState()
    intel.interpret("Explain firewalls", state)
    intel.interpret("forget that", state)
    assert len(store.list_memories()) == 1
    memory_id = store.list_memories()[0].id
    _slash(f"/forget {memory_id}", tmp_path, memory=store)
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories() == []


def test_full_yes_after_old_question_cannot_create_reminder(tmp_path: Path) -> None:
    reminders = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)),
    )
    intel = ConversationIntelligence()
    state = ConversationState()
    intel.interpret("Give me options", state)
    intel.observe_assistant_reply(
        "Should I create a reminder?",
        state,
        intel.interpret("Give me options", ConversationState()),
    )
    state.set_unresolved_question(None)
    later = intel.interpret("yes", state)
    assert later.authorizes_privileged_action is False
    assert reminders.count_all() == 0
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    assert orchestrator.try_handle("yes") is None


def test_full_restart_retrieves_persisted_memory_not_conversation(
    tmp_path: Path,
) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("Persisted gateway 10.1.1.1")
    history = ConversationHistory()
    history.add_user_message("session only")
    history.add_assistant_message("ephemeral")
    restarted_store = JsonMemoryStore(tmp_path / "memories.json")
    restarted_history = ConversationHistory()
    assert restarted_store.list_memories()[0].text.endswith("10.1.1.1")
    assert restarted_history.turns == []


def test_full_contradictory_memory_and_document_stay_separated(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    vault = JsonDocumentVault(tmp_path / "documents.json")
    store.add_memory("The port is 22.")
    from src.document_ingestion import ingest_local_document

    source = tmp_path / "note.md"
    source.write_text("The port is 443.", encoding="utf-8")
    ingest_local_document(str(source), vault=vault, extractor=DefaultTextExtractor())
    memories = JsonMemoryStore(tmp_path / "memories.json").list_memories()
    docs = JsonDocumentVault(tmp_path / "documents.json").list_documents()
    assert "22" in memories[0].text
    assert "443" in docs[0].extracted_text
    assert memories[0].id != docs[0].id


def test_full_reminder_nl_correction_does_not_create_duplicate(tmp_path: Path) -> None:
    reminders = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)),
    )
    created = reminders.create(
        title="Standup",
        local_due="2026-08-20 09:00",
        timezone_name="UTC",
    )
    intel = ConversationIntelligence()
    state = ConversationState()
    intel.interpret("make that 10 instead", state)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    assert orchestrator.try_handle("yes") is None
    assert reminders.count_all() == 1
    assert reminders.get(created.reminder_id) is not None


def test_full_voice_interrupt_then_reminder_command_stays_isolated(tmp_path: Path) -> None:
    state = ConversationState()
    delivery = SpeechDeliveryState()
    delivery.record_completed_chunk("partial spoken answer")
    intel = ConversationIntelligence()
    guidance = intel.interpret("/reminder-add should not run from speech", state)
    assert guidance.authorizes_privileged_action is False
    reminders = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"),
        clock=ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)),
    )
    result = _slash(
        "/reminder-add Standup | 2026-08-20 09:00 | UTC | none | notes",
        tmp_path,
        state=state,
        delivery=delivery,
        reminders=reminders,
    )
    assert "Cortana:" in result.message
    assert reminders.count_all() in {0, 1}


def test_full_partial_memory_write_failure_then_restart_is_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path)
    store.add_memory("stable")
    original = path.read_text(encoding="utf-8")
    with patch("src.memory_store.os.replace", side_effect=OSError("crash")):
        with pytest.raises(Exception):
            store.add_memory("should not commit")
    restarted = JsonMemoryStore(path)
    assert path.read_text(encoding="utf-8") == original
    assert len(restarted.list_memories()) == 1
    assert restarted.list_memories()[0].text == "stable"
