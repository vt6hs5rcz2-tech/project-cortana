"""Pre-M30 hardening tests: restart, recovery, and partial-failure isolation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.commands import handle_slash_command
from src.conversation import ConversationHistory
from src.conversation_loop import process_conversation_turn
from src.conversation_state import ConversationState
from src.document_extractor import DefaultTextExtractor
from src.document_vault import DocumentStorageError, JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore, MemoryStorageError
from src.reminder import ManualReminderClock
from src.reminder_repository import JsonReminderRepository, ReminderStorageError
from src.reminder_service import ReminderService
from src.settings import Settings
from src.speech_delivery import SpeechDeliveryState
from src.study_repository import JsonStudyRepository, StudyStorageError
from src.tool_repository import JsonToolControlRepository, ToolStorageError
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
        reminder_service=kwargs.get("reminders"),
        study_service=kwargs.get("study"),
    )


def test_persistent_stores_survive_service_rebuild(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("VPN is 10.0.0.1")
    vault = JsonDocumentVault(tmp_path / "documents.json")
    source = tmp_path / "notes.md"
    source.write_text("Allow SSH on 22.\n", encoding="utf-8")
    _slash(f"/add-document {source}", tmp_path, vault=vault, memory=memory)
    clock = ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc))
    reminders = ReminderService(
        JsonReminderRepository(tmp_path / "reminders.json"), clock=clock
    )
    _slash(
        "/reminder-add Rebuild me | 2026-08-22 09:00 | America/New_York | none | -",
        tmp_path,
        reminders=reminders,
        memory=memory,
        vault=vault,
    )
    study, study_vault, _fake, _repo = _service(tmp_path)
    doc_id = _add(study_vault, "guide.md", "Isolate the host immediately please now.")
    _slash(f"/study-start {doc_id}", tmp_path, study=study, vault=study_vault)

    memory2 = JsonMemoryStore(tmp_path / "memories.json")
    vault2 = JsonDocumentVault(tmp_path / "documents.json")
    reminders2 = ReminderService(JsonReminderRepository(tmp_path / "reminders.json"))
    study2 = JsonStudyRepository(tmp_path / "study_state.json")
    assert memory2.list_memories()
    assert vault2.list_documents()
    assert reminders2.count_all() >= 1
    assert study2.get_active_session() is not None
    assert ConversationHistory().turns == []
    assert ActiveMemoryContext().list_active() == []
    assert SpeechDeliveryState().pending_chunks == []


def test_malformed_memory_store_does_not_corrupt_documents(tmp_path: Path) -> None:
    vault = JsonDocumentVault(tmp_path / "documents.json")
    source = tmp_path / "ok.md"
    source.write_text("Keep this document.\n", encoding="utf-8")
    _slash(f"/add-document {source}", tmp_path, vault=vault)
    (tmp_path / "memories.json").write_text("{partial", encoding="utf-8")
    with pytest.raises(MemoryStorageError):
        JsonMemoryStore(tmp_path / "memories.json").list_memories()
    still = JsonDocumentVault(tmp_path / "documents.json")
    assert still.list_documents()
    assert (tmp_path / "memories.json").read_text(encoding="utf-8") == "{partial"


def test_malformed_reminder_store_does_not_corrupt_memory(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("keep me")
    (tmp_path / "reminders.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ReminderStorageError):
        JsonReminderRepository(tmp_path / "reminders.json").list_reminders()
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories()[0].text == "keep me"


def test_missing_optional_store_starts_empty_and_others_work(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("present")
    reminders = ReminderService(
        JsonReminderRepository(tmp_path / "missing-reminders.json")
    )
    assert reminders.list_scheduled() == []
    listed = _slash("/memories", tmp_path, memory=memory, reminders=reminders)
    assert "present" in (listed.message or "")
    empty = _slash("/reminders", tmp_path, memory=memory, reminders=reminders)
    assert "No scheduled reminders" in (empty.message or "")


def test_stale_temp_file_beside_memory_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path)
    store.add_memory("real")
    leftover = tmp_path / ".memories-abc.tmp"
    leftover.write_text("garbage", encoding="utf-8")
    reloaded = JsonMemoryStore(path)
    assert [item.text for item in reloaded.list_memories()] == ["real"]
    assert leftover.exists()


def test_memory_write_failure_does_not_poison_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path)
    store.add_memory("first")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("src.memory_store.os.replace", boom)
    failed = _slash("/remember second", tmp_path, memory=store)
    assert failed.message is not None
    assert failed.message.startswith("Cortana:")
    assert "Traceback" not in failed.message
    monkeypatch.undo()
    answer = process_conversation_turn(
        client=cast(
            Any,
            MagicMock(
                responses=MagicMock(
                    create=MagicMock(return_value=MagicMock(output_text="Still here."))
                )
            ),
        ),
        settings=_settings(),
        user_message="Are you still working?",
        logger=logging.getLogger("hunt3"),
        conversation_history=ConversationHistory(),
        conversation_state=ConversationState(),
    )
    assert answer == "Still here."
    assert len(JsonMemoryStore(path).list_memories()) == 1


def test_ai_failure_then_unrelated_memory_list_still_works(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("survives")

    def fail(**kwargs: Any) -> str:
        raise RuntimeError("upstream")

    monkeypatch.setattr("src.conversation_loop.generate_response", fail)
    answer = process_conversation_turn(
        client=cast(Any, object()),
        settings=_settings(),
        user_message="hello",
        logger=logging.getLogger("hunt3"),
        conversation_history=ConversationHistory(),
        conversation_state=ConversationState(),
    )
    assert answer is None
    listed = _slash("/memories", tmp_path, memory=store)
    assert "survives" in (listed.message or "")


def test_study_persistence_failure_does_not_break_memory(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("independent")
    (tmp_path / "study_state.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(StudyStorageError):
        JsonStudyRepository(tmp_path / "study_state.json").list_sessions()
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories()[0].text == "independent"


def test_unreadable_document_store_leaves_file_and_memory_intact(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("vault-independent")
    docs = tmp_path / "documents.json"
    docs.write_text("{broken", encoding="utf-8")
    with pytest.raises(DocumentStorageError):
        JsonDocumentVault(docs).list_documents()
    assert docs.read_text(encoding="utf-8") == "{broken"
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories()


def test_tool_store_malformed_does_not_wipe_incidents(tmp_path: Path) -> None:
    incidents = JsonIncidentRepository(tmp_path / "incidents.json")
    assert incidents.incident_count() == 0
    (tmp_path / "tool_control.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ToolStorageError):
        JsonToolControlRepository(tmp_path / "tool_control.json").list_scopes()
    assert JsonIncidentRepository(tmp_path / "incidents.json").incident_count() == 0
    listed = _slash("/memories", tmp_path, memory=JsonMemoryStore(tmp_path / "memories.json"))
    assert listed.message is not None
    assert listed.message.startswith("Cortana:")
