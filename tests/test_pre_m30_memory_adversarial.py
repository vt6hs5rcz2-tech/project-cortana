"""Pre-M30 hardening tests: persistent and active memory contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.active_memory import ActiveMemoryContext
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.commands import (
    FORGET_ALL_PROMPT,
    CommandOutcome,
    handle_slash_command,
)
from src.incident_repository import JsonIncidentRepository
from src.conversation import ConversationHistory
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory import MemoryTextTooLongError
from src.config import MAX_STORED_MEMORIES
from src.memory_store import JsonMemoryStore, MemoryCountLimitError, MemoryStorageError
from src.settings import Settings


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_model="test-model")


def _slash(message: str, tmp_path: Path, store: JsonMemoryStore | None = None) -> object:
    memory = store or JsonMemoryStore(tmp_path / "memories.json")
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=memory,
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )


def test_full_remember_reload_and_forget(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    result = _slash("/remember The SOC phone is 555-0100", tmp_path, store)
    assert "saved" in str(getattr(result, "message", "")).casefold() or store.list_memories()
    reloaded = JsonMemoryStore(tmp_path / "memories.json")
    memories = reloaded.list_memories()
    assert len(memories) == 1
    assert "555-0100" in memories[0].text
    deleted = reloaded.delete_memory(memories[0].id)
    assert deleted is True
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories() == []


def test_full_duplicate_memory_texts_are_both_stored(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    first = store.add_memory("Duplicate fact")
    second = store.add_memory("Duplicate fact")
    assert first.id != second.id
    assert len(store.list_memories()) == 2


def test_full_empty_and_too_long_memory_rejected(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    with pytest.raises(Exception):
        store.add_memory("   ")
    with pytest.raises(MemoryTextTooLongError):
        store.add_memory("x" * 2001)
    assert store.list_memories() == []


def test_full_malformed_and_empty_memory_json_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    path.write_text("{not json", encoding="utf-8")
    store = JsonMemoryStore(path)
    with pytest.raises(MemoryStorageError):
        store.list_memories()
    assert path.read_text(encoding="utf-8") == "{not json"

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    empty_store = JsonMemoryStore(empty)
    with pytest.raises(MemoryStorageError):
        empty_store.list_memories()
    assert empty.read_text(encoding="utf-8") == ""


def test_full_missing_memory_file_starts_empty(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "missing.json")
    assert store.list_memories() == []
    store.add_memory("first")
    assert JsonMemoryStore(tmp_path / "missing.json").list_memories()[0].text == "first"


def test_full_forget_all_requires_confirm(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("keep me")
    result = _slash("/forget-all", tmp_path, store)
    assert getattr(result, "message", "") == FORGET_ALL_PROMPT
    assert len(store.list_memories()) == 1
    confirmed = _slash("/forget-all confirm", tmp_path, store)
    assert getattr(confirmed, "outcome") == CommandOutcome.CONTINUE
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories() == []


def test_full_conversational_forget_that_does_not_delete_persistent(
    tmp_path: Path,
) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("Persistent SOC contact")
    intel = ConversationIntelligence()
    state = ConversationState()
    intel.interpret("Explain firewall rules", state)
    guidance = intel.interpret("forget that and erase memory", state)
    assert guidance.authorizes_privileged_action is False
    assert len(store.list_memories()) == 1


def test_full_nl_remember_writes_and_list_reads(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    written = orchestrator.try_handle("remember: badge color is blue")
    assert written is not None
    listed = orchestrator.try_handle("list my memories")
    assert listed is not None
    assert "badge color is blue" in listed.safe_user_message
    assert JsonMemoryStore(tmp_path / "memories.json").list_memories()


def test_full_recall_is_session_only(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    record = store.add_memory("VPN gateway is 10.0.0.1")
    active = ActiveMemoryContext()
    handle_slash_command(
        f"/recall {record.id}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=store,
        active_memory_context=active,
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )
    assert active.list_active()
    other_session = ActiveMemoryContext()
    assert other_session.list_active() == []


def test_full_memory_count_is_bounded_by_config(tmp_path: Path) -> None:
    """Persistent memory count is fail-closed at MAX_STORED_MEMORIES (default 500)."""
    assert MAX_STORED_MEMORIES == 500
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=3)
    for index in range(3):
        store.add_memory(f"memory item {index}")
    with pytest.raises(MemoryCountLimitError) as error:
        store.add_memory("overflow")
    assert "capacity" in error.value.user_message.casefold()
    assert [memory.text for memory in store.list_memories()] == [
        "memory item 0",
        "memory item 1",
        "memory item 2",
    ]
    reloaded = JsonMemoryStore(tmp_path / "memories.json", max_memories=3)
    assert len(reloaded.list_memories()) == 3


def test_full_memory_store_declares_a_count_cap() -> None:
    import src.config as config

    assert hasattr(config, "MAX_STORED_MEMORIES")
    assert int(getattr(config, "MAX_STORED_MEMORIES")) >= 1


def test_full_failed_memory_write_does_not_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path)
    store.add_memory("first")
    original = path.read_text(encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("src.memory_store.os.replace", boom)
    with pytest.raises(MemoryStorageError):
        store.add_memory("second")
    assert path.read_text(encoding="utf-8") == original
    assert len(JsonMemoryStore(path).list_memories()) == 1
