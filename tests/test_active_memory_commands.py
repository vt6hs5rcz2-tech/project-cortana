"""Tests for active-memory slash commands and related interactions."""

import logging
from pathlib import Path
from typing import Any, cast

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.commands import (
    ACTIVE_MEMORIES_EMPTY,
    CLEAR_CONFIRMATION,
    FORGET_ALL_SUCCESS,
    RECALL_ALREADY_ACTIVE_TEMPLATE,
    RECALL_CHAR_LIMIT,
    RECALL_COUNT_LIMIT,
    RECALL_MISSING_ID,
    RECALL_NOT_FOUND_TEMPLATE,
    RECALL_SUCCESS_TEMPLATE,
    RELEASE_ALL_ALREADY_EMPTY,
    RELEASE_ALL_SUCCESS,
    RELEASE_MISSING_ID,
    RELEASE_NOT_ACTIVE_TEMPLATE,
    RELEASE_SUCCESS_TEMPLATE,
    CommandOutcome,
    CommandResult,
    handle_slash_command,
)
from src.config import MAX_ACTIVE_MEMORIES, MAX_ACTIVE_MEMORY_CHARS
from src.conversation import ConversationHistory
from src.conversation_loop import run_conversation_loop
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.settings import Settings

FAKE_CLIENT = cast(OpenAIClient, object())


class FakeLogger(logging.Logger):
    """Logger substitute used during command tests."""

    def __init__(self) -> None:
        super().__init__("ProjectCortanaTest")
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(
        self,
        msg: object,
        *args: object,
        **kwargs: Any,
    ) -> None:
        """Record informational log messages."""
        message = str(msg)
        self.info_messages.append(message % args if args else message)

    def error(
        self,
        msg: object,
        *args: object,
        **kwargs: Any,
    ) -> None:
        """Record error log messages."""
        message = str(msg)
        self.error_messages.append(message % args if args else message)


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )


def _document_vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _document_extractor() -> DefaultTextExtractor:
    return DefaultTextExtractor()


def _memory_store(tmp_path: Path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "memories.json")


def _run(
    message: str,
    *,
    tmp_path: Path,
    store: JsonMemoryStore | None = None,
    active: ActiveMemoryContext | None = None,
    history: ConversationHistory | None = None,
) -> tuple[CommandResult, JsonMemoryStore, ActiveMemoryContext, ConversationHistory]:
    memory_store = store or _memory_store(tmp_path)
    active_context = active or ActiveMemoryContext()
    conversation_history = history or ConversationHistory()
    result = handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=conversation_history,
        memory_store=memory_store,
        active_memory_context=active_context,
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
    )
    return result, memory_store, active_context, conversation_history


def test_recall_activates_existing_memory(tmp_path: Path) -> None:
    """ /recall should activate a saved memory for the current session. """
    store = _memory_store(tmp_path)
    record = store.add_memory("Authorized subnet inventory")
    active = ActiveMemoryContext()

    result, _, active, history = _run(
        f"/recall {record.id}",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message == RECALL_SUCCESS_TEMPLATE.format(memory_id=record.id)
    assert active.list_active_ids() == [record.id]
    assert history.turns == []
    assert store.list_memories()[0].text == "Authorized subnet inventory"


def test_recall_missing_argument(tmp_path: Path) -> None:
    """ /recall without an ID should return a local usage message. """
    result, _, active, _ = _run("/recall", tmp_path=tmp_path)

    assert result.message == RECALL_MISSING_ID
    assert active.active_count == 0


def test_recall_nonexistent_id(tmp_path: Path) -> None:
    """ /recall should report when the saved memory does not exist. """
    result, _, active, _ = _run("/recall missing-id", tmp_path=tmp_path)

    assert result.message == RECALL_NOT_FOUND_TEMPLATE.format(memory_id="missing-id")
    assert active.active_count == 0


def test_recall_duplicate_id(tmp_path: Path) -> None:
    """ /recall should reject a memory that is already active. """
    store = _memory_store(tmp_path)
    record = store.add_memory("Already active candidate")
    active = ActiveMemoryContext()
    active.activate(record)

    result, _, active, _ = _run(
        f"/recall {record.id}",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message == RECALL_ALREADY_ACTIVE_TEMPLATE.format(memory_id=record.id)
    assert active.active_count == 1


def test_recall_count_limit(tmp_path: Path) -> None:
    """ /recall should enforce the centralized active-memory count limit. """
    store = _memory_store(tmp_path)
    active = ActiveMemoryContext()
    for index in range(MAX_ACTIVE_MEMORIES):
        record = store.add_memory(f"Memory {index}")
        active.activate(record)
    overflow = store.add_memory("Overflow memory")

    result, _, active, _ = _run(
        f"/recall {overflow.id}",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message == RECALL_COUNT_LIMIT
    assert active.active_count == MAX_ACTIVE_MEMORIES
    assert overflow.id not in active.list_active_ids()


def test_recall_character_limit(tmp_path: Path) -> None:
    """ /recall should enforce the centralized active-memory character limit. """
    store = _memory_store(tmp_path)
    active = ActiveMemoryContext()
    # Four max-length saved memories fill the 8000-character active budget.
    filled_records = [
        store.add_memory("a" * 2000),
        store.add_memory("b" * 2000),
        store.add_memory("c" * 2000),
        store.add_memory("d" * 2000),
    ]
    for record in filled_records:
        active.activate(record)
    overflow = store.add_memory("overflow")

    result, _, active, _ = _run(
        f"/recall {overflow.id}",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message == RECALL_CHAR_LIMIT
    assert active.active_count == 4
    assert active.total_character_usage == MAX_ACTIVE_MEMORY_CHARS
    assert overflow.id not in active.list_active_ids()


def test_active_memories_lists_active_records(tmp_path: Path) -> None:
    """ /active-memories should list active IDs and text. """
    store = _memory_store(tmp_path)
    first = store.add_memory("First active")
    second = store.add_memory("Second active")
    active = ActiveMemoryContext()
    active.activate(first)
    active.activate(second)

    result, _, _, _ = _run(
        "/active-memories",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message is not None
    assert "Active memories:" in result.message
    assert first.id in result.message
    assert second.id in result.message
    assert "First active" in result.message
    assert "Second active" in result.message
    assert "memories.json" not in result.message
    assert str(store.file_path) not in result.message


def test_active_memories_empty_state(tmp_path: Path) -> None:
    """ /active-memories should report a clear empty state. """
    result, _, _, _ = _run("/active-memories", tmp_path=tmp_path)

    assert result.message == ACTIVE_MEMORIES_EMPTY


def test_release_removes_active_memory_only(tmp_path: Path) -> None:
    """ /release should remove session context without deleting storage. """
    store = _memory_store(tmp_path)
    record = store.add_memory("Keep persisted")
    active = ActiveMemoryContext()
    active.activate(record)

    result, store, active, _ = _run(
        f"/release {record.id}",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message == RELEASE_SUCCESS_TEMPLATE.format(memory_id=record.id)
    assert active.active_count == 0
    assert store.list_memories()[0].id == record.id


def test_release_missing_argument(tmp_path: Path) -> None:
    """ /release without an ID should return a local usage message. """
    result, _, _, _ = _run("/release", tmp_path=tmp_path)

    assert result.message == RELEASE_MISSING_ID


def test_release_inactive_id(tmp_path: Path) -> None:
    """ /release should report when a memory is not active. """
    result, _, _, _ = _run("/release inactive-id", tmp_path=tmp_path)

    assert result.message == RELEASE_NOT_ACTIVE_TEMPLATE.format(memory_id="inactive-id")


def test_release_all_clears_active_context(tmp_path: Path) -> None:
    """ /release-all should clear active context and keep persistent memories. """
    store = _memory_store(tmp_path)
    first = store.add_memory("Persist one")
    second = store.add_memory("Persist two")
    active = ActiveMemoryContext()
    active.activate(first)
    active.activate(second)

    result, store, active, _ = _run(
        "/release-all",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message == RELEASE_ALL_SUCCESS
    assert active.active_count == 0
    assert len(store.list_memories()) == 2


def test_release_all_already_empty(tmp_path: Path) -> None:
    """ /release-all should report clearly when nothing is active. """
    result, _, _, _ = _run("/release-all", tmp_path=tmp_path)

    assert result.message == RELEASE_ALL_ALREADY_EMPTY


def test_remember_does_not_auto_activate(tmp_path: Path) -> None:
    """ /remember must save persistently without activating the memory. """
    active = ActiveMemoryContext()

    result, store, active, _ = _run(
        "/remember Saved but inactive",
        tmp_path=tmp_path,
        active=active,
    )

    assert result.message is not None
    assert "Memory saved" in result.message
    assert len(store.list_memories()) == 1
    assert active.active_count == 0


def test_clear_does_not_remove_active_memories(tmp_path: Path) -> None:
    """ /clear should clear conversation history only. """
    store = _memory_store(tmp_path)
    record = store.add_memory("Stay active")
    active = ActiveMemoryContext()
    active.activate(record)
    history = ConversationHistory()
    history.add_user_message("Hello")
    history.add_assistant_message("Hi")

    result, _, active, history = _run(
        "/clear",
        tmp_path=tmp_path,
        store=store,
        active=active,
        history=history,
    )

    assert result.message == CLEAR_CONFIRMATION
    assert history.turns == []
    assert active.list_active_ids() == [record.id]


def test_forget_removes_currently_active_memory(tmp_path: Path) -> None:
    """Forgetting a persistent memory must also deactivate it."""
    store = _memory_store(tmp_path)
    record = store.add_memory("Delete and deactivate")
    active = ActiveMemoryContext()
    active.activate(record)

    result, store, active, _ = _run(
        f"/forget {record.id}",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message is not None
    assert record.id in result.message
    assert store.list_memories() == []
    assert active.active_count == 0


def test_forget_all_confirm_clears_active_context(tmp_path: Path) -> None:
    """ /forget-all confirm should clear active selections and persistent memories. """
    store = _memory_store(tmp_path)
    first = store.add_memory("One")
    second = store.add_memory("Two")
    active = ActiveMemoryContext()
    active.activate(first)
    active.activate(second)

    result, store, active, _ = _run(
        "/forget-all confirm",
        tmp_path=tmp_path,
        store=store,
        active=active,
    )

    assert result.message == FORGET_ALL_SUCCESS
    assert store.list_memories() == []
    assert active.active_count == 0


def test_memory_commands_do_not_alter_conversation_history(tmp_path: Path) -> None:
    """Active-memory commands must not mutate temporary conversation history."""
    store = _memory_store(tmp_path)
    record = store.add_memory("History untouched")
    history = ConversationHistory()
    history.add_user_message("Prior")
    history.add_assistant_message("Reply")
    active = ActiveMemoryContext()

    for message in (
        f"/recall {record.id}",
        "/active-memories",
        f"/release {record.id}",
        "/release-all",
    ):
        result, _, active, history = _run(
            message,
            tmp_path=tmp_path,
            store=store,
            active=active,
            history=history,
        )
        assert result.outcome == CommandOutcome.CONTINUE

    assert [turn.content for turn in history.turns] == ["Prior", "Reply"]


def test_new_memory_commands_avoid_ai_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """All new active-memory commands should stay local."""
    logger = FakeLogger()
    ai_calls = 0
    store = _memory_store(tmp_path)
    record = store.add_memory("Local recall target")
    inputs = iter(
        [
            f"/recall {record.id}",
            "/active-memories",
            f"/release {record.id}",
            "/release-all",
            "exit",
        ]
    )

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal ai_calls
        ai_calls += 1

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        memory_store=store,
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert ai_calls == 0
    assert "now active" in output
    assert "Active memories:" in output or "not currently active" in output
