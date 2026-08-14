"""Tests for Project Cortana local slash commands."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.commands import (
    ABOUT_TEXT,
    CLEAR_ALREADY_EMPTY,
    CLEAR_CONFIRMATION,
    COMMAND_HELP,
    COMMAND_STATUS,
    FORGET_ALL_NOT_CONFIRMED,
    FORGET_ALL_PROMPT,
    FORGET_ALL_SUCCESS,
    FORGET_MISSING_ID,
    FORGET_NOT_FOUND_TEMPLATE,
    HELP_TEXT,
    MEMORIES_EMPTY,
    REMEMBER_CAPACITY,
    REMEMBER_MISSING_TEXT,
    REMEMBER_TOO_LONG,
    CommandOutcome,
    clear_conversation_history,
    format_status,
    handle_slash_command,
    normalize_command_name,
    parse_slash_input,
)
from src.command_argument_utils import extract_command_argument
from src.config import (
    ACTIVE_MEMORY_PERSISTENCE_ENABLED,
    EXPLICIT_PERSISTENT_MEMORY_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    MAX_ACTIVE_MEMORIES,
    MAX_ACTIVE_MEMORY_CHARS,
    MAX_MEMORY_TEXT_LENGTH,
    MAX_STORED_MEMORIES,
)
from src.conversation import ConversationHistory, SHUTDOWN_MESSAGE
from src.conversation_loop import handle_message, run_conversation_loop
from src.memory import BlankMemoryTextError, MemoryTextTooLongError
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore, MemoryCountLimitError
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


def _active_memory_context() -> ActiveMemoryContext:
    return ActiveMemoryContext()


def test_parse_slash_input_recognizes_command_like_messages() -> None:
    """Command-like slash input should return a normalized command name."""
    assert parse_slash_input("/help") == COMMAND_HELP
    assert parse_slash_input("  /STATUS  ") == COMMAND_STATUS
    assert parse_slash_input("/unknown") == "unknown"
    assert parse_slash_input(" /help") == COMMAND_HELP
    assert parse_slash_input("/HELP") == COMMAND_HELP
    assert parse_slash_input("/") is None
    assert parse_slash_input("/ ") is None
    assert parse_slash_input(" / ") is None
    assert parse_slash_input(" / help") is None
    assert parse_slash_input("/helper") == "helper"


def test_parse_slash_input_ignores_path_like_messages() -> None:
    """Absolute paths should not be treated as local slash commands."""
    assert parse_slash_input("/etc/passwd") is None
    assert parse_slash_input("/var/log/auth.log") is None
    assert parse_slash_input("/home/user/file") is None
    assert parse_slash_input("help") is None
    assert parse_slash_input("please /help") is None


def test_normalize_command_name_is_case_insensitive() -> None:
    """Command names should normalize without the slash and in lowercase."""
    assert normalize_command_name("/HELP") == COMMAND_HELP
    assert normalize_command_name("  /Status  ") == COMMAND_STATUS
    assert normalize_command_name("/clear") == "clear"


def test_extract_command_argument_preserves_capitalization() -> None:
    """Arguments after the command token should keep original capitalization."""
    assert extract_command_argument("/remember Keep CASE") == "Keep CASE"
    assert extract_command_argument("/forget-all confirm") == "confirm"
    assert extract_command_argument("/remember") == ""


def test_handle_slash_command_help_lists_commands(tmp_path: Path) -> None:
    """The /help command should describe available local commands."""
    history = ConversationHistory()
    result = handle_slash_command(
        "/help",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message == HELP_TEXT
    assert "/help" in result.message
    assert "/status" in result.message
    assert "/clear" in result.message
    assert "/remember" in result.message
    assert "/memories" in result.message
    assert "/forget" in result.message
    assert "/forget-all" in result.message
    assert "/recall" in result.message
    assert "/active-memories" in result.message
    assert "/release" in result.message
    assert "/release-all" in result.message
    assert "/about" in result.message
    assert "/exit" in result.message


def test_handle_slash_command_about_describes_milestone(tmp_path: Path) -> None:
    """The /about command should explain the current software milestone."""
    history = ConversationHistory()
    result = handle_slash_command(
        "/about",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message == ABOUT_TEXT
    assert "early software milestone" in result.message
    assert "persistent memory" in result.message.lower()


def test_handle_slash_command_status_reports_session_information(
    tmp_path: Path,
) -> None:
    """The /status command should report safe local session details."""
    history = ConversationHistory(max_completed_turns=5)
    history.add_user_message("Hello")
    history.add_assistant_message("Hi there.")
    store = _memory_store(tmp_path)
    store.add_memory("Status memory")

    result = handle_slash_command(
        "/status",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "Status: online" in result.message
    assert "Model: test-model" in result.message
    assert "Retained completed turns: 1" in result.message
    assert "Maximum retained turns: 5" in result.message
    persistence_label = "enabled" if HISTORY_PERSISTENCE_ENABLED else "disabled"
    memory_label = "enabled" if EXPLICIT_PERSISTENT_MEMORY_ENABLED else "disabled"
    assert f"History persistence: {persistence_label}" in result.message
    assert f"Explicit persistent memory: {memory_label}" in result.message
    assert "Saved memories: 1" in result.message
    assert f"Maximum stored memories: {MAX_STORED_MEMORIES}" in result.message
    assert "Active memories: 0" in result.message
    assert f"Maximum active memories: {MAX_ACTIVE_MEMORIES}" in result.message
    assert "Active memory characters: 0" in result.message
    assert (
        f"Maximum active memory characters: {MAX_ACTIVE_MEMORY_CHARS}"
        in result.message
    )
    active_persistence = "enabled" if ACTIVE_MEMORY_PERSISTENCE_ENABLED else "disabled"
    assert f"Active memory persistence: {active_persistence}" in result.message


def test_format_status_reports_centralized_persistence_capability(
    tmp_path: Path,
) -> None:
    """Status output should reflect the centralized persistence capability."""
    history = ConversationHistory()
    status_text = format_status(
        _settings(),
        history,
        _memory_store(tmp_path),
        _active_memory_context(),
        _document_vault(tmp_path),
    )
    persistence_label = "enabled" if HISTORY_PERSISTENCE_ENABLED else "disabled"
    memory_label = "enabled" if EXPLICIT_PERSISTENT_MEMORY_ENABLED else "disabled"
    active_persistence = "enabled" if ACTIVE_MEMORY_PERSISTENCE_ENABLED else "disabled"

    assert f"History persistence: {persistence_label}" in status_text
    assert f"Explicit persistent memory: {memory_label}" in status_text
    assert HISTORY_PERSISTENCE_ENABLED is False
    assert EXPLICIT_PERSISTENT_MEMORY_ENABLED is True
    assert ACTIVE_MEMORY_PERSISTENCE_ENABLED is False
    assert "Saved memories: 0" in status_text
    assert "Active memories: 0" in status_text
    assert f"Active memory persistence: {active_persistence}" in status_text


def test_format_status_does_not_expose_sensitive_configuration(
    tmp_path: Path,
) -> None:
    """Status output must not reveal secrets, paths, or environment values."""
    history = ConversationHistory()
    store = _memory_store(tmp_path)
    status_text = format_status(_settings(), history, store, _active_memory_context(), _document_vault(tmp_path)).lower()

    assert "test-api-key" not in status_text
    assert "openai_api_key" not in status_text
    assert ".env" not in status_text
    assert "api key" not in status_text
    assert "memories.json" not in status_text
    assert str(store.file_path).lower() not in status_text


def test_handle_slash_command_clear_removes_active_history(tmp_path: Path) -> None:
    """The /clear command should remove in-memory conversation history."""
    history = ConversationHistory()
    history.add_user_message("Hello")
    history.add_assistant_message("Hi there.")

    result = handle_slash_command(
        "/clear",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message == CLEAR_CONFIRMATION
    assert history.turns == []
    assert history.completed_turn_count == 0


def test_clear_conversation_history_when_already_empty() -> None:
    """Clearing empty history should return a helpful confirmation."""
    history = ConversationHistory()

    message = clear_conversation_history(history)

    assert message == CLEAR_ALREADY_EMPTY
    assert history.turns == []


def test_handle_slash_command_exit_requests_shutdown(tmp_path: Path) -> None:
    """The /exit command should signal clean session termination."""
    history = ConversationHistory()
    result = handle_slash_command(
        "/exit",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.outcome == CommandOutcome.EXIT
    assert result.message is None


def test_handle_slash_command_unknown_suggests_help(tmp_path: Path) -> None:
    """Unknown slash commands should suggest /help."""
    history = ConversationHistory()
    result = handle_slash_command(
        "/unknown",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "/unknown" in result.message
    assert "/help" in result.message


def test_handle_slash_command_matches_with_surrounding_whitespace(
    tmp_path: Path,
) -> None:
    """Commands should match case-insensitively with surrounding whitespace."""
    history = ConversationHistory()
    result = handle_slash_command(
        "  /HELP  ",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.message == HELP_TEXT


def test_remember_saves_memory_and_returns_id(tmp_path: Path) -> None:
    """The /remember command should save one memory and return its ID."""
    store = _memory_store(tmp_path)
    history = ConversationHistory()

    result = handle_slash_command(
        "/remember Remember this fact",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    memories = store.list_memories()
    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert len(memories) == 1
    assert memories[0].text == "Remember this fact"
    assert memories[0].id in result.message


def test_slash_remember_remains_authoritative_for_deictic_looking_text(
    tmp_path: Path,
) -> None:
    """Slash /remember still persists explicit text that NL memory would reject."""
    store = _memory_store(tmp_path)
    result = handle_slash_command(
        "/remember this forever",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )
    assert result.message is not None
    assert "Memory saved" in result.message
    assert [item.text for item in store.list_memories()] == ["this forever"]


def test_slash_remember_this_thing_remains_explicit_control_plane(
    tmp_path: Path,
) -> None:
    """Slash /remember this thing stays explicit persistence, unlike NL memory."""
    store = _memory_store(tmp_path)
    result = handle_slash_command(
        "/remember this thing",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )
    assert result.message is not None
    assert "Memory saved" in result.message
    assert [item.text for item in store.list_memories()] == ["this thing"]


def test_remember_preserves_argument_capitalization(tmp_path: Path) -> None:
    """Memory text should preserve the user's original capitalization."""
    store = _memory_store(tmp_path)

    handle_slash_command(
        "/remember Keep CamelCase Values",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert store.list_memories()[0].text == "Keep CamelCase Values"


def test_remember_accepts_slash_containing_text(tmp_path: Path) -> None:
    """Ordinary slash characters in memory text should be preserved."""
    store = _memory_store(tmp_path)

    handle_slash_command(
        "/remember Check /var/log/auth.log carefully",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert store.list_memories()[0].text == "Check /var/log/auth.log carefully"


def test_remember_rejects_missing_text(tmp_path: Path) -> None:
    """The /remember command should reject blank or missing text."""
    store = _memory_store(tmp_path)

    result = handle_slash_command(
        "/remember   ",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message == REMEMBER_MISSING_TEXT
    assert store.list_memories() == []


def test_remember_rejects_oversized_text(tmp_path: Path) -> None:
    """The /remember command should reject text above the maximum length."""
    store = _memory_store(tmp_path)
    oversized = "x" * (MAX_MEMORY_TEXT_LENGTH + 1)

    result = handle_slash_command(
        f"/remember {oversized}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message == REMEMBER_TOO_LONG
    assert store.list_memories() == []


def test_remember_rejects_when_store_is_at_capacity(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=1)
    store.add_memory("kept")

    result = handle_slash_command(
        "/remember another fact",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message == MemoryCountLimitError(max_memories=1).user_message
    assert result.message.startswith("Cortana:")
    assert REMEMBER_CAPACITY == MemoryCountLimitError().user_message
    assert [memory.text for memory in store.list_memories()] == ["kept"]


def test_help_ignores_oversized_extra_arguments(tmp_path: Path) -> None:
    result = handle_slash_command(
        "/help " + ("x" * 200_000),
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )
    assert result.message == HELP_TEXT


def test_remember_maps_validation_errors_by_type_not_message_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remember responses should follow typed validation errors, not exception wording."""
    store = _memory_store(tmp_path)

    def raise_too_long(text: str) -> object:
        raise MemoryTextTooLongError("unrelated wording that must not be inspected")

    monkeypatch.setattr(store, "add_memory", raise_too_long)
    too_long_result = handle_slash_command(
        "/remember Any text",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    def raise_blank(text: str) -> object:
        raise BlankMemoryTextError("also unrelated wording")

    monkeypatch.setattr(store, "add_memory", raise_blank)
    blank_result = handle_slash_command(
        "/remember Any text",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert too_long_result.message == REMEMBER_TOO_LONG
    assert blank_result.message == REMEMBER_MISSING_TEXT


def test_memories_lists_records(tmp_path: Path) -> None:
    """The /memories command should list saved memory details."""
    store = _memory_store(tmp_path)
    record = store.add_memory("Listed memory")

    result = handle_slash_command(
        "/memories",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message is not None
    assert record.id in result.message
    assert record.created_at in result.message
    assert "Listed memory" in result.message


def test_memories_empty_state(tmp_path: Path) -> None:
    """The /memories command should return a clear empty-state message."""
    result = handle_slash_command(
        "/memories",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.message == MEMORIES_EMPTY


def test_forget_deletes_matching_id(tmp_path: Path) -> None:
    """The /forget command should delete a matching memory ID."""
    store = _memory_store(tmp_path)
    record = store.add_memory("Delete me")

    result = handle_slash_command(
        f"/forget {record.id}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message is not None
    assert record.id in result.message
    assert store.list_memories() == []


def test_forget_missing_id(tmp_path: Path) -> None:
    """The /forget command should require a memory ID argument."""
    result = handle_slash_command(
        "/forget",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.message == FORGET_MISSING_ID


def test_forget_nonexistent_id(tmp_path: Path) -> None:
    """The /forget command should report when an ID does not exist."""
    result = handle_slash_command(
        "/forget missing-id",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )

    assert result.message == FORGET_NOT_FOUND_TEMPLATE.format(memory_id="missing-id")


def test_forget_all_requires_confirmation(tmp_path: Path) -> None:
    """The first /forget-all command should not delete anything."""
    store = _memory_store(tmp_path)
    store.add_memory("Keep me")

    result = handle_slash_command(
        "/forget-all",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message == FORGET_ALL_PROMPT
    assert len(store.list_memories()) == 1


def test_forget_all_confirm_deletes_everything(tmp_path: Path) -> None:
    """Confirmed /forget-all should delete every saved memory."""
    store = _memory_store(tmp_path)
    store.add_memory("One")
    store.add_memory("Two")

    result = handle_slash_command(
        "/forget-all confirm",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message == FORGET_ALL_SUCCESS
    assert store.list_memories() == []


def test_forget_all_failed_confirmation_leaves_memories_intact(
    tmp_path: Path,
) -> None:
    """Any non-confirm follow-up should leave memories intact."""
    store = _memory_store(tmp_path)
    store.add_memory("Still here")

    result = handle_slash_command(
        "/forget-all yes",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message == FORGET_ALL_NOT_CONFIRMED
    assert len(store.list_memories()) == 1


def test_memory_commands_do_not_alter_temporary_conversation_history(
    tmp_path: Path,
) -> None:
    """Memory commands should not modify temporary conversation history."""
    store = _memory_store(tmp_path)
    history = ConversationHistory()
    history.add_user_message("Hello")
    history.add_assistant_message("Hi")
    original_turns = history.turns

    handle_slash_command(
        "/remember Important note",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )
    handle_slash_command(
        "/memories",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )
    memory_id = store.list_memories()[0].id
    handle_slash_command(
        f"/forget {memory_id}",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert history.turns == original_turns


def test_clear_does_not_delete_persistent_memories(tmp_path: Path) -> None:
    """Clearing conversation history must not delete persistent memories."""
    store = _memory_store(tmp_path)
    store.add_memory("Persistent")
    history = ConversationHistory()
    history.add_user_message("Hello")
    history.add_assistant_message("Hi")

    handle_slash_command(
        "/clear",
        settings=_settings(),
        conversation_history=history,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert history.turns == []
    assert len(store.list_memories()) == 1


def test_storage_errors_return_safe_local_messages(tmp_path: Path) -> None:
    """Memory storage failures should return safe local messages."""
    path = tmp_path / "memories.json"
    path.write_text("{bad-json", encoding="utf-8")
    store = JsonMemoryStore(path)

    result = handle_slash_command(
        "/memories",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.message is not None
    assert "could not be loaded safely" in result.message
    assert path.read_text(encoding="utf-8") == "{bad-json"


def test_run_conversation_loop_handles_commands_without_ai_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Slash commands should be handled locally without calling the AI service."""
    logger = FakeLogger()
    ai_calls = 0
    inputs = iter(["/help", "exit"])

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
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert ai_calls == 0
    assert "/status" in output
    assert SHUTDOWN_MESSAGE in output


def test_memory_commands_avoid_ai_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """All memory commands should be handled locally without AI calls."""
    logger = FakeLogger()
    ai_calls = 0
    store = _memory_store(tmp_path)
    inputs = iter(
        [
            "/remember Local only",
            "/memories",
            "/forget-all",
            "/forget-all confirm",
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
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert ai_calls == 0
    assert "Memory saved" in output
    assert FORGET_ALL_SUCCESS in output


def test_run_conversation_loop_exit_command_uses_clean_shutdown(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The /exit command should use the same shutdown behavior as exit text."""
    logger = FakeLogger()
    inputs = iter(["/exit"])

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_run_conversation_loop_normal_message_still_calls_ai(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Non-command input should continue through the AI conversation path."""
    logger = FakeLogger()
    inputs = iter(["Analyze this log", "exit"])
    handled_messages: list[str] = []

    def fake_handle_message(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        logger: logging.Logger,
        conversation_history: ConversationHistory | None = None,
        active_memory_context: ActiveMemoryContext | None = None,
        **_kwargs: object,
    ) -> None:
        handled_messages.append(user_message)
        print("Cortana: AI response")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert handled_messages == ["Analyze this log"]
    assert "Cortana: AI response" in output


@pytest.mark.parametrize(
    "path_message",
    [
        "/etc/passwd",
        "/var/log/auth.log",
        "/home/user/file",
    ],
)
def test_run_conversation_loop_path_like_messages_call_ai(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    path_message: str,
) -> None:
    """Absolute paths should continue through the AI conversation path."""
    logger = FakeLogger()
    inputs = iter([path_message, "exit"])
    handled_messages: list[str] = []

    def fake_handle_message(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        logger: logging.Logger,
        conversation_history: ConversationHistory | None = None,
        active_memory_context: ActiveMemoryContext | None = None,
        **_kwargs: object,
    ) -> None:
        handled_messages.append(user_message)
        if conversation_history is not None:
            conversation_history.add_user_message(user_message)
            conversation_history.add_assistant_message("Path reviewed.")
        print("Cortana: Path reviewed.")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    history = ConversationHistory()

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
        conversation_history=history,
    )

    output = capsys.readouterr().out

    assert handled_messages == [path_message]
    assert "Cortana: Path reviewed." in output
    assert "Unknown command" not in output
    assert history.turns[0].content == path_message


def test_handle_message_records_path_like_input_in_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Path-like user input should be stored in conversation history normally."""
    logger = FakeLogger()
    history = ConversationHistory()

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **kwargs: "Review complete.",
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=_settings(),
        user_message="/etc/passwd",
        logger=logger,
        conversation_history=history,
    )

    capsys.readouterr()

    assert history.turns[0].content == "/etc/passwd"
    assert history.turns[1].content == "Review complete."


def test_status_storage_error_does_not_crash_session(tmp_path: Path) -> None:
    """Status should surface storage errors without raising from command handling."""
    path = tmp_path / "memories.json"
    path.write_text("", encoding="utf-8")
    store = JsonMemoryStore(path)

    result = handle_slash_command(
        "/status",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=store,
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "could not be loaded safely" in result.message


def _cortana_temp_dir_names() -> set[str]:
    root = Path(tempfile.gettempdir())
    return {path.name for path in root.glob("cortana-*") if path.is_dir()}


def test_uninjected_help_does_not_leak_ephemeral_temp_dirs(tmp_path: Path) -> None:
    before = _cortana_temp_dir_names()
    result = handle_slash_command(
        "/help",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )
    after = _cortana_temp_dir_names()
    assert result.message == HELP_TEXT
    assert after - before == set()


def test_uninjected_reminders_uses_lazy_fallback_then_cleans_up(
    tmp_path: Path,
) -> None:
    before = _cortana_temp_dir_names()
    result = handle_slash_command(
        "/reminders",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
    )
    after = _cortana_temp_dir_names()
    assert result.message is not None
    assert result.message.startswith("Cortana:")
    assert after - before == set()
