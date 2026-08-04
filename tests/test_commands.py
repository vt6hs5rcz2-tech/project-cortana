"""Tests for Project Cortana local slash commands."""

import logging
from typing import Any, cast

import pytest

from src.ai_service import OpenAIClient
from src.commands import (
    ABOUT_TEXT,
    CLEAR_ALREADY_EMPTY,
    CLEAR_CONFIRMATION,
    COMMAND_HELP,
    COMMAND_STATUS,
    HELP_TEXT,
    CommandOutcome,
    clear_conversation_history,
    format_status,
    handle_slash_command,
    normalize_command_name,
    parse_slash_input,
)
from src.config import HISTORY_PERSISTENCE_ENABLED
from src.conversation import ConversationHistory, SHUTDOWN_MESSAGE
from src.conversation_loop import handle_message, run_conversation_loop
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


def test_parse_slash_input_recognizes_command_like_messages() -> None:
    """Command-like slash input should return a normalized command name."""
    assert parse_slash_input("/help") == COMMAND_HELP
    assert parse_slash_input("  /STATUS  ") == COMMAND_STATUS
    assert parse_slash_input("/unknown") == "unknown"
    assert parse_slash_input("/") == ""


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


def test_handle_slash_command_help_lists_commands() -> None:
    """The /help command should describe available local commands."""
    history = ConversationHistory()
    result = handle_slash_command("/help", settings=_settings(), conversation_history=history)

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message == HELP_TEXT
    assert "/status" in result.message
    assert "/clear" in result.message
    assert "/about" in result.message
    assert "/exit" in result.message


def test_handle_slash_command_about_describes_milestone() -> None:
    """The /about command should explain the current software milestone."""
    history = ConversationHistory()
    result = handle_slash_command("/about", settings=_settings(), conversation_history=history)

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message == ABOUT_TEXT
    assert "early software milestone" in result.message


def test_handle_slash_command_status_reports_session_information() -> None:
    """The /status command should report safe local session details."""
    history = ConversationHistory(max_completed_turns=5)
    history.add_user_message("Hello")
    history.add_assistant_message("Hi there.")

    result = handle_slash_command("/status", settings=_settings(), conversation_history=history)

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "Status: online" in result.message
    assert "Model: test-model" in result.message
    assert "Retained completed turns: 1" in result.message
    assert "Maximum retained turns: 5" in result.message
    persistence_label = "enabled" if HISTORY_PERSISTENCE_ENABLED else "disabled"
    assert f"History persistence: {persistence_label}" in result.message


def test_format_status_reports_centralized_persistence_capability() -> None:
    """Status output should reflect the centralized persistence capability."""
    history = ConversationHistory()
    status_text = format_status(_settings(), history)
    persistence_label = "enabled" if HISTORY_PERSISTENCE_ENABLED else "disabled"

    assert f"History persistence: {persistence_label}" in status_text
    assert HISTORY_PERSISTENCE_ENABLED is False


def test_format_status_does_not_expose_sensitive_configuration() -> None:
    """Status output must not reveal secrets or environment values."""
    history = ConversationHistory()
    status_text = format_status(_settings(), history).lower()

    assert "test-api-key" not in status_text
    assert "openai_api_key" not in status_text
    assert ".env" not in status_text
    assert "api key" not in status_text


def test_handle_slash_command_clear_removes_active_history() -> None:
    """The /clear command should remove in-memory conversation history."""
    history = ConversationHistory()
    history.add_user_message("Hello")
    history.add_assistant_message("Hi there.")

    result = handle_slash_command("/clear", settings=_settings(), conversation_history=history)

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


def test_handle_slash_command_exit_requests_shutdown() -> None:
    """The /exit command should signal clean session termination."""
    history = ConversationHistory()
    result = handle_slash_command("/exit", settings=_settings(), conversation_history=history)

    assert result.outcome == CommandOutcome.EXIT
    assert result.message is None


def test_handle_slash_command_unknown_suggests_help() -> None:
    """Unknown slash commands should suggest /help."""
    history = ConversationHistory()
    result = handle_slash_command("/unknown", settings=_settings(), conversation_history=history)

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "/unknown" in result.message
    assert "/help" in result.message


def test_handle_slash_command_matches_with_surrounding_whitespace() -> None:
    """Commands should match case-insensitively with surrounding whitespace."""
    history = ConversationHistory()
    result = handle_slash_command("  /HELP  ", settings=_settings(), conversation_history=history)

    assert result.message == HELP_TEXT


def test_run_conversation_loop_handles_commands_without_ai_call(
    monkeypatch,
    capsys,
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
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert ai_calls == 0
    assert "/status" in output
    assert SHUTDOWN_MESSAGE in output


def test_run_conversation_loop_exit_command_uses_clean_shutdown(
    capsys,
) -> None:
    """The /exit command should use the same shutdown behavior as exit text."""
    logger = FakeLogger()
    inputs = iter(["/exit"])

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_run_conversation_loop_normal_message_still_calls_ai(
    monkeypatch,
    capsys,
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
    monkeypatch,
    capsys,
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
        read_input=lambda: next(inputs),
        conversation_history=history,
    )

    output = capsys.readouterr().out

    assert handled_messages == [path_message]
    assert "Cortana: Path reviewed." in output
    assert "Unknown command" not in output
    assert history.turns[0].content == path_message


def test_handle_message_records_path_like_input_in_history(
    monkeypatch,
    capsys,
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
