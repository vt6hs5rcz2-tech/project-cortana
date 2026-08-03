"""Tests for Project Cortana conversation loop."""

import logging
from typing import Any, cast

import main as main_module
from src.ai_service import OpenAIClient
from src.conversation import ConversationHistory, SHUTDOWN_MESSAGE, STARTUP_GREETING
from src.conversation_loop import handle_message, run_conversation_loop
from src.settings import Settings

FAKE_CLIENT = cast(OpenAIClient, object())


class FakeLogger(logging.Logger):
    """Logger substitute used during conversation loop tests."""

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


def test_handle_message_prints_ai_response(
    monkeypatch,
    capsys,
) -> None:
    """A successful AI response should be displayed to the user."""
    logger = FakeLogger()
    fake_client = FAKE_CLIENT
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    captured_message: str | None = None

    def fake_generate_response(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        conversation_history: ConversationHistory | None = None,
    ) -> str:
        nonlocal captured_message
        captured_message = user_message

        assert client is FAKE_CLIENT
        assert settings is fake_settings

        return "Analysis complete."

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    handle_message(
        client=fake_client,
        settings=fake_settings,
        user_message="Analyze this log",
        logger=logger,
    )

    output = capsys.readouterr().out

    assert captured_message == "Analyze this log"
    assert "Cortana: Analysis complete." in output
    assert logger.info_messages == ["Response completed."]


def test_handle_message_updates_conversation_history(
    monkeypatch,
    capsys,
) -> None:
    """Successful responses should be stored in session history."""
    logger = FakeLogger()
    history = ConversationHistory()

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **kwargs: "Here is the answer.",
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        user_message="What is MFA?",
        logger=logger,
        conversation_history=history,
    )

    capsys.readouterr()
    turns = history.turns

    assert len(turns) == 2
    assert turns[0].content == "What is MFA?"
    assert turns[1].content == "Here is the answer."


def test_handle_message_does_not_update_history_on_error(
    monkeypatch,
    capsys,
) -> None:
    """Failed AI requests should not alter conversation history."""
    logger = FakeLogger()
    history = ConversationHistory()

    def fake_generate_response(**kwargs: object) -> str:
        raise RuntimeError("Temporary outage")

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        user_message="Hello",
        logger=logger,
        conversation_history=history,
    )

    capsys.readouterr()

    assert history.turns == []


def test_handle_message_logs_only_safe_error_type(
    monkeypatch,
    capsys,
) -> None:
    """AI failures should not expose the exception message."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )

    def fake_generate_response(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        conversation_history: ConversationHistory | None = None,
    ) -> str:
        raise RuntimeError("Sensitive simulated error details")

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=fake_settings,
        user_message="Hello",
        logger=logger,
    )

    output = capsys.readouterr().out

    assert "Cortana: I could not complete that request." in output
    assert "Sensitive simulated error details" not in output
    assert logger.error_messages == [
        "The OpenAI request failed with error type: RuntimeError"
    ]


def test_run_conversation_loop_displays_startup_greeting(
    capsys,
) -> None:
    """The loop should greet the user before accepting input."""
    logger = FakeLogger()
    inputs = iter(["exit"])

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert STARTUP_GREETING in output


def test_run_conversation_loop_exits_on_exit_command(
    monkeypatch,
    capsys,
) -> None:
    """Exit commands should end the session with a shutdown message."""
    logger = FakeLogger()
    inputs = iter(["  QUIT  "])

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_run_conversation_loop_rejects_blank_input_without_ai_call(
    monkeypatch,
    capsys,
) -> None:
    """Blank input should be rejected and the loop should continue."""
    logger = FakeLogger()
    ai_calls = 0
    inputs = iter(["", "exit"])

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal ai_calls
        ai_calls += 1

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert "Cortana: Please enter a message." in output
    assert ai_calls == 0


def test_run_conversation_loop_continues_after_ai_error(
    monkeypatch,
    capsys,
) -> None:
    """Temporary AI failures should not terminate the session."""
    logger = FakeLogger()
    inputs = iter(["Hello", "exit"])
    call_count = 0

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            print("Cortana: I could not complete that request.")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert call_count == 1
    assert "Cortana: I could not complete that request." in output
    assert SHUTDOWN_MESSAGE in output


def test_run_conversation_loop_processes_multiple_messages(
    monkeypatch,
    capsys,
) -> None:
    """The loop should handle several messages before exiting."""
    logger = FakeLogger()
    inputs = iter(["First question", "Second question", "goodbye"])
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
        print(f"Cortana: Response to {user_message}")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert handled_messages == ["First question", "Second question"]
    assert "Response to First question" in output
    assert "Response to Second question" in output
    assert SHUTDOWN_MESSAGE in output


def test_run_conversation_loop_rejects_whitespace_only_input_without_ai_call(
    monkeypatch,
    capsys,
) -> None:
    """Whitespace-only input should be rejected and the loop should continue."""
    logger = FakeLogger()
    ai_calls = 0
    inputs = iter(["   ", "exit"])

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal ai_calls
        ai_calls += 1

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert "Cortana: Please enter a message." in output
    assert ai_calls == 0


def test_run_conversation_loop_exits_cleanly_on_eof(
    capsys,
) -> None:
    """EOF during input should print the shutdown message without a traceback."""
    logger = FakeLogger()

    def raise_eof() -> str:
        raise EOFError

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=raise_eof,
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_run_conversation_loop_exits_cleanly_on_keyboard_interrupt(
    capsys,
) -> None:
    """KeyboardInterrupt during input should exit cleanly without a traceback."""
    logger = FakeLogger()

    def raise_keyboard_interrupt() -> str:
        raise KeyboardInterrupt

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        read_input=raise_keyboard_interrupt,
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_main_orchestrates_conversation_loop(monkeypatch) -> None:
    """Main should initialize Cortana and start the conversation loop."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    fake_client = cast(OpenAIClient, object())
    received_arguments: tuple[
        OpenAIClient,
        Settings,
        logging.Logger,
    ] | None = None

    monkeypatch.setattr(main_module, "setup_logging", lambda: logger)
    monkeypatch.setattr(
        main_module,
        "initialize_ai",
        lambda supplied_logger: (fake_settings, fake_client),
    )

    def fake_run_conversation_loop(
        *,
        client: OpenAIClient,
        settings: Settings,
        logger: logging.Logger,
    ) -> None:
        nonlocal received_arguments
        received_arguments = (client, settings, logger)

    monkeypatch.setattr(
        main_module,
        "run_conversation_loop",
        fake_run_conversation_loop,
    )

    main_module.main()

    assert received_arguments == (fake_client, fake_settings, logger)
