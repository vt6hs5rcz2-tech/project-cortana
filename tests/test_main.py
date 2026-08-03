"""Tests for Project Cortana application startup."""

import logging
from typing import Any

import main as main_module
from src.ai_service import OpenAIClient
from src.settings import Settings


class FakeLogger(logging.Logger):
    """Logger substitute used during application tests."""

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


def test_initialize_ai_handles_missing_api_key(
    monkeypatch,
    capsys,
) -> None:
    """Initialization should explain a missing API key without crashing."""
    logger = FakeLogger()

    def fake_load_settings() -> Settings:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to the private .env file."
        )

    monkeypatch.setattr(main_module, "load_settings", fake_load_settings)

    result = main_module.initialize_ai(logger)

    output = capsys.readouterr().out

    assert result is None
    assert "Cortana is not connected to OpenAI yet" in output
    assert logger.error_messages == [
        "OPENAI_API_KEY is missing. Add it to the private .env file."
    ]


def test_initialize_ai_returns_settings_and_client(
    monkeypatch,
) -> None:
    """Valid settings should produce an OpenAI client."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    fake_client = object()
    received_settings: Settings | None = None

    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda: fake_settings,
    )

    def fake_create_openai_client(settings: Settings) -> object:
        nonlocal received_settings
        received_settings = settings
        return fake_client

    monkeypatch.setattr(
        main_module,
        "create_openai_client",
        fake_create_openai_client,
    )

    result = main_module.initialize_ai(logger)

    assert result is not None
    settings, client = result

    assert settings is fake_settings
    assert client is fake_client
    assert received_settings is fake_settings


def test_request_user_message_rejects_blank_input(
    monkeypatch,
    capsys,
) -> None:
    """A blank message should be rejected before reaching the AI service."""
    monkeypatch.setattr("builtins.input", lambda prompt: "   ")

    result = main_module.request_user_message()

    output = capsys.readouterr().out

    assert result is None
    assert "Cortana: Please enter a message." in output


def test_request_user_message_returns_cleaned_input(
    monkeypatch,
) -> None:
    """A valid message should be stripped of surrounding whitespace."""
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "  Analyze this log  ",
    )

    result = main_module.request_user_message()

    assert result == "Analyze this log"


def test_handle_message_prints_ai_response(
    monkeypatch,
    capsys,
) -> None:
    """A successful AI response should be displayed to the user."""
    logger = FakeLogger()
    fake_client = object()
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
    ) -> str:
        nonlocal captured_message
        captured_message = user_message

        assert client is fake_client
        assert settings is fake_settings

        return "Analysis complete."

    monkeypatch.setattr(
        main_module,
        "generate_response",
        fake_generate_response,
    )

    main_module.handle_message(
        client=fake_client,
        settings=fake_settings,
        user_message="Analyze this log",
        logger=logger,
    )

    output = capsys.readouterr().out

    assert captured_message == "Analyze this log"
    assert "Cortana: Analysis complete." in output
    assert logger.info_messages == ["Response completed."]


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
    ) -> str:
        raise RuntimeError("Sensitive simulated error details")

    monkeypatch.setattr(
        main_module,
        "generate_response",
        fake_generate_response,
    )

    main_module.handle_message(
        client=object(),
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


def test_main_orchestrates_application(monkeypatch) -> None:
    """Main should pass initialized objects and input to the handler."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    fake_client = object()
    received_arguments: tuple[
        object,
        Settings,
        str,
        logging.Logger,
    ] | None = None

    monkeypatch.setattr(main_module, "setup_logging", lambda: logger)
    monkeypatch.setattr(
        main_module,
        "initialize_ai",
        lambda supplied_logger: (fake_settings, fake_client),
    )
    monkeypatch.setattr(
        main_module,
        "request_user_message",
        lambda: "Analyze this log",
    )

    def fake_handle_message(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        logger: logging.Logger,
    ) -> None:
        nonlocal received_arguments
        received_arguments = (
            client,
            settings,
            user_message,
            logger,
        )

    monkeypatch.setattr(
        main_module,
        "handle_message",
        fake_handle_message,
    )

    main_module.main()

    assert received_arguments == (
        fake_client,
        fake_settings,
        "Analyze this log",
        logger,
    )
    