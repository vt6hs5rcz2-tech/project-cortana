"""Tests for Project Cortana application startup."""

import inspect
import logging
from typing import Any, cast

import pytest

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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    assert "Cortana: Not connected to OpenAI yet" in output
    assert "Add your API key to the private .env file." in output
    assert not output.startswith("Cortana: Cortana:")
    assert "Traceback" not in output
    assert logger.error_messages == [
        "OPENAI_API_KEY is missing. Add it to the private .env file."
    ]


def test_initialize_ai_returns_settings_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid settings should produce an OpenAI client."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    fake_client = cast(OpenAIClient, object())
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


def test_startup_user_messages_use_cortana_prefix() -> None:
    """Boot and config status lines use the standard Cortana: prefix."""
    assert main_module.AI_NOT_CONNECTED_MESSAGE.startswith("Cortana:")
    assert main_module.AI_CONNECTION_CONFIGURED_MESSAGE.startswith("Cortana:")
    assert not main_module.AI_NOT_CONNECTED_MESSAGE.startswith("Cortana: Cortana:")
    assert not main_module.AI_CONNECTION_CONFIGURED_MESSAGE.startswith(
        "Cortana: Cortana:"
    )
    assert "Add your API key to the private .env file." in (
        main_module.AI_NOT_CONNECTED_MESSAGE
    )
    init_source = inspect.getsource(main_module.initialize_ai)
    main_source = inspect.getsource(main_module.main)
    assert "Cortana:" in init_source
    assert 'print("Cortana:' in main_source
