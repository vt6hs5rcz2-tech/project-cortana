"""Tests for Project Cortana application startup."""

import main as main_module


class FakeLogger:
    """Logger substitute used during startup tests."""

    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        """Record informational log messages."""
        self.info_messages.append(message % args if args else message)

    def error(self, message: str, *args: object) -> None:
        """Record error log messages."""
        self.error_messages.append(message % args if args else message)


def test_main_handles_missing_api_key(monkeypatch, capsys) -> None:
    """Startup should explain a missing API key without crashing."""
    logger = FakeLogger()

    monkeypatch.setattr(main_module, "setup_logging", lambda: logger)

    def fake_load_settings():
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to the private .env file."
        )

    monkeypatch.setattr(main_module, "load_settings", fake_load_settings)

    main_module.main()

    output = capsys.readouterr().out

    assert "Cortana is not connected to OpenAI yet" in output
    assert logger.error_messages == [
        "OPENAI_API_KEY is missing. Add it to the private .env file."
    ]


def test_main_initializes_client_when_settings_are_valid(
    monkeypatch,
    capsys,
) -> None:
    """Startup should create the OpenAI client when settings are valid."""
    logger = FakeLogger()
    fake_settings = object()
    received_settings = None

    monkeypatch.setattr(main_module, "setup_logging", lambda: logger)
    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda: fake_settings,
    )

    def fake_create_openai_client(settings):
        nonlocal received_settings
        received_settings = settings
        return object()

    monkeypatch.setattr(
        main_module,
        "create_openai_client",
        fake_create_openai_client,
    )

    main_module.main()

    output = capsys.readouterr().out

    assert received_settings is fake_settings
    assert "Cortana's AI connection is configured." in output
    assert "Initialization complete." in logger.info_messages
    