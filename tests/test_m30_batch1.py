"""Milestone 30 Batch 1: version identity, readiness, and first-impression UX."""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any, cast

import pytest

import main as main_module
from src.ai_service import OpenAIClient
from src.commands import (
    ABOUT_TEXT,
    CORE_HELP_TEXT,
    HELP_TEXT,
    format_diagnostics,
    handle_slash_command,
)
from src.config import VERSION, product_display_name
from src.conversation import STARTUP_GREETING, ConversationHistory
from src.conversation_loop import (
    AI_AUTH_FAILURE,
    AI_GENERIC_FAILURE,
    AI_NETWORK_FAILURE,
    AI_TEMPORARY_FAILURE,
    THINKING_MESSAGE,
    classify_ai_failure,
    handle_message,
)
from src.active_memory import ActiveMemoryContext
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.logger import setup_logging
from src.memory_store import JsonMemoryStore
from src.readiness import (
    ReadinessOutcome,
    evaluate_readiness,
    format_startup_banner,
)
from src.settings import (
    INVALID_REALTIME_MODEL_MESSAGE,
    INVALID_TTS_VOICE_MESSAGE,
    MISSING_API_KEY_MESSAGE,
    Settings,
    load_settings,
    user_facing_settings_error,
)


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _slash(message: str, tmp_path: Path) -> Any:
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider-secret-should-never-appear")
        self.status_code = status_code


def test_version_identity_is_pilot() -> None:
    assert VERSION == "1.0.0-pilot"
    assert product_display_name() == "Cortana 1.0.0-pilot"


def test_about_is_pilot_facing() -> None:
    assert "1.0.0-pilot" in ABOUT_TEXT
    assert "early software milestone" not in ABOUT_TEXT.casefold()
    assert "milestone 12" not in ABOUT_TEXT.casefold()
    assert "e152657" not in ABOUT_TEXT
    assert "JsonMemoryStore" not in ABOUT_TEXT
    assert "1641" not in ABOUT_TEXT
    assert "AI assistant" in ABOUT_TEXT


def test_about_command_hides_internals(tmp_path: Path) -> None:
    result = _slash("/about", tmp_path)
    assert result.message is not None
    assert "1.0.0-pilot" in result.message
    assert "early software milestone" not in result.message.casefold()
    assert "test-api-key" not in result.message


def test_help_default_is_compact(tmp_path: Path) -> None:
    result = _slash("/help", tmp_path)
    assert result.message == CORE_HELP_TEXT
    assert "/remember" in result.message
    assert "/ask-docs" in result.message
    assert "/help more" in result.message
    assert "/playbooks" not in result.message
    assert "/incident-analysis-prepare" not in result.message
    assert "TOOL_AI_CONTEXT" not in result.message


def test_help_more_shows_full_catalog(tmp_path: Path) -> None:
    result = _slash("/help more", tmp_path)
    assert result.message == HELP_TEXT
    assert "/playbooks" in result.message
    assert "/incident-analysis-prepare" in result.message
    assert "/diagnostics" in result.message


def test_status_is_compact_and_hides_reserved_flags(tmp_path: Path) -> None:
    result = _slash("/status", tmp_path)
    assert result.message is not None
    assert "1.0.0-pilot" in result.message
    assert "Model: test-model" in result.message
    assert "Memory count:" in result.message
    assert "reserved (not implemented)" not in result.message
    assert "TOOL_AI_CONTEXT_INJECTION_ENABLED" not in result.message
    assert "test-api-key" not in result.message


def test_status_verbose_keeps_operator_catalog(tmp_path: Path) -> None:
    result = _slash("/status verbose", tmp_path)
    assert result.message is not None
    assert "Tool AI-context injection: reserved (not implemented)" in result.message
    assert "test-api-key" not in result.message


def test_diagnostics_is_safe(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("private memory text must not appear")
    result = handle_slash_command(
        "/diagnostics",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=store,
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )
    assert result.message is not None
    assert "1.0.0-pilot" in result.message
    assert "Python:" in result.message
    assert "Platform:" in result.message
    assert "Readiness:" in result.message
    assert "test-api-key" not in result.message
    assert "private memory text must not appear" not in result.message
    assert "OPENAI_API_KEY=" not in result.message
    assert "sk-" not in result.message


def test_diagnostics_function_hides_secrets() -> None:
    text = format_diagnostics(
        Settings(openai_api_key="super-secret-api-key", openai_model="test-model")
    )
    assert "super-secret-api-key" not in text
    assert "OpenAI configured: yes" in text
    assert "Realtime metadata gate: PASS" in text
    assert "not measured" not in text
    assert "cortana_user_item_id" not in text
    assert "cortana_generation" not in text
    assert "Metadata echo validation" not in text
    assert "private memory text" not in text


def test_readiness_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.readiness.calendar_capability_available", lambda _s=None: True)
    monkeypatch.setattr("src.readiness.voice_platform_available", lambda: True)
    monkeypatch.setattr("src.readiness.multimodal_platform_available", lambda: True)
    monkeypatch.setattr("src.readiness.process_isolation_available", lambda: True)
    report = evaluate_readiness(settings=_settings(), data_dir=tmp_path)
    assert report.outcome is ReadinessOutcome.READY
    assert report.required_issues == ()
    assert report.store_issues == ()


def test_readiness_optional_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.readiness.calendar_capability_available", lambda _s=None: False)
    monkeypatch.setattr("src.readiness.voice_platform_available", lambda: False)
    monkeypatch.setattr("src.readiness.multimodal_platform_available", lambda: False)
    report = evaluate_readiness(settings=_settings(), data_dir=tmp_path)
    assert report.outcome is (
        ReadinessOutcome.READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE
    )
    assert "calendar" in report.optional_unavailable
    assert "voice" in report.optional_unavailable


def test_readiness_blocked_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = evaluate_readiness(data_dir=tmp_path)
    assert report.outcome is ReadinessOutcome.BLOCKED_BY_REQUIRED_CONFIGURATION
    assert "missing_api_key" in report.required_issues


def test_readiness_malformed_primary_store_is_not_ready(tmp_path: Path) -> None:
    (tmp_path / "memories.json").write_text("{partial", encoding="utf-8")
    report = evaluate_readiness(settings=_settings(), data_dir=tmp_path)
    assert report.outcome is not ReadinessOutcome.READY
    assert "memory" in report.store_issues


def test_readiness_makes_no_network_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("readiness must not use the network")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    evaluate_readiness(settings=_settings(), data_dir=tmp_path)


def test_startup_config_errors_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("m30-config")

    def missing() -> Settings:
        raise ValueError("OPENAI_API_KEY is missing. Add it to the private .env file.")

    monkeypatch.setattr(main_module, "load_settings", missing)
    assert main_module.initialize_ai(logger) is None
    assert MISSING_API_KEY_MESSAGE in capsys.readouterr().out

    def bad_voice() -> Settings:
        raise ValueError("CORTANA_TTS_VOICE is not an allowed text-to-speech voice.")

    monkeypatch.setattr(main_module, "load_settings", bad_voice)
    assert main_module.initialize_ai(logger) is None
    output = capsys.readouterr().out
    assert INVALID_TTS_VOICE_MESSAGE in output
    assert "OPENAI_API_KEY is missing" not in output

    def bad_realtime() -> Settings:
        raise ValueError("CORTANA_REALTIME_MODEL is not an allowed realtime voice model.")

    monkeypatch.setattr(main_module, "load_settings", bad_realtime)
    assert main_module.initialize_ai(logger) is None
    output = capsys.readouterr().out
    assert INVALID_REALTIME_MODEL_MESSAGE in output
    assert "OPENAI_API_KEY is missing" not in output


def test_load_settings_user_facing_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("CORTANA_TTS_VOICE", "not-a-voice")
    with pytest.raises(ValueError) as error:
        load_settings()
    assert user_facing_settings_error(error.value) == INVALID_TTS_VOICE_MESSAGE


def test_logger_does_not_print_info_to_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = setup_logging()
    logger.info("ordinary startup information should stay in the file")
    output = capsys.readouterr()
    assert "ordinary startup information should stay in the file" not in output.out
    assert "ordinary startup information should stay in the file" not in output.err


def test_chat_prints_one_thinking_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **_kwargs: "Final answer.",
    )
    handle_message(
        client=cast(OpenAIClient, object()),
        settings=_settings(),
        user_message="Hello",
        logger=logging.getLogger("m30-think"),
    )
    output = capsys.readouterr().out
    assert output.count(THINKING_MESSAGE) == 1
    assert "Cortana: Final answer." in output


def test_ai_failure_mapping() -> None:
    assert classify_ai_failure(_StatusError(401)) == AI_AUTH_FAILURE
    assert classify_ai_failure(ConnectionError("offline")) == AI_NETWORK_FAILURE
    assert classify_ai_failure(_StatusError(429)) == AI_TEMPORARY_FAILURE
    assert classify_ai_failure(RuntimeError("mystery")) == AI_GENERIC_FAILURE
    assert "provider-secret-should-never-appear" not in classify_ai_failure(
        _StatusError(401)
    )


def test_startup_banner_and_greeting(tmp_path: Path) -> None:
    ready = evaluate_readiness(settings=_settings(), data_dir=tmp_path)
    banner = format_startup_banner(ready)
    assert "1.0.0-pilot" in banner
    assert "AI connection is configured." not in banner
    assert "Realtime metadata gate" not in banner
    assert STARTUP_GREETING.startswith("Cortana:")
    assert "human" not in STARTUP_GREETING.casefold()
    assert "/help" in STARTUP_GREETING


def test_env_example_and_readme_are_structural() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in env_example
    assert "OPENAI_MODEL=" in env_example
    assert "# CORTANA_GOOGLE_OAUTH_CLIENT_FILE=" in env_example
    assert "# CORTANA_TRANSCRIPTION_MODEL=" in env_example
    assert "# CORTANA_TTS_VOICE=" in env_example
    assert "# CORTANA_REALTIME_MODEL=" in env_example
    assert "sk-proj" not in env_example
    assert "## Quickstart" in readme
    assert "pip install -r requirements.txt" in readme
    assert "python main.py" in readme
    assert "/help" in readme
    assert "/status" in readme
    assert "/about" in readme
