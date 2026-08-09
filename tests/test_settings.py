"""Tests for Project Cortana environment settings."""

from pathlib import Path

import pytest

import src.settings
from src.settings import Settings, load_settings


def test_load_settings_calls_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment-file loading should occur only when settings are loaded."""
    dotenv_called = False

    def fake_load_dotenv() -> None:
        nonlocal dotenv_called
        dotenv_called = True

    monkeypatch.setattr(src.settings, "load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    load_settings()

    assert dotenv_called is True


def test_load_settings_uses_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings should use values supplied through environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    settings = load_settings()

    assert settings.openai_api_key == "test-api-key"
    assert settings.openai_model == "test-model"


def test_load_settings_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should remove accidental surrounding whitespace."""
    monkeypatch.setenv("OPENAI_API_KEY", "  test-api-key  ")
    monkeypatch.setenv("OPENAI_MODEL", "  test-model  ")

    settings = load_settings()

    assert settings.openai_api_key == "test-api-key"
    assert settings.openai_model == "test-model"


def test_load_settings_uses_default_model_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank model value should fall back to the default model."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    settings = load_settings()

    assert settings.openai_model == "gpt-5"


def test_load_settings_rejects_blank_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank API key should produce a clear configuration error."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5")

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY is missing",
    ):
        load_settings()


def test_settings_repr_hides_api_key() -> None:
    """The API key must not appear when settings are displayed."""
    settings = Settings(
        openai_api_key="super-secret-api-key",
        openai_model="test-model",
    )

    displayed_settings = repr(settings)

    assert "super-secret-api-key" not in displayed_settings
    assert "test-model" in displayed_settings


def test_load_settings_optional_google_oauth_client_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Optional OAuth client path should load when configured."""
    client = tmp_path / "client.json"
    client.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("CORTANA_GOOGLE_OAUTH_CLIENT_FILE", str(client))

    settings = load_settings()

    assert settings.google_oauth_client_file == client


def test_load_settings_omits_blank_google_oauth_client_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank OAuth client path should remain unset."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("CORTANA_GOOGLE_OAUTH_CLIENT_FILE", "   ")

    settings = load_settings()

    assert settings.google_oauth_client_file is None
