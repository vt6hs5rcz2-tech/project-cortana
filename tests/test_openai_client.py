"""Tests for Project Cortana OpenAI client creation."""

import src.openai_client
from src.openai_client import create_openai_client
from src.settings import Settings


def test_create_openai_client_uses_settings_api_key(monkeypatch) -> None:
    """The OpenAI client should receive the validated API key."""
    captured_api_key = None
    fake_client = object()

    def fake_openai(*, api_key: str):
        nonlocal captured_api_key
        captured_api_key = api_key
        return fake_client

    monkeypatch.setattr(src.openai_client, "OpenAI", fake_openai)

    settings = Settings(
        openai_api_key="test-api-key",
        openai_model="gpt-5",
    )

    client = create_openai_client(settings)

    assert client is fake_client
    assert captured_api_key == "test-api-key"
    