"""Tests for Project Cortana AI response service."""

from types import SimpleNamespace

import pytest

from src.ai_service import generate_response
from src.settings import Settings


class FakeResponses:
    """Fake Responses API used without network access."""

    def __init__(self) -> None:
        self.model = None
        self.input = None

    def create(self, *, model: str, input: str):
        """Record the request and return a fake response."""
        self.model = model
        self.input = input
        return SimpleNamespace(output_text="Test response")


class FakeClient:
    """Fake OpenAI client containing the fake Responses API."""

    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_generate_response_uses_model_and_cleaned_message() -> None:
    """The service should send the configured model and trimmed message."""
    client = FakeClient()
    settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )

    result = generate_response(
        client=client,
        settings=settings,
        user_message="  Analyze this log  ",
    )

    assert result == "Test response"
    assert client.responses.model == "test-model"
    assert client.responses.input == "Analyze this log"


def test_generate_response_rejects_blank_message() -> None:
    """A blank user message should raise a clear validation error."""
    client = FakeClient()
    settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="User message cannot be blank",
    ):
        generate_response(
            client=client,
            settings=settings,
            user_message="   ",
        )
        