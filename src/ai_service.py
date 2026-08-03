"""AI response service for Project Cortana."""

from typing import Protocol

from src.settings import Settings


class AIResponse(Protocol):
    """Minimum AI response interface required by Cortana."""

    output_text: str


class ResponsesClient(Protocol):
    """Minimum Responses API interface required by Cortana."""

    def create(self, *, model: str, input: str) -> AIResponse:
        """Create an AI response."""


class OpenAIClient(Protocol):
    """Minimum OpenAI client interface required by Cortana."""

    responses: ResponsesClient


def generate_response(
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
) -> str:
    """Generate a text response for a validated user message."""
    cleaned_message = user_message.strip()

    if not cleaned_message:
        raise ValueError("User message cannot be blank.")

    response = client.responses.create(
        model=settings.openai_model,
        input=cleaned_message,
    )

    return response.output_text
