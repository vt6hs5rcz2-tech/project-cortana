"""AI response service for Project Cortana."""

from typing import Protocol

from src.conversation import (
    ConversationApiInput,
    ConversationHistory,
    build_conversation_input,
)
from src.settings import Settings


class AIResponse(Protocol):
    """Minimum AI response interface required by Cortana."""

    output_text: str


class ResponsesClient(Protocol):
    """Minimum Responses API interface required by Cortana."""

    def create(self, *, model: str, input: ConversationApiInput) -> AIResponse:
        """Create an AI response."""


class OpenAIClient(Protocol):
    """Minimum OpenAI client interface required by Cortana."""

    responses: ResponsesClient


def generate_response(
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
    conversation_history: ConversationHistory | None = None,
) -> str:
    """Generate a text response for a validated user message."""
    cleaned_message = user_message.strip()

    if not cleaned_message:
        raise ValueError("User message cannot be blank.")

    ai_input: ConversationApiInput
    if conversation_history is not None:
        ai_input = build_conversation_input(
            conversation_history,
            cleaned_message,
        )
    else:
        ai_input = cleaned_message

    response = client.responses.create(
        model=settings.openai_model,
        input=ai_input,
    )

    return response.output_text
