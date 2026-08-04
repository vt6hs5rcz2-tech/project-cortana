"""AI response service for Project Cortana."""

from collections.abc import Sequence
from typing import Protocol

from src.conversation import (
    ApiInputMessage,
    ConversationApiInput,
    ConversationHistory,
    build_conversation_input,
)
from src.identity import CORTANA_SYSTEM_INSTRUCTIONS
from src.memory import MemoryRecord
from src.memory_context import build_active_memory_api_messages
from src.settings import Settings


class AIResponse(Protocol):
    """Minimum AI response interface required by Cortana."""

    output_text: str


class ResponsesClient(Protocol):
    """Minimum Responses API interface required by Cortana."""

    def create(
        self,
        *,
        model: str,
        input: ConversationApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        """Create an AI response."""


class OpenAIClient(Protocol):
    """Minimum OpenAI client interface required by Cortana."""

    responses: ResponsesClient


def generate_response(
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
    conversation_history: ConversationHistory | None = None,
    active_memories: Sequence[MemoryRecord] | None = None,
    memory_boundary_token: str | None = None,
) -> str:
    """Generate a text response for a validated user message."""
    cleaned_message = user_message.strip()

    if not cleaned_message:
        raise ValueError("User message cannot be blank.")

    ai_input = _build_ai_input(
        cleaned_message=cleaned_message,
        conversation_history=conversation_history,
        active_memories=active_memories,
        memory_boundary_token=memory_boundary_token,
    )

    response = client.responses.create(
        model=settings.openai_model,
        input=ai_input,
        instructions=CORTANA_SYSTEM_INSTRUCTIONS,
    )

    return response.output_text


def _build_ai_input(
    *,
    cleaned_message: str,
    conversation_history: ConversationHistory | None,
    active_memories: Sequence[MemoryRecord] | None,
    memory_boundary_token: str | None,
) -> ConversationApiInput:
    """Build API input, injecting active memories only when explicitly provided."""
    memories = tuple(active_memories or ())
    if memories:
        if memory_boundary_token is None:
            raise ValueError(
                "A memory boundary token is required when active memories are provided."
            )
        memory_messages = build_active_memory_api_messages(
            memories,
            boundary_token=memory_boundary_token,
        )
    else:
        memory_messages = []

    conversation_input: ConversationApiInput
    if conversation_history is not None:
        conversation_input = build_conversation_input(
            conversation_history,
            cleaned_message,
        )
    else:
        conversation_input = cleaned_message

    if not memory_messages:
        return conversation_input

    conversation_messages: list[ApiInputMessage]
    if isinstance(conversation_input, str):
        conversation_messages = [
            {"role": "user", "content": conversation_input},
        ]
    else:
        conversation_messages = list(conversation_input)

    combined: list[ApiInputMessage] = [
        {"role": message["role"], "content": message["content"]}
        for message in memory_messages
    ]
    combined.extend(conversation_messages)
    return combined
