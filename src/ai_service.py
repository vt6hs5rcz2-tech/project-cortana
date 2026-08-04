"""AI response service for Project Cortana."""

from collections.abc import Sequence
from typing import Protocol

from src.conversation import (
    ApiInputMessage,
    ConversationApiInput,
    ConversationHistory,
    build_conversation_input,
)
from src.document_context import build_document_context_api_messages
from src.document_retrieval import RetrievalResult
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
    document_results: Sequence[RetrievalResult] | None = None,
    document_boundary_token: str | None = None,
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
        document_results=document_results,
        document_boundary_token=document_boundary_token,
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
    document_results: Sequence[RetrievalResult] | None,
    document_boundary_token: str | None,
) -> ConversationApiInput:
    """Build API input with intentional context ordering.

    Order when context is present:
    1. active-memory developer context
    2. retrieved-document developer context
    3. conversation history
    4. current user question

    Identity instructions remain in the separate ``instructions`` field.
    """
    prefix_messages: list[ApiInputMessage] = []

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
        prefix_messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in memory_messages
        )

    documents = tuple(document_results or ())
    if documents:
        if document_boundary_token is None:
            raise ValueError(
                "A document boundary token is required when document "
                "results are provided."
            )
        document_messages = build_document_context_api_messages(
            documents,
            boundary_token=document_boundary_token,
        )
        prefix_messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in document_messages
        )

    conversation_input: ConversationApiInput
    if conversation_history is not None:
        conversation_input = build_conversation_input(
            conversation_history,
            cleaned_message,
        )
    else:
        conversation_input = cleaned_message

    if not prefix_messages:
        return conversation_input

    conversation_messages: list[ApiInputMessage]
    if isinstance(conversation_input, str):
        conversation_messages = [
            {"role": "user", "content": conversation_input},
        ]
    else:
        conversation_messages = list(conversation_input)

    combined: list[ApiInputMessage] = list(prefix_messages)
    combined.extend(conversation_messages)
    return combined
