"""Continuous conversation loop for Project Cortana."""

import logging
from collections.abc import Callable

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient, generate_response
from src.commands import CommandOutcome, handle_slash_command, parse_slash_input
from src.conversation import (
    STARTUP_GREETING,
    SHUTDOWN_MESSAGE,
    ConversationHistory,
    is_exit_command,
)
from src.document_extractor import TextExtractor
from src.document_vault import DocumentVault
from src.memory_store import MemoryStore
from src.settings import Settings

BLANK_INPUT_MESSAGE = "Cortana: Please enter a message."


def end_conversation(*, logger: logging.Logger) -> None:
    """Print the shutdown message and record session termination."""
    print(SHUTDOWN_MESSAGE)
    logger.info("Conversation session ended by user.")


def read_session_input(input_reader: Callable[[], str]) -> str | None:
    """Read one line of user input, or None when the session should end."""
    try:
        return input_reader().strip()
    except (EOFError, KeyboardInterrupt):
        return None


def handle_message(
    *,
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
    logger: logging.Logger,
    conversation_history: ConversationHistory | None = None,
    active_memory_context: ActiveMemoryContext | None = None,
) -> None:
    """Generate and display one Cortana response."""
    active_memories = (
        active_memory_context.list_active()
        if active_memory_context is not None
        else None
    )
    memory_boundary_token = (
        active_memory_context.boundary_token
        if active_memory_context is not None
        else None
    )

    try:
        answer = generate_response(
            client=client,
            settings=settings,
            user_message=user_message,
            conversation_history=conversation_history,
            active_memories=active_memories,
            memory_boundary_token=memory_boundary_token,
        )
    except Exception as error:
        logger.error(
            "The OpenAI request failed with error type: %s",
            type(error).__name__,
        )
        print("Cortana: I could not complete that request.")
        return

    if conversation_history is not None:
        conversation_history.add_user_message(user_message.strip())
        conversation_history.add_assistant_message(answer)

    print(f"Cortana: {answer}")
    logger.info("Response completed.")


def run_conversation_loop(
    *,
    client: OpenAIClient,
    settings: Settings,
    logger: logging.Logger,
    memory_store: MemoryStore,
    active_memory_context: ActiveMemoryContext,
    document_vault: DocumentVault,
    document_extractor: TextExtractor,
    read_input: Callable[[], str] | None = None,
    conversation_history: ConversationHistory | None = None,
) -> None:
    """Run the interactive conversation until the user exits."""
    input_reader = read_input or (lambda: input("You: "))
    history = conversation_history or ConversationHistory()

    print(STARTUP_GREETING)

    while True:
        user_message = read_session_input(input_reader)

        if user_message is None:
            end_conversation(logger=logger)
            return

        if not user_message:
            print(BLANK_INPUT_MESSAGE)
            continue

        if is_exit_command(user_message):
            end_conversation(logger=logger)
            return

        if parse_slash_input(user_message) is not None:
            command_result = handle_slash_command(
                user_message,
                settings=settings,
                conversation_history=history,
                memory_store=memory_store,
                active_memory_context=active_memory_context,
                document_vault=document_vault,
                document_extractor=document_extractor,
            )
            if command_result.message:
                print(command_result.message)
            if command_result.outcome == CommandOutcome.EXIT:
                end_conversation(logger=logger)
                return
            continue

        handle_message(
            client=client,
            settings=settings,
            user_message=user_message,
            logger=logger,
            conversation_history=history,
            active_memory_context=active_memory_context,
        )
