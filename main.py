"""Project Cortana application entry point."""

import logging
from typing import cast

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.config import (
    APP_NAME,
    VERSION,
    get_default_document_vault_file_path,
    get_default_memory_file_path,
)
from src.conversation_loop import run_conversation_loop
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.logger import setup_logging
from src.memory_store import JsonMemoryStore
from src.openai_client import create_openai_client
from src.settings import Settings, load_settings


def initialize_ai(
    logger: logging.Logger,
) -> tuple[Settings, OpenAIClient] | None:
    """Load settings and create the OpenAI client."""
    try:
        settings = load_settings()
    except ValueError as error:
        logger.error("%s", error)
        print(
            "Cortana is not connected to OpenAI yet. "
            "Add your API key to the private .env file."
        )
        return None

    client = cast(
        OpenAIClient,
        create_openai_client(settings),
    )

    return settings, client


def main() -> None:
    """Initialize Project Cortana and run the conversation loop."""
    logger = setup_logging()
    logger.info("Starting %s v%s", APP_NAME, VERSION)

    initialized = initialize_ai(logger)

    if initialized is None:
        return

    settings, client = initialized
    print("Cortana's AI connection is configured.")

    memory_store = JsonMemoryStore(get_default_memory_file_path())
    active_memory_context = ActiveMemoryContext()
    document_vault = JsonDocumentVault(get_default_document_vault_file_path())
    document_extractor = DefaultTextExtractor()

    run_conversation_loop(
        client=client,
        settings=settings,
        logger=logger,
        memory_store=memory_store,
        active_memory_context=active_memory_context,
        document_vault=document_vault,
        document_extractor=document_extractor,
    )


if __name__ == "__main__":
    main()
