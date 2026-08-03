"""Project Cortana application entry point."""

import logging
from typing import cast

from src.ai_service import OpenAIClient, generate_response
from src.config import APP_NAME, VERSION
from src.logger import setup_logging
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


def request_user_message() -> str | None:
    """Request and validate one user message."""
    user_message = input("You: ").strip()

    if not user_message:
        print("Cortana: Please enter a message.")
        return None

    return user_message


def handle_message(
    *,
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
    logger: logging.Logger,
) -> None:
    """Generate and display one Cortana response."""
    try:
        answer = generate_response(
            client=client,
            settings=settings,
            user_message=user_message,
        )
    except Exception as error:
        logger.error(
            "The OpenAI request failed with error type: %s",
            type(error).__name__,
        )
        print("Cortana: I could not complete that request.")
        return

    print(f"Cortana: {answer}")
    logger.info("Response completed.")


def main() -> None:
    """Initialize Project Cortana and process one user message."""
    logger = setup_logging()
    logger.info("Starting %s v%s", APP_NAME, VERSION)

    initialized = initialize_ai(logger)

    if initialized is None:
        return

    settings, client = initialized
    print("Cortana's AI connection is configured.")

    user_message = request_user_message()

    if user_message is None:
        return

    handle_message(
        client=client,
        settings=settings,
        user_message=user_message,
        logger=logger,
    )


if __name__ == "__main__":
    main()

