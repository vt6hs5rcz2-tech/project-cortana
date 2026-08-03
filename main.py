from src.config import APP_NAME, VERSION
from src.logger import setup_logging
from src.openai_client import create_openai_client
from src.settings import load_settings


def main() -> None:
    logger = setup_logging()
    logger.info("Starting %s v%s", APP_NAME, VERSION)

    try:
        settings = load_settings()
        create_openai_client(settings)
    except ValueError as error:
        logger.error("%s", error)
        print(
            "Cortana is not connected to OpenAI yet. "
            "Add an API key to the private .env file."
        )
        return

    print("Cortana's AI connection is configured.")
    logger.info("Initialization complete.")


if __name__ == "__main__":
    main()

