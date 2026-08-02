from src.assistant import greet
from src.config import APP_NAME, VERSION
from src.logger import setup_logging


def main() -> None:
    logger = setup_logging()

    logger.info("Starting %s v%s", APP_NAME, VERSION)

    # User-facing assistant output uses print().
    greet()

    logger.info("Initialization complete.")


if __name__ == "__main__":
    main()
    