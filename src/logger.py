"""Logging configuration for Project Cortana."""

import logging

from src.config import LOG_DIR

LOG_FILE = LOG_DIR / "cortana.log"


def setup_logging() -> logging.Logger:
    """Configure application logging and return the Cortana logger."""
    LOG_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
        force=True,
    )

    return logging.getLogger("ProjectCortana")
