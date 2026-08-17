"""Logging configuration for Project Cortana."""

import logging

from src.config import LOG_DIR

LOG_FILE = LOG_DIR / "cortana.log"


def setup_logging() -> logging.Logger:
    """Configure application logging and return the Cortana logger.

    INFO and above go to the log file. The console shows WARNING and ERROR
    only so ordinary pilot use is not cluttered with developer INFO lines.
    """
    LOG_DIR.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[file_handler, console_handler],
        force=True,
    )

    return logging.getLogger("ProjectCortana")
