import logging

from src.logger import LOG_FILE, setup_logging


def test_setup_logging_returns_named_logger() -> None:
    logger = setup_logging()

    assert isinstance(logger, logging.Logger)
    assert logger.name == "ProjectCortana"


def test_setup_logging_creates_log_file() -> None:
    setup_logging()

    assert LOG_FILE.exists()
    assert LOG_FILE.is_file()


def test_setup_logging_configures_root_handlers() -> None:
    setup_logging()

    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) == 2
    stream_handlers = [
        handler
        for handler in root_logger.handlers
        if handler.__class__.__name__ == "StreamHandler"
        and handler.__class__.__module__ == "logging"
    ]
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if handler.__class__.__name__ == "FileHandler"
    ]
    assert file_handlers
    assert file_handlers[0].level == logging.INFO
    assert stream_handlers
    assert stream_handlers[0].level == logging.WARNING
    