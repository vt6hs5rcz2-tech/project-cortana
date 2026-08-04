from src.config import (
    APP_NAME,
    DATA_DIR,
    DOCS_DIR,
    HISTORY_PERSISTENCE_ENABLED,
    LOG_DIR,
    PROJECT_ROOT,
    TESTS_DIR,
    VERSION,
)


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_expected_directories_are_under_project_root() -> None:
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert DOCS_DIR == PROJECT_ROOT / "docs"
    assert LOG_DIR == PROJECT_ROOT / "logs"
    assert TESTS_DIR == PROJECT_ROOT / "tests"


def test_application_metadata() -> None:
    assert APP_NAME == "Project Cortana"
    assert VERSION == "0.1.0"


def test_history_persistence_capability_is_disabled() -> None:
    """Conversation history persistence should remain disabled in this milestone."""
    assert HISTORY_PERSISTENCE_ENABLED is False