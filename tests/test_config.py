import pytest

from src.config import (
    APP_NAME,
    APP_DATA_DIR_NAME,
    DATA_DIR,
    DOCS_DIR,
    EXPLICIT_PERSISTENT_MEMORY_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    LOG_DIR,
    MAX_MEMORY_TEXT_LENGTH,
    MEMORY_FILENAME,
    PROJECT_ROOT,
    TESTS_DIR,
    VERSION,
    get_default_memory_file_path,
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


def test_explicit_persistent_memory_capability_is_enabled() -> None:
    """Explicit persistent memory should be enabled for this milestone."""
    assert EXPLICIT_PERSISTENT_MEMORY_ENABLED is True


def test_memory_limits_and_filename_are_centralized() -> None:
    """Memory configuration should use centralized constants."""
    assert MAX_MEMORY_TEXT_LENGTH == 2000
    assert MEMORY_FILENAME == "memories.json"
    assert APP_DATA_DIR_NAME == "ProjectCortana"


def test_default_memory_path_is_outside_project_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default memory storage should use a user-local application data path."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")

    path = get_default_memory_file_path()

    assert path.name == MEMORY_FILENAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents
    assert "src" not in path.parts
    assert "tests" not in path.parts
