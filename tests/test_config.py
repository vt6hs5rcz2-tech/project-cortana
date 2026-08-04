import pytest

from src.config import (
    ACTIVE_MEMORY_PERSISTENCE_ENABLED,
    ALLOWED_DOCUMENT_EXTENSIONS,
    APP_NAME,
    APP_DATA_DIR_NAME,
    CHUNK_OVERLAP,
    DATA_DIR,
    DOCS_DIR,
    DOCUMENT_CONTEXT_INJECTION_ENABLED,
    DOCUMENT_VAULT_FILENAME,
    EXPLICIT_PERSISTENT_MEMORY_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    KNOWLEDGE_VAULT_ENABLED,
    LOCAL_DOCUMENT_RETRIEVAL_ENABLED,
    LOG_DIR,
    MAX_ACTIVE_MEMORIES,
    MAX_ACTIVE_MEMORY_CHARS,
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_DOCUMENT_SOURCE_BYTES,
    MAX_DOCUMENT_TEXT_LENGTH,
    MAX_MEMORY_TEXT_LENGTH,
    MAX_RETRIEVED_CHUNKS,
    MAX_RETRIEVED_CONTEXT_CHARS,
    MAX_STORED_DOCUMENTS,
    MEMORY_FILENAME,
    MIN_CHUNK_LENGTH,
    PROJECT_ROOT,
    SEMANTIC_RETRIEVAL_ENABLED,
    SOURCE_MANIFEST_PERSISTENCE_ENABLED,
    TARGET_CHUNK_SIZE,
    TESTS_DIR,
    VERSION,
    EVIDENCE_STORE_DIRNAME,
    INCIDENT_REPOSITORY_FILENAME,
    get_default_document_vault_file_path,
    get_default_evidence_store_dir_path,
    get_default_incident_repository_file_path,
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
    assert MAX_ACTIVE_MEMORIES == 10
    assert MAX_ACTIVE_MEMORY_CHARS == 8000
    assert ACTIVE_MEMORY_PERSISTENCE_ENABLED is False


def test_knowledge_vault_limits_and_capabilities_are_centralized() -> None:
    """Knowledge Vault configuration should use centralized constants."""
    assert KNOWLEDGE_VAULT_ENABLED is True
    assert DOCUMENT_CONTEXT_INJECTION_ENABLED is True
    assert LOCAL_DOCUMENT_RETRIEVAL_ENABLED is True
    assert SEMANTIC_RETRIEVAL_ENABLED is False
    assert SOURCE_MANIFEST_PERSISTENCE_ENABLED is False
    assert MAX_DOCUMENT_SOURCE_BYTES == 10 * 1024 * 1024
    assert MAX_DOCUMENT_TEXT_LENGTH == 500_000
    assert MAX_STORED_DOCUMENTS == 100
    assert ALLOWED_DOCUMENT_EXTENSIONS == frozenset({".txt", ".md", ".pdf"})
    assert DOCUMENT_VAULT_FILENAME == "documents.json"
    assert TARGET_CHUNK_SIZE == 1200
    assert CHUNK_OVERLAP == 150
    assert MIN_CHUNK_LENGTH == 40
    assert MAX_CHUNKS_PER_DOCUMENT == 500
    assert MAX_RETRIEVED_CHUNKS == 8
    assert MAX_RETRIEVED_CONTEXT_CHARS == 12_000


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


def test_default_document_vault_path_is_outside_project_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Knowledge Vault storage should use a user-local application data path."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")

    path = get_default_document_vault_file_path()

    assert path.name == DOCUMENT_VAULT_FILENAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents
    assert "src" not in path.parts
    assert "tests" not in path.parts


def test_default_incident_repository_path_is_outside_project_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default incident repository storage should use a user-local application data path."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")

    path = get_default_incident_repository_file_path()

    assert path.name == INCIDENT_REPOSITORY_FILENAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents
    assert "src" not in path.parts
    assert "tests" not in path.parts


def test_default_evidence_store_path_is_outside_project_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default evidence store directory should use a user-local application data path."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")

    path = get_default_evidence_store_dir_path()

    assert path.name == EVIDENCE_STORE_DIRNAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents
    assert "src" not in path.parts
    assert "tests" not in path.parts
