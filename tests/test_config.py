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
    ARBITRARY_SHELL_EXECUTION_ENABLED,
    AUTONOMOUS_REMEDIATION_ENABLED,
    DEFENSIVE_TOOL_FRAMEWORK_ENABLED,
    EVIDENCE_STORE_DIRNAME,
    EXTERNAL_TOOL_EXECUTION_ENABLED,
    INCIDENT_REPOSITORY_FILENAME,
    TOOL_CONTROL_REPOSITORY_FILENAME,
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
    PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED,
    TOOL_DRY_RUN_ENFORCEMENT_ENABLED,
    TOOL_HUMAN_APPROVAL_ENABLED,
    TOOL_SCOPE_ENFORCEMENT_ENABLED,
    DEFENSIVE_WORKFLOW_ORCHESTRATION_ENABLED,
    MAX_WORKFLOW_RUNTIME_SECONDS,
    MAX_WORKFLOW_STEPS,
    WORKFLOW_AI_CONTEXT_INJECTION_ENABLED,
    WORKFLOW_BACKGROUND_EXECUTION_ENABLED,
    WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED,
    WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED,
    WORKFLOW_INCIDENT_LINKAGE_ENABLED,
    WORKFLOW_NESTED_PLAYBOOKS_ENABLED,
    WORKFLOW_PARALLEL_EXECUTION_ENABLED,
    WORKFLOW_REPOSITORY_FILENAME,
    WORKFLOW_REPOSITORY_SCHEMA_VERSION,
    WORKFLOW_RUN_PERSISTENCE_ENABLED,
    WORKFLOW_SINGLE_INSTANCE_COORDINATION_ENABLED,
    AI_INCIDENT_ANALYSIS_ENABLED,
    AI_INCIDENT_ANALYSIS_NOTE_AUTHOR,
    AI_INCIDENT_ANALYSIS_NOTE_TAG,
    AI_INCIDENT_ANALYSIS_NOTE_TYPE,
    AI_INCIDENT_NOTE_SAVE_ENABLED,
    INCIDENT_AI_CONTEXT_INJECTION_ENABLED,
    MAX_ANALYSIS_EVENTS,
    MAX_ANALYSIS_INDICATORS,
    MAX_ANALYSIS_NOTES,
    MAX_ANALYSIS_TOOL_SUMMARIES,
    MAX_ANALYSIS_WORKFLOW_SUMMARIES,
    MAX_INCIDENT_ANALYSIS_OUTPUT_CHARS,
    MAX_INCIDENT_ANALYSIS_PACKET_CHARS,
    MAX_RETAINED_INCIDENT_ANALYSES,
    get_default_document_vault_file_path,
    get_default_evidence_store_dir_path,
    get_default_incident_repository_file_path,
    get_default_memory_file_path,
    get_default_tool_control_repository_file_path,
    get_default_workflow_repository_file_path,
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


def test_defensive_tool_framework_capabilities_are_centralized() -> None:
    """Milestone 9 capability flags should remain defensive and human-supervised."""
    assert DEFENSIVE_TOOL_FRAMEWORK_ENABLED is True
    assert TOOL_SCOPE_ENFORCEMENT_ENABLED is True
    assert TOOL_HUMAN_APPROVAL_ENABLED is True
    assert TOOL_DRY_RUN_ENFORCEMENT_ENABLED is True
    assert ARBITRARY_SHELL_EXECUTION_ENABLED is False
    assert EXTERNAL_TOOL_EXECUTION_ENABLED is False
    assert AUTONOMOUS_REMEDIATION_ENABLED is False
    assert TOOL_CONTROL_REPOSITORY_FILENAME == "tool_control.json"


def test_default_tool_control_repository_path_is_outside_project_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default tool-control repository storage should use a user-local path."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")

    path = get_default_tool_control_repository_file_path()

    assert path.name == TOOL_CONTROL_REPOSITORY_FILENAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents
    assert "src" not in path.parts
    assert "tests" not in path.parts


def test_defensive_workflow_orchestration_capabilities_are_centralized() -> None:
    """Milestone 10 workflow flags should remain bounded and non-autonomous."""
    assert DEFENSIVE_WORKFLOW_ORCHESTRATION_ENABLED is True
    assert WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED is False
    assert WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED is False
    assert WORKFLOW_PARALLEL_EXECUTION_ENABLED is False
    assert WORKFLOW_BACKGROUND_EXECUTION_ENABLED is False
    assert WORKFLOW_NESTED_PLAYBOOKS_ENABLED is False
    assert WORKFLOW_AI_CONTEXT_INJECTION_ENABLED is False
    assert MAX_WORKFLOW_STEPS == 8
    assert MAX_WORKFLOW_RUNTIME_SECONDS == 60


def test_workflow_persistence_and_linkage_capabilities_are_independent() -> None:
    """Milestone 11 persistence and incident linkage must be separately flagged."""
    assert WORKFLOW_RUN_PERSISTENCE_ENABLED is True
    assert WORKFLOW_INCIDENT_LINKAGE_ENABLED is True
    assert WORKFLOW_SINGLE_INSTANCE_COORDINATION_ENABLED is False
    assert WORKFLOW_REPOSITORY_SCHEMA_VERSION == 1
    assert WORKFLOW_REPOSITORY_FILENAME == "workflow_runs.json"


def test_default_workflow_repository_path_is_outside_project_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default workflow repository storage should use a user-local path."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")

    path = get_default_workflow_repository_file_path()

    assert path.name == WORKFLOW_REPOSITORY_FILENAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents
    assert "src" not in path.parts
    assert "tests" not in path.parts


def test_incident_ai_analysis_capabilities_default_disabled() -> None:
    """Milestone 12 analysis and note saving must remain opt-in and bounded."""
    assert AI_INCIDENT_ANALYSIS_ENABLED is False
    assert AI_INCIDENT_NOTE_SAVE_ENABLED is False
    assert WORKFLOW_AI_CONTEXT_INJECTION_ENABLED is False
    assert INCIDENT_AI_CONTEXT_INJECTION_ENABLED is False
    assert AI_INCIDENT_ANALYSIS_NOTE_AUTHOR == "ai-analyst-assistance"
    assert AI_INCIDENT_ANALYSIS_NOTE_TAG == "ai-assisted"
    assert AI_INCIDENT_ANALYSIS_NOTE_TYPE == "hypothesis"
    assert MAX_ANALYSIS_EVENTS == 10
    assert MAX_ANALYSIS_INDICATORS == 20
    assert MAX_ANALYSIS_NOTES == 10
    assert MAX_ANALYSIS_WORKFLOW_SUMMARIES == 5
    assert MAX_ANALYSIS_TOOL_SUMMARIES == 10
    assert 20_000 <= MAX_INCIDENT_ANALYSIS_PACKET_CHARS <= 40_000
    assert MAX_INCIDENT_ANALYSIS_OUTPUT_CHARS == 4_000
    assert MAX_RETAINED_INCIDENT_ANALYSES == 50


def test_process_isolated_tool_execution_defaults_disabled() -> None:
    """Milestone 13 process isolation must remain opt-in and dual-flagged."""
    assert PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED is False
    assert PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED is False
