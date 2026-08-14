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
    MAX_COMPARE_CHUNKS_PER_DOCUMENT,
    MAX_COMPARE_CONTEXT_CHARS,
    MAX_COMPARE_DOCUMENTS,
    MAX_GROUNDED_ANSWER_CHARS,
    MAX_RETRIEVED_CHUNKS,
    MAX_RETRIEVED_CONTEXT_CHARS,
    MAX_STORED_DOCUMENTS,
    MAX_SUMMARY_MAP_STAGES,
    MAX_SUMMARY_OUTPUT_CHARS,
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
    MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES,
    PROCESS_FILE_TOOL_ISOLATION_ENABLED,
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
    PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED,
    PROCESS_JOB_ACTIVE_PROCESS_LIMIT,
    PROCESS_RESOURCE_LIMITS_ENABLED,
    PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS,
    TOOL_AI_CONTEXT_INJECTION_ENABLED,
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
    STUDY_PARTNER_ENABLED,
    STUDY_REPOSITORY_FILENAME,
    STUDY_REPOSITORY_SCHEMA_VERSION,
    MAX_STUDY_DOCUMENTS,
    MAX_STORED_STUDY_SESSIONS,
    VISION_ANALYSIS_ENABLED,
    ALLOWED_VISION_EXTENSIONS,
    MAX_VISION_SOURCE_BYTES,
    MAX_VISION_WIDTH,
    MAX_VISION_HEIGHT,
    MAX_VISION_SOURCE_PIXELS,
    MAX_VISION_NORMALIZED_BYTES,
    MAX_VISION_QUESTION_CHARS,
    MAX_VISION_OUTPUT_CHARS,
    VISION_IMAGE_DETAIL,
    VOICE_INTERACTION_ENABLED,
    REALTIME_VOICE_ENABLED,
    REALTIME_MULTIMODAL_ENABLED,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_FRAME_MS,
    REALTIME_VOICE_INPUT_QUEUE_FRAMES,
    REALTIME_VOICE_OUTPUT_QUEUE_FRAMES,
    MAX_REALTIME_VOICE_SESSION_MINUTES,
    DEFAULT_REALTIME_MODEL,
    MAX_REALTIME_VISUAL_WIDTH,
    MAX_REALTIME_VISUAL_HEIGHT,
    MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS,
    REALTIME_VISUAL_SAMPLE_INTERVAL_SECONDS,
    REALTIME_VISUAL_IMAGE_DETAIL,
    REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    MAX_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS,
    MIN_REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS,
    MAX_REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS,
    bounded_realtime_multimodal_transcript_wait_seconds,
    bounded_realtime_multimodal_visual_ack_wait_seconds,
    CONVERSATIONAL_INTELLIGENCE_ENABLED,
    MAX_CONVERSATIONAL_REFERENTS,
    MAX_CONVERSATIONAL_STATE_CHARS,
    MAX_CONVERSATIONAL_TOPIC_CHARS,
    MAX_CONVERSATIONAL_GOAL_CHARS,
    MAX_CONVERSATIONAL_QUESTION_CHARS,
    MAX_CONVERSATIONAL_REFERENT_CHARS,
    CONVERSATIONAL_RECENT_TURN_WINDOW,
    MAX_RECENT_ASSISTANT_ACK_TRACK,
    DEFAULT_RESPONSE_DEPTH,
    SPEECH_DELIVERY_ENABLED,
    SPEECH_CHUNK_CHARS_BRIEF,
    SPEECH_CHUNK_CHARS_NORMAL,
    SPEECH_CHUNK_CHARS_DETAILED,
    MAX_SPEECH_CHUNKS,
    MAX_SPOKEN_LIST_ITEMS,
    MAX_SPEECH_DELIVERY_STATE_CHARS,
    MAX_RECENT_SPOKEN_FINGERPRINTS,
    MAX_VOICE_UTTERANCE_SECONDS,
    MAX_VOICE_PCM_BYTES,
    MAX_VOICE_AUDIO_BYTES,
    VOICE_WAV_HEADER_BYTES,
    MAX_TTS_CHARS,
    MAX_VOICE_TRANSCRIPT_CHARS,
    ALLOWED_TTS_VOICES,
    DEFAULT_TTS_VOICE,
    get_default_document_vault_file_path,
    get_default_evidence_store_dir_path,
    get_default_incident_repository_file_path,
    get_default_memory_file_path,
    get_default_study_repository_file_path,
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
    assert MAX_GROUNDED_ANSWER_CHARS == 4_000
    assert MAX_SUMMARY_OUTPUT_CHARS == 4_000
    assert MAX_SUMMARY_MAP_STAGES == 8
    assert MAX_COMPARE_DOCUMENTS == 2
    assert MAX_COMPARE_CHUNKS_PER_DOCUMENT == 4
    assert MAX_COMPARE_CONTEXT_CHARS == 12_000


def test_study_partner_limits_and_capabilities_are_centralized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Study Partner configuration should use centralized constants."""
    assert STUDY_PARTNER_ENABLED is True
    assert STUDY_REPOSITORY_SCHEMA_VERSION == 1
    assert STUDY_REPOSITORY_FILENAME == "study_state.json"
    assert MAX_STUDY_DOCUMENTS == 5
    assert MAX_STORED_STUDY_SESSIONS == 50
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")
    path = get_default_study_repository_file_path()
    assert path.name == STUDY_REPOSITORY_FILENAME
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents


def test_vision_analysis_limits_and_capabilities_are_centralized() -> None:
    """Visual analysis configuration should use centralized constants."""
    assert VISION_ANALYSIS_ENABLED is True
    assert ALLOWED_VISION_EXTENSIONS == frozenset({".png", ".jpg", ".jpeg", ".webp"})
    assert MAX_VISION_SOURCE_BYTES == 10 * 1024 * 1024
    assert MAX_VISION_WIDTH == 4096
    assert MAX_VISION_HEIGHT == 4096
    assert MAX_VISION_SOURCE_PIXELS == 16_777_216
    assert MAX_VISION_NORMALIZED_BYTES == 5 * 1024 * 1024
    assert MAX_VISION_QUESTION_CHARS == 2_000
    assert MAX_VISION_OUTPUT_CHARS == 4_000
    assert VISION_IMAGE_DETAIL == "auto"


def test_voice_interaction_limits_and_capabilities_are_centralized() -> None:
    """Voice interaction configuration should use centralized derived bounds."""
    assert VOICE_INTERACTION_ENABLED is True
    assert MAX_VOICE_UTTERANCE_SECONDS == 30
    assert MAX_VOICE_PCM_BYTES == 960_000
    assert VOICE_WAV_HEADER_BYTES == 44
    assert MAX_VOICE_AUDIO_BYTES == 960_044
    assert MAX_VOICE_TRANSCRIPT_CHARS == 4_000
    assert MAX_TTS_CHARS == 4_096
    assert DEFAULT_TTS_VOICE == "coral"
    assert "coral" in ALLOWED_TTS_VOICES
    assert "alloy" in ALLOWED_TTS_VOICES
    assert "voice_1234" not in ALLOWED_TTS_VOICES


def test_realtime_voice_limits_and_capabilities_are_centralized() -> None:
    """Realtime voice configuration should use derived 24 kHz frame bounds."""
    assert REALTIME_VOICE_ENABLED is True
    assert REALTIME_VOICE_SAMPLE_RATE_HZ == 24_000
    assert REALTIME_VOICE_FRAME_MS == 20
    assert REALTIME_VOICE_FRAME_BYTES == 960
    assert REALTIME_VOICE_INPUT_QUEUE_FRAMES == 50
    assert REALTIME_VOICE_OUTPUT_QUEUE_FRAMES == 100
    assert MAX_REALTIME_VOICE_SESSION_MINUTES == 20
    assert DEFAULT_REALTIME_MODEL == "gpt-realtime-mini"


def test_realtime_multimodal_limits_and_capabilities_are_centralized() -> None:
    """Realtime multimodal configuration should use bounded live-vision constants."""
    assert REALTIME_MULTIMODAL_ENABLED is True
    assert MAX_REALTIME_VISUAL_WIDTH == 1280
    assert MAX_REALTIME_VISUAL_HEIGHT == 720
    assert MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS == 3.0
    assert REALTIME_VISUAL_SAMPLE_INTERVAL_SECONDS == 0.5
    assert REALTIME_VISUAL_IMAGE_DETAIL == "low"
    assert REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS == 2.5
    assert MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS == 0.25
    assert MAX_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS == 8.0
    assert REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS == 8.0
    assert MIN_REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS == 0.25
    assert MAX_REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS == 30.0


def test_multimodal_transcript_wait_seconds_are_normalized() -> None:
    """H: valid waits are accepted; invalid values are safely normalized."""
    assert bounded_realtime_multimodal_transcript_wait_seconds(2.5) == 2.5
    assert bounded_realtime_multimodal_transcript_wait_seconds(1.0) == 1.0
    assert (
        bounded_realtime_multimodal_transcript_wait_seconds(0.15)
        == MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_transcript_wait_seconds(99.0)
        == MAX_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_transcript_wait_seconds("nope")
        == REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_transcript_wait_seconds(float("nan"))
        == REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_transcript_wait_seconds(float("inf"))
        == REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_transcript_wait_seconds(None)
        == REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    )


def test_multimodal_visual_ack_wait_seconds_are_normalized() -> None:
    assert bounded_realtime_multimodal_visual_ack_wait_seconds(8.0) == 8.0
    assert bounded_realtime_multimodal_visual_ack_wait_seconds(2.0) == 2.0
    assert (
        bounded_realtime_multimodal_visual_ack_wait_seconds(0.05)
        == MIN_REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_visual_ack_wait_seconds(99.0)
        == MAX_REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_visual_ack_wait_seconds("nope")
        == REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_visual_ack_wait_seconds(float("nan"))
        == REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_visual_ack_wait_seconds(float("inf"))
        == REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS
    )
    assert (
        bounded_realtime_multimodal_visual_ack_wait_seconds(None)
        == REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS
    )


def test_conversational_intelligence_limits_are_centralized() -> None:
    """Milestone 27 conversational-intelligence bounds should be centralized."""
    assert CONVERSATIONAL_INTELLIGENCE_ENABLED is True
    assert MAX_CONVERSATIONAL_REFERENTS == 8
    assert MAX_CONVERSATIONAL_STATE_CHARS == 4_000
    assert MAX_CONVERSATIONAL_TOPIC_CHARS == 200
    assert MAX_CONVERSATIONAL_GOAL_CHARS == 500
    assert MAX_CONVERSATIONAL_QUESTION_CHARS == 500
    assert MAX_CONVERSATIONAL_REFERENT_CHARS == 200
    assert CONVERSATIONAL_RECENT_TURN_WINDOW == 6
    assert MAX_RECENT_ASSISTANT_ACK_TRACK == 3
    assert DEFAULT_RESPONSE_DEPTH == "normal"


def test_speech_delivery_limits_are_centralized() -> None:
    """Milestone 29 speech-delivery bounds should be centralized."""
    assert SPEECH_DELIVERY_ENABLED is True
    assert SPEECH_CHUNK_CHARS_BRIEF == 320
    assert SPEECH_CHUNK_CHARS_NORMAL == 220
    assert SPEECH_CHUNK_CHARS_DETAILED == 140
    assert SPEECH_CHUNK_CHARS_BRIEF > SPEECH_CHUNK_CHARS_NORMAL
    assert SPEECH_CHUNK_CHARS_NORMAL > SPEECH_CHUNK_CHARS_DETAILED
    assert MAX_SPEECH_CHUNKS == 24
    assert MAX_SPOKEN_LIST_ITEMS == 6
    assert MAX_SPEECH_DELIVERY_STATE_CHARS == 2_000
    assert MAX_RECENT_SPOKEN_FINGERPRINTS == 8


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
    assert TOOL_AI_CONTEXT_INJECTION_ENABLED is False
    assert PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS == 10
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


def test_process_resource_governance_defaults_disabled() -> None:
    """Milestone 14/15 resource limits and file-tool isolation must remain opt-in."""
    assert PROCESS_RESOURCE_LIMITS_ENABLED is False
    assert PROCESS_FILE_TOOL_ISOLATION_ENABLED is False
    assert PROCESS_JOB_ACTIVE_PROCESS_LIMIT == 1
    assert MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES == 256 * 1024 * 1024
    from src.config import (
        MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS,
        MAX_PROCESS_IPC_REQUEST_BYTES,
        MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS,
        MAX_TOOL_TEXT_SEARCH_QUERY_CHARS,
    )

    assert MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS == 4_096
    assert MAX_PROCESS_IPC_REQUEST_BYTES == 24_576
    assert MAX_TOOL_TEXT_SEARCH_QUERY_CHARS == 200
    assert MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS == 65_536
