"""
Configuration settings for Project Cortana.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Important directories
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Application information
APP_NAME = "Project Cortana"
APP_DATA_DIR_NAME = "ProjectCortana"
VERSION = "0.1.0"

# Session capabilities
HISTORY_PERSISTENCE_ENABLED = False
EXPLICIT_PERSISTENT_MEMORY_ENABLED = True
ACTIVE_MEMORY_PERSISTENCE_ENABLED = False
KNOWLEDGE_VAULT_ENABLED = True
# Document text reaches the AI only through explicit /ask-docs.
DOCUMENT_CONTEXT_INJECTION_ENABLED = True
LOCAL_DOCUMENT_RETRIEVAL_ENABLED = True
SEMANTIC_RETRIEVAL_ENABLED = False
SOURCE_MANIFEST_PERSISTENCE_ENABLED = False

# Explicit persistent memory limits and storage
MAX_MEMORY_TEXT_LENGTH = 2000
MEMORY_FILENAME = "memories.json"

# Active memory context limits (session-only AI reference context)
MAX_ACTIVE_MEMORIES = 10
MAX_ACTIVE_MEMORY_CHARS = 8000

# Knowledge Vault limits and storage
MAX_DOCUMENT_SOURCE_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_TEXT_LENGTH = 500_000
MAX_STORED_DOCUMENTS = 100
ALLOWED_DOCUMENT_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})
DOCUMENT_VAULT_FILENAME = "documents.json"

# Deterministic local document chunking and retrieval
TARGET_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 40
MAX_CHUNKS_PER_DOCUMENT = 500
MAX_RETRIEVED_CHUNKS = 8
MAX_RETRIEVED_CONTEXT_CHARS = 12_000
MAX_SEARCH_RESULT_PREVIEW_CHARS = 200
MAX_SEARCH_DOCS_RESULTS = 5
MAX_GROUNDED_ANSWER_CHARS = 4_000
MAX_SUMMARY_OUTPUT_CHARS = 4_000
MAX_SUMMARY_MAP_STAGES = 8
MAX_COMPARE_DOCUMENTS = 2
MAX_COMPARE_CHUNKS_PER_DOCUMENT = 4
MAX_COMPARE_CONTEXT_CHARS = 12_000

# Study Partner / Tutor (Milestone 22)
STUDY_PARTNER_ENABLED = True
STUDY_REPOSITORY_SCHEMA_VERSION = 1
STUDY_REPOSITORY_FILENAME = "study_state.json"
MAX_STUDY_DOCUMENTS = 5
MAX_STORED_STUDY_SESSIONS = 50
MAX_STUDY_QUESTIONS_PER_SESSION = 200
MAX_STUDY_ATTEMPTS_PER_SESSION = 200
MAX_STUDY_QUESTION_PROMPT_CHARS = 2_000
MAX_STUDY_CHOICE_CHARS = 500
MAX_STUDY_CORRECT_ANSWER_CHARS = 1_000
MAX_STUDY_EXPLANATION_CHARS = 2_000
MAX_STUDY_FEEDBACK_CHARS = 2_000
MAX_STUDY_USER_ANSWER_CHARS = 2_000
MAX_STUDY_TOPIC_CHARS = 500
MAX_STUDY_PENDING_PROMPT_PREVIEW_CHARS = 200

# Multimodal Vision / Visual Understanding (Milestone 23)
# Static authorized local images only. Windows safe-open is required.
VISION_ANALYSIS_ENABLED = True
ALLOWED_VISION_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
ALLOWED_VISION_PILLOW_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
MAX_VISION_SOURCE_BYTES = 10 * 1024 * 1024
MAX_VISION_WIDTH = 4096
MAX_VISION_HEIGHT = 4096
MAX_VISION_SOURCE_PIXELS = 16_777_216
MAX_VISION_NORMALIZED_BYTES = 5 * 1024 * 1024
MAX_VISION_QUESTION_CHARS = 2_000
MAX_VISION_OUTPUT_CHARS = 4_000
VISION_IMAGE_DETAIL = "auto"

# Natural Voice Conversation (Milestone 24)
# Explicit push-to-talk / one utterance. No realtime, wake word, or barge-in.
VOICE_INTERACTION_ENABLED = True
VOICE_SAMPLE_RATE_HZ = 16_000
VOICE_CHANNELS = 1
VOICE_SAMPLE_WIDTH_BYTES = 2
MAX_VOICE_UTTERANCE_SECONDS = 30
MIN_VOICE_UTTERANCE_MS = 250
# PCM capacity for one max-length utterance:
# rate * channels * sample_width * seconds = 16000 * 1 * 2 * 30 = 960000
MAX_VOICE_PCM_BYTES = (
    VOICE_SAMPLE_RATE_HZ
    * VOICE_CHANNELS
    * VOICE_SAMPLE_WIDTH_BYTES
    * MAX_VOICE_UTTERANCE_SECONDS
)
# stdlib wave writes a stable 44-byte RIFF header for this canonical PCM shape.
VOICE_WAV_HEADER_BYTES = 44
MAX_VOICE_AUDIO_BYTES = MAX_VOICE_PCM_BYTES + VOICE_WAV_HEADER_BYTES
MAX_VOICE_TRANSCRIPT_CHARS = 4_000
MAX_TTS_CHARS = 4_096
ALLOWED_TRANSCRIPTION_MODELS = frozenset(
    {
        "whisper-1",
        "gpt-transcribe",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
        "gpt-4o-transcribe-diarize",
    }
)
ALLOWED_TTS_MODELS = frozenset(
    {
        "tts-1",
        "tts-1-hd",
        "gpt-4o-mini-tts",
        "gpt-4o-mini-tts-2025-12-15",
    }
)
ALLOWED_TTS_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "coral"
VOICE_CAPTURE_BLOCKSIZE = 1024

# Realtime Voice Conversation (Milestone 25)
# Explicit /voice-realtime session. No wake word, ambient listen, or auto-reconnect.
REALTIME_VOICE_ENABLED = True
REALTIME_VOICE_SAMPLE_RATE_HZ = 24_000
REALTIME_VOICE_CHANNELS = 1
REALTIME_VOICE_SAMPLE_WIDTH_BYTES = 2
REALTIME_VOICE_FRAME_MS = 20
REALTIME_VOICE_FRAME_SAMPLES = (
    REALTIME_VOICE_SAMPLE_RATE_HZ * REALTIME_VOICE_FRAME_MS // 1000
)
REALTIME_VOICE_FRAME_BYTES = (
    REALTIME_VOICE_FRAME_SAMPLES
    * REALTIME_VOICE_CHANNELS
    * REALTIME_VOICE_SAMPLE_WIDTH_BYTES
)
# ~1 second of 20 ms frames.
REALTIME_VOICE_INPUT_QUEUE_MS = 1_000
REALTIME_VOICE_INPUT_QUEUE_FRAMES = (
    REALTIME_VOICE_INPUT_QUEUE_MS // REALTIME_VOICE_FRAME_MS
)
REALTIME_VOICE_INPUT_QUEUE_BYTES = (
    REALTIME_VOICE_INPUT_QUEUE_FRAMES * REALTIME_VOICE_FRAME_BYTES
)
# ~2 seconds of assistant audio buffering.
REALTIME_VOICE_OUTPUT_QUEUE_MS = 2_000
REALTIME_VOICE_OUTPUT_QUEUE_FRAMES = (
    REALTIME_VOICE_OUTPUT_QUEUE_MS // REALTIME_VOICE_FRAME_MS
)
REALTIME_VOICE_OUTPUT_QUEUE_BYTES = (
    REALTIME_VOICE_OUTPUT_QUEUE_FRAMES * REALTIME_VOICE_FRAME_BYTES
)
MAX_REALTIME_VOICE_SESSION_MINUTES = 20
MAX_CANCELLED_REALTIME_RESPONSE_IDS = 16
REALTIME_VOICE_RECV_TIMEOUT_SECONDS = 0.02
REALTIME_VOICE_CAPTURE_BLOCKSIZE = REALTIME_VOICE_FRAME_SAMPLES
ALLOWED_REALTIME_MODELS = frozenset(
    {
        "gpt-realtime",
        "gpt-realtime-1.5",
        "gpt-realtime-2",
        "gpt-realtime-2.1",
        "gpt-realtime-2.1-mini",
        "gpt-realtime-2025-08-28",
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-realtime-preview-2025-06-03",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-realtime-preview-2024-12-17",
        "gpt-realtime-mini",
        "gpt-realtime-mini-2025-10-06",
        "gpt-realtime-mini-2025-12-15",
        "gpt-audio-1.5",
        "gpt-audio-mini",
        "gpt-audio-mini-2025-10-06",
        "gpt-audio-mini-2025-12-15",
    }
)
ALLOWED_REALTIME_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)
DEFAULT_REALTIME_MODEL = "gpt-realtime-mini"
DEFAULT_REALTIME_VOICE = "coral"

# Security incident repository and evidence foundation (Milestone 8)
INCIDENT_REPOSITORY_ENABLED = True
EVIDENCE_COPY_ENABLED = True
CHAIN_OF_CUSTODY_ENABLED = True
AUTOMATED_RESPONSE_ENABLED = False
EXTERNAL_THREAT_INTELLIGENCE_LOOKUPS_ENABLED = False
INCIDENT_AI_CONTEXT_INJECTION_ENABLED = False
INCIDENT_REPOSITORY_PERSISTENCE_ENABLED = True
INCIDENT_SINGLE_INSTANCE_COORDINATION_ENABLED = False

INCIDENT_REPOSITORY_SCHEMA_VERSION = 1
INCIDENT_REPOSITORY_FILENAME = "incidents.json"
EVIDENCE_STORE_DIRNAME = "evidence"

MAX_EVENT_TITLE_LENGTH = 200
MAX_EVENT_DESCRIPTION_LENGTH = 5_000
MAX_EVENT_SOURCE_LENGTH = 200
MAX_INCIDENT_TITLE_LENGTH = 200
MAX_INCIDENT_SUMMARY_LENGTH = 5_000
MAX_INDICATOR_VALUE_LENGTH = 2_000
MAX_INDICATOR_NOTES_LENGTH = 2_000
MAX_EVIDENCE_TITLE_LENGTH = 200
MAX_EVIDENCE_DESCRIPTION_LENGTH = 5_000
MAX_EVIDENCE_FILENAME_LENGTH = 255
MAX_EVIDENCE_SOURCE_BYTES = 50 * 1024 * 1024
MAX_NOTE_TEXT_LENGTH = 5_000
MAX_CUSTODY_ACTOR_LENGTH = 200
MAX_CUSTODY_REASON_LENGTH = 1_000
MAX_CUSTODY_NOTES_LENGTH = 2_000
MAX_TAG_LENGTH = 64
MAX_TAGS_PER_RECORD = 20
MAX_SECURITY_LIST_PREVIEW_CHARS = 120
INDICATOR_CONFIDENCE_MIN = 0
INDICATOR_CONFIDENCE_MAX = 100

# Defensive tool framework (Milestone 9)
DEFENSIVE_TOOL_FRAMEWORK_ENABLED = True
TOOL_SCOPE_ENFORCEMENT_ENABLED = True
TOOL_HUMAN_APPROVAL_ENABLED = True
TOOL_DRY_RUN_ENFORCEMENT_ENABLED = True
ARBITRARY_SHELL_EXECUTION_ENABLED = False
EXTERNAL_TOOL_EXECUTION_ENABLED = False
AUTONOMOUS_REMEDIATION_ENABLED = False
TOOL_AUDIT_PERSISTENCE_ENABLED = True
TOOL_SINGLE_INSTANCE_COORDINATION_ENABLED = False
TOOL_AI_CONTEXT_INJECTION_ENABLED = False

# Process-isolated defensive tool execution foundation (Milestone 13)
# Both default False. Termination cannot independently enable process execution.
PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED = False
PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED = False
# Process isolation improves terminability for a tiny trusted subset only.

# Process resource governance and safe file-opening foundation (Milestone 14/15)
# Resource limits apply only to process-isolated tools when explicitly enabled.
# File-tool isolation requires PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED as well.
# Alone, PROCESS_FILE_TOOL_ISOLATION_ENABLED does not make any tool eligible.
PROCESS_RESOURCE_LIMITS_ENABLED = False
PROCESS_FILE_TOOL_ISOLATION_ENABLED = False
# CPU-rate and handle-count Job Object limits are deferred.

TOOL_CONTROL_REPOSITORY_SCHEMA_VERSION = 1
TOOL_CONTROL_REPOSITORY_FILENAME = "tool_control.json"
TOOL_PROCESS_SCRATCH_DIRNAME = "tool_process_scratch"

MAX_TOOL_NAME_LENGTH = 100
MAX_TOOL_DESCRIPTION_LENGTH = 1_000
MAX_TOOL_JUSTIFICATION_LENGTH = 2_000
MAX_TOOL_SCOPE_NAME_LENGTH = 200
MAX_TOOL_SCOPE_NOTES_LENGTH = 2_000
MAX_TOOL_APPROVAL_REASON_LENGTH = 1_000
MAX_TOOL_RESULT_SUMMARY_LENGTH = 2_000
MAX_TOOL_AUDIT_DETAILS_LENGTH = 1_000
MAX_TOOL_PARAMETER_STRING_LENGTH = 2_000
MAX_TOOL_FILE_BYTES = 10 * 1024 * 1024
MAX_TOOL_OUTPUT_CHARS = 8_000
MAX_TOOL_TEXT_SEARCH_MATCHES = 50
MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS = 120
# Bound for one assembled line during process-isolated text-search streaming.
# Much larger than preview length, large enough for realistic logs, far below
# MAX_TOOL_FILE_BYTES so a no-newline file cannot grow memory without bound.
MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS = 65_536
# Existing tool schema bound for the literal search query.
MAX_TOOL_TEXT_SEARCH_QUERY_CHARS = 200
MAX_TOOL_TIMEOUT_SECONDS = 30
DEFAULT_TOOL_TIMEOUT_SECONDS = 10
MAX_TOOL_LIST_PREVIEW_CHARS = 120
FINGERPRINT_DISPLAY_PREFIX_CHARS = 12

# Milestone 13 process-isolation IPC and termination bounds
PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS = 10
PROCESS_TERMINATION_CONFIRMATION_TIMEOUT_SECONDS = 5
# Raised for Milestone 15 nested file-authorization payloads (path bounds).
MAX_PROCESS_IPC_REQUEST_BYTES = 24_576
MAX_PROCESS_IPC_RESPONSE_BYTES = 16_384
MAX_PROCESS_DIAGNOSTIC_STDOUT_BYTES = 4_096
MAX_PROCESS_DIAGNOSTIC_STDERR_BYTES = 4_096
MAX_PROCESS_PARAMETER_COUNT = 16
MAX_PROCESS_PARAMETER_NAME_CHARS = 64
MAX_PROCESS_PARAMETER_STRING_CHARS = 2_000
# Depth 3 is required for text-search structured results:
# structured_data → matches[] → {line_number, preview} → scalar values.
MAX_PROCESS_PARAMETER_NESTING_DEPTH = 3
MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS = 4_096
PROCESS_IPC_SCHEMA_VERSION = 1

# Milestone 14 Job Object bounds (Windows-only when resource limits are enabled).
# 256 MiB covers Python 3.13 startup, system-summary, simulated-log-check, and
# typical Windows/AV overhead while still providing meaningful containment.
# This is not a complete memory sandbox.
MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES = 256 * 1024 * 1024
PROCESS_JOB_ACTIVE_PROCESS_LIMIT = 1
MIN_PROCESS_ISOLATED_JOB_MEMORY_BYTES = 64 * 1024 * 1024
MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES_CAP = 1024 * 1024 * 1024

# Defensive workflow orchestration (Milestone 10)
DEFENSIVE_WORKFLOW_ORCHESTRATION_ENABLED = True
WORKFLOW_AI_CONTEXT_INJECTION_ENABLED = False
WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED = False
WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED = False
WORKFLOW_PARALLEL_EXECUTION_ENABLED = False
WORKFLOW_BACKGROUND_EXECUTION_ENABLED = False
WORKFLOW_NESTED_PLAYBOOKS_ENABLED = False

# Persistent workflow operations and incident linkage (Milestone 11)
WORKFLOW_RUN_PERSISTENCE_ENABLED = True
WORKFLOW_INCIDENT_LINKAGE_ENABLED = True
WORKFLOW_SINGLE_INSTANCE_COORDINATION_ENABLED = False
WORKFLOW_REPOSITORY_SCHEMA_VERSION = 1
WORKFLOW_REPOSITORY_FILENAME = "workflow_runs.json"

# Controlled security analyst assistance foundation (Milestone 12)
# Explicit prepare/run/save-note only. Ordinary chat never receives packets.
AI_INCIDENT_ANALYSIS_ENABLED = False
AI_INCIDENT_NOTE_SAVE_ENABLED = False
# WORKFLOW_AI_CONTEXT_INJECTION_ENABLED (defined with Milestone 10 flags)
# independently gates whether Milestone 11 safe workflow/tool summaries may be
# selected into an incident-analysis packet. It does not enable analysis itself.

AI_INCIDENT_ANALYSIS_NOTE_AUTHOR = "ai-analyst-assistance"
AI_INCIDENT_ANALYSIS_NOTE_TAG = "ai-assisted"
AI_INCIDENT_ANALYSIS_NOTE_TYPE = "hypothesis"

MAX_ANALYSIS_EVENTS = 10
MAX_ANALYSIS_INDICATORS = 20
MAX_ANALYSIS_NOTES = 10
MAX_ANALYSIS_WORKFLOW_SUMMARIES = 5
MAX_ANALYSIS_TOOL_SUMMARIES = 10
MAX_INCIDENT_ANALYSIS_PACKET_CHARS = 32_000
MAX_INCIDENT_ANALYSIS_OUTPUT_CHARS = 4_000
MAX_RETAINED_INCIDENT_ANALYSES = 50
MAX_INCIDENT_ANALYSIS_AUDIT_ENTRIES = 200
MAX_ANALYSIS_FIELD_CHARS = 800
MAX_ANALYSIS_SOURCE_LABEL_CHARS = 100
MAX_ANALYSIS_TAG_CHARS = 64
MAX_ANALYSIS_SAFE_SUMMARY_CHARS = 500
ANALYSIS_TRUNCATION_MARKER = " [truncated]"

MAX_WORKFLOW_STEPS = 8
MAX_WORKFLOW_RUNTIME_SECONDS = 60
MAX_WORKFLOW_NAME_LENGTH = 100
MAX_WORKFLOW_DESCRIPTION_LENGTH = 1_000
MAX_WORKFLOW_VERSION_LENGTH = 32
MAX_WORKFLOW_STEP_ID_LENGTH = 100
MAX_WORKFLOW_STEP_DESCRIPTION_LENGTH = 500
MAX_WORKFLOW_ERROR_MESSAGE_LENGTH = 500
MAX_WORKFLOW_RUNS_RETAINED = 100
MAX_WORKFLOW_AUDIT_ENTRIES_RETAINED = 500
MAX_WORKFLOW_LIST_PREVIEW_CHARS = 120
MAX_WORKFLOW_AUDIT_DETAILS_LENGTH = 1_000

# Personal assistant scheduling core (Milestone 19)
REMINDER_REPOSITORY_ENABLED = True
REMINDER_REPOSITORY_PERSISTENCE_ENABLED = True
REMINDER_REPOSITORY_SCHEMA_VERSION = 1
REMINDER_REPOSITORY_FILENAME = "reminders.json"

MAX_REMINDER_TITLE_LENGTH = 200
MAX_REMINDER_MESSAGE_LENGTH = 2_000
MAX_STORED_REMINDERS = 500
MAX_REMINDER_AUDIT_ENTRIES = 5_000
MAX_REMINDER_DAILY_INTERVAL = 365
MAX_REMINDER_WEEKLY_INTERVAL = 52
MAX_REMINDER_MONTHLY_INTERVAL = 24
MAX_REMINDER_LIST_RESULTS = 100
MAX_REMINDER_LIST_PREVIEW_CHARS = 120
MAX_REMINDER_SHOW_AUDIT_ENTRIES = 10
MAX_REMINDER_RECURRENCE_SEARCH_STEPS = 10_000

# Calendar integration (Milestone 20)
CALENDAR_REPOSITORY_ENABLED = True
CALENDAR_REPOSITORY_PERSISTENCE_ENABLED = True
CALENDAR_REPOSITORY_SCHEMA_VERSION = 1
CALENDAR_REPOSITORY_FILENAME = "calendar_control.json"

GOOGLE_CALENDAR_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
)

MAX_CALENDAR_ACCOUNT_DISPLAY_LABEL_LENGTH = 200
MAX_CALENDAR_ID_LENGTH = 200
MAX_CALENDAR_EVENT_ID_LENGTH = 1024
MAX_CALENDAR_EVENT_TITLE_LENGTH = 200
MAX_CALENDAR_EVENT_DESCRIPTION_LENGTH = 2_000
MAX_CALENDAR_PROPOSALS = 200
MAX_CALENDAR_AUDIT_ENTRIES = 5_000
CALENDAR_PROPOSAL_TTL_SECONDS = 15 * 60
MAX_CALENDAR_LIST_RESULTS = 50
MAX_CALENDAR_LIST_PREVIEW_CHARS = 120
MAX_CALENDAR_FREEBUSY_CALENDARS = 10
MAX_CALENDAR_EVENT_WINDOW_DAYS = 31
MAX_CALENDAR_ERROR_MESSAGE_LENGTH = 500
SECRET_STORE_SERVICE_NAME = "ProjectCortana"


def _default_app_data_dir() -> Path:
    """Return the user-local application data directory for Project Cortana."""
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / APP_DATA_DIR_NAME
        return Path.home() / "AppData" / "Local" / APP_DATA_DIR_NAME

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return Path(xdg_data_home) / APP_DATA_DIR_NAME

    return Path.home() / ".local" / "share" / APP_DATA_DIR_NAME


def get_default_memory_file_path() -> Path:
    """Return the default user-local path for explicit persistent memories."""
    return _default_app_data_dir() / MEMORY_FILENAME


def get_default_document_vault_file_path() -> Path:
    """Return the default user-local path for Knowledge Vault documents."""
    return _default_app_data_dir() / DOCUMENT_VAULT_FILENAME


def get_default_incident_repository_file_path() -> Path:
    """Return the default user-local path for the incident repository JSON file."""
    return _default_app_data_dir() / INCIDENT_REPOSITORY_FILENAME


def get_default_evidence_store_dir_path() -> Path:
    """Return the default user-local directory for copied evidence files."""
    return _default_app_data_dir() / EVIDENCE_STORE_DIRNAME


def get_default_tool_control_repository_file_path() -> Path:
    """Return the default user-local path for the tool-control repository JSON file."""
    return _default_app_data_dir() / TOOL_CONTROL_REPOSITORY_FILENAME


def get_default_workflow_repository_file_path() -> Path:
    """Return the default user-local path for the workflow repository JSON file."""
    return _default_app_data_dir() / WORKFLOW_REPOSITORY_FILENAME


def get_default_reminder_repository_file_path() -> Path:
    """Return the default user-local path for the reminder repository JSON file."""
    return _default_app_data_dir() / REMINDER_REPOSITORY_FILENAME


def get_default_calendar_repository_file_path() -> Path:
    """Return the default user-local path for the calendar control JSON file."""
    return _default_app_data_dir() / CALENDAR_REPOSITORY_FILENAME


def get_default_study_repository_file_path() -> Path:
    """Return the default user-local path for the Study Partner JSON file."""
    return _default_app_data_dir() / STUDY_REPOSITORY_FILENAME


def get_default_tool_process_scratch_dir_path() -> Path:
    """Return the dedicated scratch directory for process-isolated tool IPC.

    Created only by trusted application code. Not an incident, tool-control, or
    workflow repository path. Eligible v1 tools do not depend on this directory
    for tool logic; it holds parent-created result files only.
    """
    return _default_app_data_dir() / TOOL_PROCESS_SCRATCH_DIRNAME
