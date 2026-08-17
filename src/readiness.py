"""Local startup readiness evaluation for Project Cortana.

Readiness is evaluated from local configuration, filesystem, and optional
platform capability checks only. It never contacts OpenAI or Google.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.calendar_service import optional_calendar_dependencies_available
from src.config import (
    CALENDAR_REPOSITORY_FILENAME,
    DOCUMENT_VAULT_FILENAME,
    MEMORY_FILENAME,
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
    REALTIME_MULTIMODAL_ENABLED,
    REALTIME_VOICE_ENABLED,
    REMINDER_REPOSITORY_FILENAME,
    VISION_ANALYSIS_ENABLED,
    VOICE_INTERACTION_ENABLED,
    data_profile_label,
    get_default_app_data_dir,
    is_custom_data_profile,
    product_display_name,
)
from src.settings import (
    INVALID_DATA_DIR_MESSAGE,
    Settings,
    SettingsError,
    load_settings,
    user_facing_settings_error,
)

logger = logging.getLogger("ProjectCortana")


class ReadinessOutcome(Enum):
    """Exactly three local readiness outcomes."""

    READY = "READY"
    READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE = (
        "READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE"
    )
    BLOCKED_BY_REQUIRED_CONFIGURATION = "BLOCKED_BY_REQUIRED_CONFIGURATION"


@dataclass(frozen=True)
class ReadinessReport:
    """Bounded local readiness result. Contains no secrets or user content."""

    outcome: ReadinessOutcome
    required_issues: tuple[str, ...]
    optional_unavailable: tuple[str, ...]
    store_issues: tuple[str, ...]
    data_profile: str

    @property
    def status_label(self) -> str:
        """Return the compact user-facing status label."""
        if self.outcome is ReadinessOutcome.READY:
            return "Ready"
        if self.outcome is ReadinessOutcome.READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE:
            return "Ready with optional features unavailable"
        return "Blocked"


def evaluate_readiness(
    *,
    settings: Settings | None = None,
    data_dir: Path | None = None,
) -> ReadinessReport:
    """Evaluate local readiness without network access.

    When ``settings`` is omitted, environment settings are loaded locally.
    When ``data_dir`` is omitted, the default application data directory is used.
    """
    required_issues: list[str] = []
    optional_unavailable: list[str] = []
    store_issues: list[str] = []

    loaded_settings = settings
    if loaded_settings is None:
        try:
            loaded_settings = load_settings()
        except (SettingsError, ValueError) as error:
            required_issues.append(_required_issue_from_settings_error(error))

    app_data_dir = data_dir if data_dir is not None else get_default_app_data_dir()
    if not _data_dir_ready(app_data_dir):
        if is_custom_data_profile():
            required_issues.append("invalid_data_directory")
        else:
            required_issues.append("data_directory_not_writable")

    if loaded_settings is not None:
        optional_unavailable.extend(_optional_feature_gaps(loaded_settings))

    store_issues.extend(_primary_store_issues(app_data_dir))

    if required_issues:
        outcome = ReadinessOutcome.BLOCKED_BY_REQUIRED_CONFIGURATION
    elif optional_unavailable or store_issues:
        outcome = ReadinessOutcome.READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE
    else:
        outcome = ReadinessOutcome.READY

    return ReadinessReport(
        outcome=outcome,
        required_issues=tuple(required_issues),
        optional_unavailable=tuple(optional_unavailable),
        store_issues=tuple(store_issues),
        data_profile=data_profile_label(),
    )


def format_startup_banner(report: ReadinessReport) -> str:
    """Return the pilot startup identity and readiness lines."""
    lines = [
        product_display_name(),
        f"Status: {report.status_label}.",
    ]
    if report.outcome is ReadinessOutcome.READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE:
        lines.append("Type /status for details or /help to get started.")
    elif report.outcome is ReadinessOutcome.BLOCKED_BY_REQUIRED_CONFIGURATION:
        lines.append(_blocked_startup_explanation(report))
    return "\n".join(lines)


def voice_platform_available() -> bool:
    """Return True when the local platform can support voice capture."""
    if sys.platform != "win32" or not VOICE_INTERACTION_ENABLED:
        return False
    return _module_importable("sounddevice")


def realtime_voice_available() -> bool:
    """Return True when realtime voice is locally supportable."""
    return voice_platform_available() and REALTIME_VOICE_ENABLED


def multimodal_platform_available() -> bool:
    """Return True when realtime multimodal is locally supportable."""
    if not (
        realtime_voice_available()
        and VISION_ANALYSIS_ENABLED
        and REALTIME_MULTIMODAL_ENABLED
    ):
        return False
    return _module_importable("cv2")


def process_isolation_available() -> bool:
    """Return True when process isolation can operate on this platform."""
    return sys.platform == "win32"


def calendar_capability_available(settings: Settings | None = None) -> bool:
    """Return True when calendar dependencies and OAuth config are present."""
    if not optional_calendar_dependencies_available():
        return False
    if settings is None:
        return True
    return settings.google_oauth_client_file is not None


def _required_issue_from_settings_error(error: BaseException) -> str:
    code = getattr(error, "code", "")
    if code:
        return str(code)
    text = str(error)
    if "OPENAI_API_KEY" in text:
        return "missing_api_key"
    if "CORTANA_TTS_VOICE" in text:
        return "invalid_tts_voice"
    if "CORTANA_REALTIME_MODEL" in text:
        return "invalid_realtime_model"
    if "CORTANA_REALTIME_VOICE" in text:
        return "invalid_realtime_voice"
    if "CORTANA_TTS_MODEL" in text:
        return "invalid_tts_model"
    if "CORTANA_TRANSCRIPTION_MODEL" in text:
        return "invalid_transcription_model"
    return "invalid_required_settings"


def _blocked_startup_explanation(report: ReadinessReport) -> str:
    if "missing_api_key" in report.required_issues:
        return user_facing_settings_error(SettingsError("missing_api_key", ""))
    if "invalid_tts_voice" in report.required_issues:
        return user_facing_settings_error(SettingsError("invalid_tts_voice", ""))
    if "invalid_realtime_model" in report.required_issues:
        return user_facing_settings_error(SettingsError("invalid_realtime_model", ""))
    if "invalid_realtime_voice" in report.required_issues:
        return user_facing_settings_error(SettingsError("invalid_realtime_voice", ""))
    if "invalid_tts_model" in report.required_issues:
        return user_facing_settings_error(SettingsError("invalid_tts_model", ""))
    if "invalid_transcription_model" in report.required_issues:
        return user_facing_settings_error(
            SettingsError("invalid_transcription_model", "")
        )
    if "invalid_data_directory" in report.required_issues:
        return INVALID_DATA_DIR_MESSAGE
    if "data_directory_not_writable" in report.required_issues:
        return "Cortana: The data directory is not writable."
    return user_facing_settings_error(ValueError("invalid configuration"))


def _optional_feature_gaps(settings: Settings) -> list[str]:
    gaps: list[str] = []
    if not calendar_capability_available(settings):
        gaps.append("calendar")
    if not voice_platform_available():
        gaps.append("voice")
    if not multimodal_platform_available():
        gaps.append("multimodal")
    if PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED and not process_isolation_available():
        gaps.append("process_isolation")
    return gaps


def _primary_store_issues(data_dir: Path) -> list[str]:
    issues: list[str] = []
    checks = (
        ("memory", data_dir / MEMORY_FILENAME, frozenset({"memories"})),
        ("documents", data_dir / DOCUMENT_VAULT_FILENAME, frozenset({"documents"})),
        (
            "reminders",
            data_dir / REMINDER_REPOSITORY_FILENAME,
            frozenset({"version", "reminders"}),
        ),
        (
            "calendar",
            data_dir / CALENDAR_REPOSITORY_FILENAME,
            frozenset({"version"}),
        ),
    )
    for name, path, required_keys in checks:
        if not _json_store_structurally_ok(path, required_keys):
            issues.append(name)
    return issues


def _json_store_structurally_ok(path: Path, required_keys: frozenset[str]) -> bool:
    """Return True when a store is absent or has a valid top-level JSON shape.

    Missing files are healthy (stores start empty). Existing files are checked
    for JSON object structure only. Record text is never copied into the report.
    """
    if not path.exists():
        return True
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        logger.error(
            "Readiness store read failed store_error_type=%s",
            type(error).__name__,
        )
        return False
    if raw.strip() == "":
        return False
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Readiness store JSON is malformed.")
        return False
    if not isinstance(payload, dict):
        return False
    return required_keys <= payload.keys()


def _data_dir_ready(path: Path) -> bool:
    """Return True when the data directory exists or can be created and written."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            return False
    except OSError as error:
        logger.error(
            "Readiness data directory is not writable error_type=%s",
            type(error).__name__,
        )
        return False
    return True


def _module_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True
