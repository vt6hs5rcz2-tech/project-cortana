"""Pre-M30 Deep Audit #4: failure semantics, cleanup, datetime, leaks, platform.

Discovery only.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.active_memory import ActiveMemoryContext
from src.calendar_models import CalendarValidationError, local_wall_pair_to_utc
from src.commands import clear_conversation_history
from src.config import PROJECT_ROOT
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.memory import create_memory
from src.reminder import (
    ReminderValidationError,
    local_wall_to_utc_iso,
    parse_local_wall_datetime,
)
from src.speech_delivery import SpeechDeliveryState
from src.retrieval_session import RetrievalSession


def test_deep_f8_spring_forward_nonexistent_time_is_rejected() -> None:
    """OUTSIDE-F8: DST spring-forward holes should be rejected, not normalized."""
    with pytest.raises(ReminderValidationError):
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-03-08 02:30"),
            "America/New_York",
        )


def test_deep_f8_fall_back_ambiguous_time_is_rejected_or_explicit() -> None:
    """Ambiguous fall-back local times should not silently pick fold=0."""
    with pytest.raises(ReminderValidationError, match="ambiguous"):
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-11-01 01:30"),
            "America/New_York",
        )


def test_deep_calendar_and_reminder_interpret_dst_identically() -> None:
    """Ordinary local datetimes must match across reminder and calendar conversion."""
    reminder_utc = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-08-14 09:00"),
        "America/New_York",
    )
    start_utc, end_utc = local_wall_pair_to_utc(
        start_local="2026-08-14 09:00",
        end_local="2026-08-14 10:00",
        timezone_name="America/New_York",
    )
    assert start_utc == reminder_utc
    assert end_utc.startswith("2026-08-14T14:00:00")


def test_deep_f8_spring_forward_range_collapses_in_calendar() -> None:
    """Sibling of OUTSIDE-F8: reminder and calendar reject the same DST gap."""
    with pytest.raises(ReminderValidationError):
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-03-08 02:30"),
            "America/New_York",
        )
    with pytest.raises(CalendarValidationError):
        local_wall_pair_to_utc(
            start_local="2026-03-08 02:30",
            end_local="2026-03-08 03:30",
            timezone_name="America/New_York",
        )
    reminder_start = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-03-08 03:30"),
        "America/New_York",
    )
    start_utc, end_utc = local_wall_pair_to_utc(
        start_local="2026-03-08 03:30",
        end_local="2026-03-08 04:30",
        timezone_name="America/New_York",
    )
    assert start_utc == reminder_start
    assert start_utc < end_utc


def test_deep_leap_day_year_boundary_midnight_and_naive() -> None:
    leap = local_wall_to_utc_iso(
        parse_local_wall_datetime("2024-02-29 00:00"),
        "America/New_York",
    )
    assert leap.startswith("2024-02-29T05:00:00")
    year_end = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-12-31 23:59"),
        "UTC",
    )
    assert year_end.startswith("2026-12-31T23:59:00")
    with pytest.raises(ReminderValidationError):
        parse_local_wall_datetime("2026-01-15T09:00:00Z")
    with pytest.raises(ReminderValidationError):
        local_wall_to_utc_iso(
            datetime(2026, 1, 15, 9, 0, tzinfo=ZoneInfo("UTC")),
            "UTC",
        )


def test_deep_timezone_alias_us_eastern() -> None:
    """US/Eastern is a common alias; accept or reject consistently."""
    try:
        utc_iso = local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-08-14 09:00"),
            "US/Eastern",
        )
    except ReminderValidationError:
        return
    canonical = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-08-14 09:00"),
        "America/New_York",
    )
    assert utc_iso == canonical


def test_deep_clear_does_not_reset_active_memory() -> None:
    """Cleanup symmetry: /clear should also drop session active-memory selection."""
    history = ConversationHistory()
    history.add_user_message("hello")
    history.add_assistant_message("hi")
    active = ActiveMemoryContext()
    active.activate(create_memory("keep this active"))
    state = ConversationState()
    state.set_topic("vpn")
    speech = SpeechDeliveryState()
    retrieval = RetrievalSession()
    clear_conversation_history(
        history,
        retrieval_session=retrieval,
        conversation_state=state,
        speech_delivery_state=speech,
        active_memory_context=active,
    )
    assert history.turns == []
    assert state.is_empty
    assert active.active_count == 0


def test_deep_conversation_state_reset_matches_initialized_fields() -> None:
    initialized = set(ConversationState().__dataclass_fields__)
    reset_source = inspect.getsource(ConversationState.reset)
    missing = [
        name
        for name in initialized
        if name not in reset_source and name not in {"turn_window"}
    ]
    assert missing == [], f"reset() does not mention fields {missing}"


def test_deep_realtime_cleanup_nulls_worker_refs() -> None:
    from src.realtime_voice import RealtimeVoiceSession

    source = inspect.getsource(RealtimeVoiceSession._cleanup)
    for token in (
        "_playback_thread",
        "_session_thread",
        "_microphone",
        "_connection",
        "_assembler.reset",
    ):
        assert token in source, f"voice cleanup missing {token}"
    from src.realtime_multimodal import RealtimeMultimodalSession

    multi = inspect.getsource(RealtimeMultimodalSession._cleanup)
    for token in (
        "clear_visual_context_ref",
        "_visual_turns",
        "_camera",
        "_playback_thread",
    ):
        assert token in multi, f"multimodal cleanup missing {token}"


def test_deep_camera_stop_does_not_reset_sequence() -> None:
    from src.camera_capture import CameraCaptureSession

    source = inspect.getsource(CameraCaptureSession.stop)
    assert "_sequence" in source, (
        "CameraCaptureSession.stop() does not reset _sequence; reuse after "
        "stop can continue the previous capture sequence"
    )
    assert "_consecutive_failures" in source


def test_deep_user_facing_error_interpolation_is_bounded() -> None:
    """User-facing Cortana strings must not interpolate raw exception objects."""
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if 'f"Cortana: {error}"' in text or "f'Cortana: {error}'" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == [], f"raw exception interpolation in {offenders}"


def test_deep_conversation_loop_swallows_provider_exceptions() -> None:
    from src.conversation_loop import process_conversation_turn

    source = inspect.getsource(process_conversation_turn)
    assert "I could not complete that request" in source or "return None" in source
    assert "traceback" not in source.lower()


def test_deep_google_and_keyring_are_lazy_or_optional() -> None:
    """Optional calendar deps must not break core import/startup."""
    calendar_service = (PROJECT_ROOT / "src" / "calendar_service.py").read_text(
        encoding="utf-8"
    )
    secret_store = (PROJECT_ROOT / "src" / "secret_store.py").read_text(encoding="utf-8")
    main_text = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    header = calendar_service.split("def ", 1)[0]
    google_is_lazy = "google" not in header
    keyring_is_lazy = "import keyring" not in secret_store.split("def ", 1)[0]
    calendar_always_constructed = "create_default_calendar_service" in main_text
    assert google_is_lazy or not calendar_always_constructed, (
        "google calendar imports are module-level and main.py always constructs "
        "the calendar service, so a missing optional dep breaks startup"
    )
    assert keyring_is_lazy or not calendar_always_constructed, (
        "keyring is imported at module level and pulled in at startup"
    )


def test_deep_sounddevice_and_cv2_are_lazy() -> None:
    voice = (PROJECT_ROOT / "src" / "realtime_voice.py").read_text(encoding="utf-8")
    camera = (PROJECT_ROOT / "src" / "camera_capture.py").read_text(encoding="utf-8")
    tree = ast.parse(voice)
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module.split(".")[0])
    assert "sounddevice" not in top_imports
    assert "import cv2" not in camera.split("def ")[0]


def test_deep_no_unguarded_hardcoded_drive_letters_in_src() -> None:
    """Hardcoded C:\\ is acceptable only in Windows-guarded process helpers."""
    allowed = {
        "src/tool_process_common.py",
        "src/tool_process_safe_open.py",
        "src\\tool_process_common.py",
        "src\\tool_process_safe_open.py",
    }
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(PROJECT_ROOT))
        if ("C:\\" in text or "C:/" in text) and relative not in allowed:
            offenders.append(relative)
    assert offenders == [], f"unguarded hardcoded drive letters: {offenders}"
    common = (PROJECT_ROOT / "src" / "tool_process_common.py").read_text(encoding="utf-8")
    assert 'os.name == "nt"' in common
    safe_open = (PROJECT_ROOT / "src" / "tool_process_safe_open.py").read_text(
        encoding="utf-8"
    )
    assert "_require_win32_modules" in safe_open


def test_deep_windows_realtime_helpers_are_guarded() -> None:
    for relative in (
        "src/realtime_voice.py",
        "src/realtime_voice_input.py",
        "src/realtime_multimodal.py",
        "src/camera_capture.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "win32" in text or "sys.platform" in text, relative


def test_deep_evidence_copy_before_metadata_is_documented() -> None:
    """Transactional gap: evidence bytes are copied before JSON metadata is saved."""
    text = (PROJECT_ROOT / "src" / "security_commands.py").read_text(encoding="utf-8")
    copy_at = text.find("copy_from_path")
    add_at = text.find("add_evidence")
    assert copy_at != -1 and add_at != -1
    assert copy_at < add_at
    # Desired recovery: rollback the copied bytes if add_evidence fails.
    rollback = "unlink" in text[copy_at:add_at + 400] or "rollback" in text.lower()
    assert rollback, (
        "evidence register copies bytes before JSON metadata with no rollback "
        "(POSSIBLE_FALSE_SUCCESS / orphaned evidence file)"
    )


def test_deep_tool_run_persists_running_before_execute() -> None:
    text = (PROJECT_ROOT / "src" / "tool_commands.py").read_text(encoding="utf-8")
    running_at = text.find('request_status="running"')
    execute_at = text.find(".execute(")
    assert running_at != -1 and execute_at != -1
    assert running_at < execute_at
