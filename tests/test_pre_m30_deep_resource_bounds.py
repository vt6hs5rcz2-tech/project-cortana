"""Pre-M30 Deep Audit #4: resource-bound / M26 visual-turn deep stress.

Discovery only. Failures document soft caps and unbounded in-flight retention.
"""

from __future__ import annotations

from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.camera_capture import RealtimeVisualFrame
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.realtime_multimodal import (
    RealtimeMultimodalSession,
    _MAX_VISUAL_TURNS,
    _VisualTurnState,
)
from src.settings import Settings
from src.speech_delivery import SpeechDeliveryState


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        transcription_model="gpt-4o-mini-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="coral",
        realtime_model="gpt-realtime-mini",
        realtime_voice="coral",
    )


def _multimodal_session(
    *,
    state: ConversationState | None = None,
) -> RealtimeMultimodalSession:
    return RealtimeMultimodalSession(
        settings=_settings(),
        client=cast(Any, object()),
        conversation_history=ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        conversation_state=state,
        print_fn=lambda _line: None,
        transcript_wait_seconds=0.25,
    )


def _frame(sequence: int, payload_bytes: int = 8192) -> RealtimeVisualFrame:
    return RealtimeVisualFrame(
        image_bytes=b"P" * payload_bytes,
        mime_type="image/png",
        width=64,
        height=64,
        sequence=sequence,
        captured_at_monotonic=float(sequence),
    )


def _awaiting_turn(index: int, *, payload_bytes: int = 8192) -> _VisualTurnState:
    return _VisualTurnState(
        user_item_id=f"user_{index}",
        visual_frame=_frame(index, payload_bytes),
        awaiting_remote_id=True,
        response_create_sent=True,
        stale=False,
        delete_sent=False,
        ack_deadline_monotonic=0.0,
    )


def _retained_frame_bytes(session: RealtimeMultimodalSession) -> int:
    total = 0
    for turn in session._visual_turns.values():
        if turn.visual_frame is not None:
            total += len(turn.visual_frame.image_bytes)
    return total


def test_deep_f4_declared_visual_turn_cap_is_sixteen() -> None:
    assert _MAX_VISUAL_TURNS == 16


def test_deep_f4_one_hundred_non_ack_turns_exceed_soft_cap() -> None:
    """OUTSIDE-F4: non-releasable awaiting_remote_id turns are not hard-capped at 16."""
    session = _multimodal_session()
    for index in range(100):
        session._visual_turns[f"user_{index}"] = _awaiting_turn(index)
    session._compact_completed_visual_turns()
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 8192


def test_deep_f4_five_hundred_cheap_non_ack_turns_remain_unbounded() -> None:
    """Worst-case count: 500 in-flight non-stale turns with no remote ack."""
    session = _multimodal_session()
    for index in range(500):
        session._visual_turns[f"user_{index}"] = _VisualTurnState(
            user_item_id=f"user_{index}",
            visual_frame=None,
            awaiting_remote_id=True,
            response_create_sent=True,
        )
    session._compact_completed_visual_turns()
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS


def test_deep_f4_stale_turns_are_hard_capped() -> None:
    """Control: stale/delete_sent turns compact to the declared cap."""
    session = _multimodal_session()
    for index in range(64):
        session._visual_turns[f"user_{index}"] = _VisualTurnState(
            user_item_id=f"user_{index}",
            visual_frame=_frame(index),
            stale=True,
            awaiting_remote_id=True,
        )
    session._compact_completed_visual_turns()
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    assert _retained_frame_bytes(session) == 0


def test_deep_f4_mixed_ack_and_non_ack_retains_live_turns() -> None:
    session = _multimodal_session()
    for index in range(40):
        turn = _awaiting_turn(index)
        if index % 2 == 0:
            turn.stale = True
        session._visual_turns[f"user_{index}"] = turn
    session._compact_completed_visual_turns()
    live = [turn for turn in session._visual_turns.values() if not turn.stale]
    assert len(live) <= _MAX_VISUAL_TURNS
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS


def test_deep_f4_timeout_does_not_stale_awaiting_remote_id_turns() -> None:
    """Transcript-wait expiry must not leave awaiting turns pinned forever."""
    session = _multimodal_session()
    session._visual_turns["user_late"] = _awaiting_turn(0)
    session._transcript_deadlines["user_late"] = 0.0
    if hasattr(session, "_expire_transcript_waits"):
        session._expire_transcript_waits()
    if hasattr(session, "_expire_visual_ack_waits"):
        session._expire_visual_ack_waits()
    session._compact_completed_visual_turns()
    turn = session._visual_turns.get("user_late")
    assert turn is None or turn.stale is True or turn.awaiting_remote_id is False


def test_deep_f4_cleanup_releases_all_visual_memory() -> None:
    """Control: session shutdown must drop turns, frames, acks, and orphan debt."""
    session = _multimodal_session(state=ConversationState())
    for index in range(40):
        session._visual_turns[f"user_{index}"] = _awaiting_turn(index, payload_bytes=4096)
        session._live_remote_visual_ids.add(f"remote_{index}")
    session._orphan_visual_ack_debt = 99
    session._pending_visual_acks.append("ack_x")
    session._cleanup()
    assert session._visual_turns == {}
    assert _retained_frame_bytes(session) == 0
    assert session._orphan_visual_ack_debt == 0
    assert len(session._pending_visual_acks) == 0
    assert len(session._live_remote_visual_ids) == 0


def test_deep_f4_long_session_without_shutdown_is_vulnerable() -> None:
    """Without cleanup, in-flight visual frames remain until the session ends."""
    session = _multimodal_session()
    for index in range(32):
        session._visual_turns[f"user_{index}"] = _awaiting_turn(index, payload_bytes=65536)
    session._compact_completed_visual_turns()
    retained = _retained_frame_bytes(session)
    assert retained <= _MAX_VISUAL_TURNS * 65536
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS


def test_deep_speech_pending_chunks_are_hard_capped() -> None:
    """Control: speech pending chunks reject rather than grow without bound."""
    from src.speech_delivery import MAX_PENDING_SPEECH_CHUNKS

    state = SpeechDeliveryState()
    overflow = [f"chunk {index}" for index in range(MAX_PENDING_SPEECH_CHUNKS + 32)]
    state.load_pending(overflow)
    assert len(state.pending_chunks) <= MAX_PENDING_SPEECH_CHUNKS
    assert state.pending_rejected is True or len(state.pending_chunks) <= MAX_PENDING_SPEECH_CHUNKS
