"""M26 visual-turn hard bounds, ack timeout, and late-ack safety."""

from __future__ import annotations

import time
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.camera_capture import RealtimeVisualFrame
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.realtime_multimodal import (
    MultimodalOutboundAction,
    MultimodalOutboundKind,
    RealtimeMultimodalSession,
    _MAX_DELETED_REMOTE_VISUAL_IDS,
    _MAX_PENDING_VISUAL_ACKS,
    _MAX_VISUAL_TURNS,
    _VisualTurnState,
)
from src.settings import Settings
from tests.test_realtime_multimodal import FakeConnection, FakeEvent, FakeItem


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


class _FakeClock:
    """Deterministic monotonic clock for visual-ack timeout tests."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _session(
    *,
    state: ConversationState | None = None,
    clock: _FakeClock | None = None,
) -> RealtimeMultimodalSession:
    return RealtimeMultimodalSession(
        settings=_settings(),
        client=cast(Any, object()),
        conversation_history=ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        conversation_state=state,
        print_fn=lambda _line: None,
        monotonic_fn=clock if clock is not None else time.monotonic,
        transcript_wait_seconds=0.25,
        visual_ack_wait_seconds=0.25,
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


def _awaiting(
    index: int,
    *,
    payload_bytes: int = 8192,
    deadline: float = 0.0,
) -> _VisualTurnState:
    return _VisualTurnState(
        user_item_id=f"user_{index}",
        visual_frame=_frame(index, payload_bytes),
        awaiting_remote_id=True,
        response_create_sent=True,
        ack_deadline_monotonic=deadline,
    )


def _retained_frame_bytes(session: RealtimeMultimodalSession) -> int:
    total = 0
    for turn in session._visual_turns.values():
        if turn.visual_frame is not None:
            total += len(turn.visual_frame.image_bytes)
    return total


def _maintain(session: RealtimeMultimodalSession) -> None:
    session._expire_visual_ack_waits()
    session._compact_completed_visual_turns()


def test_hundred_non_ack_turns_are_hard_capped() -> None:
    session = _session()
    for index in range(100):
        session._visual_turns[f"user_{index}"] = _awaiting(index)
    _maintain(session)
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 8192


def test_five_hundred_cheap_non_ack_turns_are_hard_capped() -> None:
    session = _session()
    for index in range(500):
        session._visual_turns[f"user_{index}"] = _VisualTurnState(
            user_item_id=f"user_{index}",
            visual_frame=None,
            awaiting_remote_id=True,
            response_create_sent=True,
        )
    _maintain(session)
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS


def test_hundred_8kib_frames_bound_payload_bytes() -> None:
    session = _session()
    for index in range(100):
        session._visual_turns[f"user_{index}"] = _awaiting(index, payload_bytes=8192)
    _maintain(session)
    assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 8192
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS


def test_mixed_ack_missing_late_and_stale_stays_bounded() -> None:
    session = _session()
    for index in range(40):
        turn = _awaiting(index)
        if index % 4 == 0:
            turn.stale = True
        if index % 4 == 1:
            turn.awaiting_remote_id = False
            turn.remote_visual_item_id = f"visual_{index}"
            turn.delete_sent = True
            turn.visual_frame = None
        session._visual_turns[f"user_{index}"] = turn
        if index % 4 == 2:
            session._queue_visual_ack(f"user_{index}")
    _maintain(session)
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 8192
    assert len(session._pending_visual_acks) <= _MAX_PENDING_VISUAL_ACKS


def test_late_ack_after_expiry_does_not_bind_newer_turn() -> None:
    session = _session()
    session._visual_turns["user_a"] = _awaiting(0, deadline=0.0)
    session._queue_visual_ack("user_a")
    _maintain(session)
    session._visual_turns["user_b"] = _VisualTurnState(
        user_item_id="user_b",
        visual_frame=_frame(1),
        awaiting_remote_id=True,
        bound_at_monotonic=100.0,
        ack_deadline_monotonic=session._monotonic() + 1_000.0,
    )
    session._queue_visual_ack("user_b")
    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_a_late"))
    )
    live = session._visual_turns.get("user_b")
    assert live is not None
    assert live.remote_visual_item_id != "visual_a_late"
    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_b"))
    )
    assert session._visual_turns["user_b"].remote_visual_item_id == "visual_b"


def test_visual_ack_timeout_does_not_create_response() -> None:
    session = _session()
    session._visual_turns["user_late"] = _awaiting(0, deadline=0.0)
    session._expire_visual_ack_waits()
    session._compact_completed_visual_turns()
    assert session._outbound.empty()
    turn = session._visual_turns.get("user_late")
    assert turn is None or turn.stale is True


def test_speech_started_releases_old_awaiting_frame() -> None:
    session = _session()
    session._visual_turns["user_old"] = _awaiting(0)
    session._visual_turns["user_old"].response_create_sent = False
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="user_new")
    )
    old = session._visual_turns.get("user_old")
    assert old is None or old.visual_frame is None
    assert _retained_frame_bytes(session) == 0


def test_continuous_maintenance_without_shutdown_stays_bounded() -> None:
    session = _session()
    for index in range(80):
        session._visual_turns[f"user_{index}"] = _awaiting(
            index,
            payload_bytes=65536,
            deadline=0.0 if index < 60 else 10_000.0,
        )
        _maintain(session)
        assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
        assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 65536
        assert len(session._pending_visual_acks) <= _MAX_PENDING_VISUAL_ACKS
    later = _VisualTurnState(
        user_item_id="user_live",
        visual_frame=_frame(99, 1024),
        awaiting_remote_id=True,
        bound_at_monotonic=session._monotonic(),
        ack_deadline_monotonic=session._monotonic() + 1_000.0,
    )
    session._visual_turns["user_live"] = later
    session._queue_visual_ack("user_live")
    _maintain(session)
    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_live"))
    )
    assert session._visual_turns["user_live"].remote_visual_item_id == "visual_live"


def test_cleanup_releases_visual_memory_and_ack_state() -> None:
    state = ConversationState()
    session = _session(state=state)
    state.set_visual_context_ref("pending:user_0")
    for index in range(20):
        session._visual_turns[f"user_{index}"] = _awaiting(index, payload_bytes=4096)
        session._live_remote_visual_ids.add(f"remote_{index}")
        session._queue_visual_ack(f"user_{index}")
    session._orphan_visual_ack_debt = 4
    session._cleanup()
    assert session._visual_turns == {}
    assert _retained_frame_bytes(session) == 0
    assert session._orphan_visual_ack_debt == 0
    assert len(session._pending_visual_acks) == 0
    assert session._live_remote_visual_ids == set()
    assert session._current_remote_visual_item_id is None
    assert len(session._deleted_remote_visual_ids) == 0
    assert state.visual_context_ref_id is None


def test_duplicate_late_remote_id_does_not_rebind() -> None:
    session = _session()
    session._visual_turns["user_a"] = _VisualTurnState(
        user_item_id="user_a",
        visual_frame=None,
        remote_visual_item_id="visual_keep",
        awaiting_remote_id=False,
        delete_sent=True,
    )
    session._forget_remote_visual_id("visual_keep")
    session._visual_turns["user_b"] = _VisualTurnState(
        user_item_id="user_b",
        visual_frame=_frame(2),
        awaiting_remote_id=True,
        ack_deadline_monotonic=session._monotonic() + 1_000.0,
    )
    session._queue_visual_ack("user_b")
    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_keep"))
    )
    assert session._visual_turns["user_b"].remote_visual_item_id != "visual_keep"


def _prepare_visual_turn(
    session: RealtimeMultimodalSession,
    connection: FakeConnection,
    user_item_id: str,
    frame: RealtimeVisualFrame,
) -> _VisualTurnState:
    turn = _VisualTurnState(
        user_item_id=user_item_id,
        visual_frame=frame,
        bound_at_monotonic=session._monotonic(),
        visual_insert_authorized=True,
    )
    session._visual_turns[user_item_id] = turn
    session._prepare_turn_response(
        connection,
        MultimodalOutboundAction(
            kind=MultimodalOutboundKind.PREPARE_TURN_RESPONSE,
            user_item_id=user_item_id,
            visual_frame=frame,
        ),
    )
    return turn


def test_visual_ack_timeout_issues_one_create_releases_and_stays_bounded() -> None:
    """Controlled withhold: timeout must not deadlock and must create once."""
    clock = _FakeClock()
    state = ConversationState()
    connection = FakeConnection(auto_ack_visual=False, auto_ack_responses=False)
    session = _session(state=state, clock=clock)
    session._connection = connection

    turn = _prepare_visual_turn(session, connection, "user_a", _frame(1))
    assert connection.conversation.item.created_items
    assert connection.response.response_creates == 0
    assert turn.awaiting_remote_id is True
    assert turn.ack_deadline_monotonic == clock.now + 0.25
    assert list(session._pending_visual_acks) == ["user_a"]
    assert turn.visual_frame is not None

    clock.advance(0.25)
    session._expire_visual_ack_waits()
    assert connection.response.response_creates == 0
    session._drain_outbound(connection)

    assert connection.response.response_creates == 1
    assert turn.awaiting_remote_id is False
    assert turn.ack_deadline_monotonic is None
    assert turn.response_create_sent is True
    assert turn.stale is False
    assert turn.remote_visual_item_id is None
    assert session._current_remote_visual_item_id is None
    assert state.visual_context_ref_id is None
    session._expire_visual_ack_waits()
    session._drain_outbound(connection)
    assert connection.response.response_creates == 1
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    assert len(session._pending_visual_acks) <= _MAX_PENDING_VISUAL_ACKS
    assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 8192


def test_late_ack_after_timeout_does_not_bind_or_create_again() -> None:
    """Late ack for a timed-out turn must not become current visual context."""
    clock = _FakeClock()
    state = ConversationState()
    connection = FakeConnection(auto_ack_visual=False, auto_ack_responses=False)
    session = _session(state=state, clock=clock)
    session._connection = connection

    turn = _prepare_visual_turn(session, connection, "user_a", _frame(1))
    clock.advance(0.25)
    session._expire_visual_ack_waits()
    session._drain_outbound(connection)
    assert connection.response.response_creates == 1

    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_a_late"))
    )
    live = session._visual_turns.get("user_a")
    if live is not None:
        assert live.remote_visual_item_id != "visual_a_late"
    assert session._current_remote_visual_item_id != "visual_a_late"
    assert state.visual_context_ref_id != "visual_a_late"
    assert "visual_a_late" not in session._live_remote_visual_ids
    assert connection.response.response_creates == 1
    assert "visual_a_late" in connection.conversation.item.deleted_ids
    assert "visual_a_late" in session._deleted_remote_visual_ids
    assert turn.visual_frame is None or live is None
    assert len(session._pending_visual_acks) <= _MAX_PENDING_VISUAL_ACKS
    assert len(session._deleted_remote_visual_ids) <= _MAX_DELETED_REMOTE_VISUAL_IDS
    assert session._orphan_visual_ack_debt >= 0


def test_late_ack_after_timeout_does_not_bind_newer_follow_up_turn() -> None:
    """Critical correlation: timed-out ack debt must not poison the next turn."""
    clock = _FakeClock()
    state = ConversationState()
    connection = FakeConnection(auto_ack_visual=False, auto_ack_responses=False)
    session = _session(state=state, clock=clock)
    session._connection = connection

    _prepare_visual_turn(session, connection, "user_a", _frame(1, payload_bytes=4096))
    clock.advance(0.25)
    session._expire_visual_ack_waits()
    session._drain_outbound(connection)
    assert connection.response.response_creates == 1

    follow = _prepare_visual_turn(
        session, connection, "user_b", _frame(2, payload_bytes=2048)
    )
    assert follow.awaiting_remote_id is True
    assert follow.remote_visual_item_id is None
    assert list(session._pending_visual_acks)[-1] == "user_b"

    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_a_late"))
    )
    assert follow.remote_visual_item_id != "visual_a_late"
    assert session._current_remote_visual_item_id != "visual_a_late"
    assert state.visual_context_ref_id != "visual_a_late"
    assert connection.response.response_creates == 1
    assert "visual_a_late" in connection.conversation.item.deleted_ids

    session._on_conversation_item_ack(
        FakeEvent(type="conversation.item.added", item=FakeItem(id="visual_b"))
    )
    session._drain_outbound(connection)
    assert follow.remote_visual_item_id == "visual_b"
    assert session._current_remote_visual_item_id == "visual_b"
    assert state.visual_context_ref_id == "visual_b"
    assert connection.response.response_creates == 2
    first = session._visual_turns.get("user_a")
    if first is not None:
        assert first.remote_visual_item_id != "visual_b"
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    assert len(session._pending_visual_acks) <= _MAX_PENDING_VISUAL_ACKS
    assert _retained_frame_bytes(session) <= _MAX_VISUAL_TURNS * 8192
    session._cleanup()
    assert session._visual_turns == {}
    assert session._pending_visual_acks.__class__() == session._pending_visual_acks
    assert len(session._pending_visual_acks) == 0
    assert session._orphan_visual_ack_debt == 0
    assert _retained_frame_bytes(session) == 0
    assert len(session._deleted_remote_visual_ids) == 0
