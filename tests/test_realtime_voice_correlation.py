"""Permanent M25 response-generation and orphan-correlation regressions."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.config import PROJECT_ROOT
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.realtime_voice import (
    RealtimeSessionState,
    RealtimeVoiceSession,
    _TurnAssembler,
    build_session_update_payload,
)
from src.settings import Settings
from src.speech_delivery import SpeechDeliveryState


@dataclass
class FakeEvent:
    type: str
    item_id: str | None = None
    transcript: str | None = None
    response_id: str | None = None
    delta: str | None = None
    response: Any | None = None


@dataclass
class FakeResponse:
    id: object
    status: str = "completed"


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


def _voice_session(
    *,
    history: ConversationHistory | None = None,
) -> RealtimeVoiceSession:
    return RealtimeVoiceSession(
        settings=_settings(),
        client=cast(Any, object()),
        conversation_history=history or ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        conversation_state=ConversationState(),
        speech_delivery_state=SpeechDeliveryState(),
        print_fn=lambda _line: None,
    )


def _commit(session: RealtimeVoiceSession, item_id: str | None) -> None:
    session._on_user_audio_committed(
        FakeEvent(type="input_audio_buffer.committed", item_id=item_id)
    )


def _speech(session: RealtimeVoiceSession, item_id: str) -> None:
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id=item_id)
    )


def _created(session: RealtimeVoiceSession, response_id: object) -> None:
    session._on_response_created(
        FakeEvent(type="response.created", response=FakeResponse(id=response_id))
    )


def _done(
    session: RealtimeVoiceSession,
    response_id: object,
    *,
    status: str = "completed",
) -> None:
    session._on_response_done(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id=response_id, status=status),
        )
    )


def _delta(session: RealtimeVoiceSession, response_id: str) -> None:
    pcm = base64.b64encode(b"\x02\x00" * 20).decode("ascii")
    session._on_audio_delta(
        FakeEvent(
            type="response.output_audio.delta",
            response_id=response_id,
            delta=pcm,
        )
    )


def _transcript(
    session: RealtimeVoiceSession,
    item_id: str,
    text: str,
) -> None:
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            transcript=text,
        )
    )


def _assistant(session: RealtimeVoiceSession, response_id: str, text: str) -> None:
    session._on_assistant_transcript(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id=response_id,
            transcript=text,
        )
    )


def _pairing(session: RealtimeVoiceSession) -> dict[str, str]:
    return dict(session._assembler._response_user_item)


def test_normal_single_turn_commits_history_pair() -> None:
    history = ConversationHistory()
    session = _voice_session(history=history)
    _commit(session, "item_a")
    _created(session, "resp_a")
    _transcript(session, "item_a", "hello cortana")
    _assistant(session, "resp_a", "hello human")
    _done(session, "resp_a")
    assert session._active_response_id is None
    assert not session._is_cancelled("resp_a")
    assert history.turns[-2].content == "hello cortana"
    assert history.turns[-1].content == "hello human"
    assert len(history.turns) == 2


def test_created_after_commit_before_transcript_still_pairs() -> None:
    history = ConversationHistory()
    session = _voice_session(history=history)
    _commit(session, "item_a")
    _created(session, "resp_a")
    assert session._active_response_id == "resp_a"
    assert _pairing(session).get("resp_a") == "item_a"
    _transcript(session, "item_a", "status please")
    _assistant(session, "resp_a", "all clear")
    _done(session, "resp_a")
    assert history.turns[-2].content == "status please"
    assert history.turns[-1].content == "all clear"


def test_stale_a_before_b_is_cancelled_and_b_remains_eligible() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _created(session, "resp_a")
    _created(session, "resp_b")
    assert session._active_response_id == "resp_b"
    assert session._is_cancelled("resp_a")
    assert not session._is_cancelled("resp_b")
    pairing = _pairing(session)
    assert pairing.get("resp_b") == "item_b"
    assert pairing.get("resp_a") != "item_b"


def test_stale_a_after_b_is_rejected_once_b_has_output() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _created(session, "resp_b")
    _delta(session, "resp_b")
    _created(session, "resp_a")
    assert session._active_response_id == "resp_b"
    assert session._is_cancelled("resp_a")
    pairing = _pairing(session)
    assert pairing.get("resp_b") == "item_b"
    assert pairing.get("resp_a") != "item_b"


def test_missing_stale_a_does_not_cancel_b() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _created(session, "resp_b")
    assert session._active_response_id == "resp_b"
    assert not session._is_cancelled("resp_b")
    assert _pairing(session).get("resp_b") == "item_b"


def test_duplicate_response_created_does_not_rebind() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _created(session, "resp_1")
    first_active = session._active_response_id
    _created(session, "resp_1")
    assert session._active_response_id == first_active
    assert list(_pairing(session).values()).count("item_a") <= 1


def test_orphan_created_with_no_user_is_tombstoned() -> None:
    session = _voice_session()
    _created(session, "resp_orphan")
    assert session._active_response_id is None
    assert session._is_cancelled("resp_orphan") or "resp_orphan" in session._tombstoned_set
    assert "resp_orphan" not in _pairing(session)


def test_orphan_does_not_bind_to_future_user_item() -> None:
    session = _voice_session()
    _created(session, "resp_orphan")
    _commit(session, "item_later")
    assert _pairing(session).get("resp_orphan") != "item_later"
    _created(session, "resp_later")
    pairing = _pairing(session)
    assert pairing.get("resp_later") == "item_later"
    assert pairing.get("resp_orphan") != "item_later"
    assert session._active_response_id == "resp_later"


def test_repeated_barge_ins_do_not_permanently_silence() -> None:
    session = _voice_session()
    for index in range(6):
        _commit(session, f"item_{index}")
        _speech(session, f"item_{index + 1}")
    _commit(session, "item_final")
    _created(session, "resp_live")
    assert session._active_response_id == "resp_live"
    assert not session._is_cancelled("resp_live")
    assert _pairing(session).get("resp_live") == "item_final"


def test_malformed_response_created_id_is_ignored() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _created(session, None)
    _created(session, 123)
    _created(session, "")
    assert session._active_response_id is None
    _created(session, "resp_ok")
    assert session._active_response_id == "resp_ok"
    assert _pairing(session).get("resp_ok") == "item_a"


def test_response_done_before_created_does_not_steal_later_live() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _done(session, "resp_early")
    assert session._active_response_id is None
    assert "resp_early" not in _pairing(session)
    _created(session, "resp_a")
    assert session._active_response_id == "resp_a"
    assert _pairing(session).get("resp_a") == "item_a"


def test_response_done_for_unknown_id_leaves_live_response() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _created(session, "resp_ok")
    _done(session, "resp_unknown")
    assert session._active_response_id == "resp_ok"
    assert _pairing(session).get("resp_ok") == "item_a"


def test_created_after_cleanup_is_tombstoned() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    session._cleanup()
    _created(session, "resp_late")
    assert session._active_response_id is None
    assert session._is_cancelled("resp_late") or "resp_late" in session._tombstoned_set
    assert "resp_late" not in _pairing(session)
    assert session._expected_generation is None
    assert session._generations == {}


def test_created_after_session_ended_is_tombstoned() -> None:
    session = _voice_session()
    session._set_state(RealtimeSessionState.CLOSED)
    session._state_writes_closed.set()
    _created(session, "resp_ended")
    assert session._active_response_id is None
    assert session._is_cancelled("resp_ended") or "resp_ended" in session._tombstoned_set


def test_hundred_deterministic_correlation_sequences() -> None:
    mispairs: list[str] = []
    map_growth: list[int] = []
    for index in range(100):
        session = _voice_session()
        kind = index % 10
        extra = index // 10
        if kind == 0:
            _commit(session, "item_a")
            _speech(session, "item_b")
            _commit(session, "item_b")
            _created(session, "resp_b")
            expected = ("resp_b", "item_b")
        elif kind == 1:
            _commit(session, "item_a")
            _speech(session, "item_b")
            _commit(session, "item_b")
            _created(session, "resp_stale")
            _created(session, "resp_b")
            expected = ("resp_b", "item_b")
        elif kind == 2:
            _commit(session, "item_a")
            _speech(session, "item_b")
            _commit(session, "item_b")
            _created(session, "resp_b")
            _delta(session, "resp_b")
            _created(session, "resp_stale")
            expected = ("resp_b", "item_b")
        elif kind == 3:
            _commit(session, "item_a")
            last_item = "item_a"
            for label in ("b", "c", "d")[: 1 + (extra % 3)]:
                last_item = f"item_{label}"
                _speech(session, last_item)
                _commit(session, last_item)
            _created(session, "resp_final")
            expected = ("resp_final", last_item)
        elif kind == 4:
            _commit(session, "item_a")
            _created(session, "resp_1")
            _created(session, "resp_1")
            expected = ("resp_1", "item_a")
        elif kind == 5:
            _commit(session, "item_a")
            _done(session, "resp_early")
            _created(session, "resp_a")
            expected = ("resp_a", "item_a")
        elif kind == 6:
            _commit(session, "item_a")
            _created(session, None)
            _created(session, "resp_a")
            expected = ("resp_a", "item_a")
        elif kind == 7:
            _commit(session, "item_a")
            _created(session, "resp_a")
            _done(session, "resp_unknown")
            expected = ("resp_a", "item_a")
        elif kind == 8:
            _created(session, "resp_orphan")
            _commit(session, "item_a")
            expected = None
        else:
            _commit(session, "item_a")
            _speech(session, "item_b")
            _created(session, "")
            _commit(session, "item_b")
            _created(session, "resp_b")
            expected = ("resp_b", "item_b")

        pairing = _pairing(session)
        map_growth.append(
            len(pairing)
            + len(session._cancelled_set)
            + len(session._tombstoned_set)
            + len(session._generations)
        )
        if expected is not None:
            response_id, user_item = expected
            bound = pairing.get(response_id)
            if bound not in {user_item, None}:
                mispairs.append(
                    f"index={index} kind={kind} bound {response_id} -> {bound!r}"
                )
            if kind == 1 and pairing.get("resp_stale") == "item_b":
                mispairs.append(f"index={index} stale bound to live item_b")
            if session._is_cancelled(response_id):
                mispairs.append(f"index={index} live response cancelled")
        if kind == 8 and pairing.get("resp_orphan") == "item_a":
            mispairs.append(f"index={index} orphan bound to later user")
        if session._active_response_id is None and kind not in {8}:
            if expected is not None and not session._is_cancelled(expected[0]):
                if pairing.get(expected[0]) is None and kind not in {5, 6}:
                    pass
    assert mispairs == [], "wrong pairing:\n" + "\n".join(mispairs)
    assert all(size <= 48 for size in map_growth)


def test_hundred_session_cleanup_cycles_clear_generation_state() -> None:
    shared = ConversationState()
    shared.set_topic("shared topic")
    for index in range(100):
        session = RealtimeVoiceSession(
            settings=_settings(),
            client=cast(Any, object()),
            conversation_history=ConversationHistory(),
            active_memory_context=ActiveMemoryContext(),
            conversation_state=shared,
            speech_delivery_state=SpeechDeliveryState(),
            print_fn=lambda _line: None,
        )
        _commit(session, f"item_{index}")
        _created(session, f"resp_{index}")
        session._cleanup()
        assert session._expected_generation is None
        assert session._generations == {}
        assert len(session._invalidated_unclaimed) == 0
        assert session._provisional_response_id is None
        assert session._claimed_response_ids == {}
        assert len(session._tombstoned_response_ids) == 0
        assert session._active_response_id is None
        assert session._cancelled_set == set()
        assert all(
            size == 0
            for size in (
                len(session._assembler._pending_user),
                len(session._assembler._response_user_item),
                len(session._assembler._unbound_response_ids),
            )
        )
    assert shared.current_topic == "shared topic"


def test_assembler_does_not_bind_orphan_created_to_later_user() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.bind_response("resp_orphan")
    assembler.set_current_user_item("item_later")
    assert assembler._response_user_item.get("resp_orphan") != "item_later"
    assembler.bind_response("resp_later")
    assembler.store_user_transcript("item_later", "later user")
    assembler.store_assistant_transcript("resp_later", "later assistant")
    result = assembler.on_response_done(response_id="resp_later", status="completed")
    assert result.outcome == "pair"
    assert result.user_item_id == "item_later"
    assert history.turns[-2].content == "later user"


def test_provider_ownership_invariants_unchanged() -> None:
    payload = build_session_update_payload(settings=_settings(), instructions="test")
    audio = payload["audio"]
    assert isinstance(audio, dict)
    turn = audio["input"]["turn_detection"]  # type: ignore[index]
    assert isinstance(turn, dict)
    assert turn["type"] == "server_vad"
    assert turn["create_response"] is True
    assert turn["interrupt_response"] is True
    source = (PROJECT_ROOT / "src/realtime_voice.py").read_text(encoding="utf-8")
    assert "connection.response.create" not in source
    assert "response.create(" not in source
