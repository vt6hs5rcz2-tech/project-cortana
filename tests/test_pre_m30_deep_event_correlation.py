"""Pre-M30 Deep Audit #4: M25 response-correlation deep stress.

Discovery only. Deterministic sequences, not random fuzz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.realtime_voice import RealtimeVoiceSession
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
    item: Any | None = None


@dataclass
class FakeResponse:
    id: object
    status: str = "completed"
    metadata: dict[str, str] | None = None


_AUTO_METADATA = object()


def _metadata_for(
    session: RealtimeVoiceSession,
    item_id: str | None,
) -> dict[str, str] | None:
    record = None
    if item_id is not None:
        record = session._generation_for_user_item(item_id)
    elif session._expected_generation is not None:
        record = session._generations.get(session._expected_generation)
    if record is None or not record.user_item_id:
        return None
    return {
        "cortana_user_item_id": record.user_item_id,
        "cortana_generation": str(record.generation),
    }


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


def _voice_session() -> RealtimeVoiceSession:
    return RealtimeVoiceSession(
        settings=_settings(),
        client=cast(Any, object()),
        conversation_history=ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        conversation_state=ConversationState(),
        speech_delivery_state=SpeechDeliveryState(),
        print_fn=lambda _line: None,
    )


def _commit(session: RealtimeVoiceSession, item_id: str) -> None:
    session._on_user_audio_committed(FakeEvent(type="input_audio_buffer.committed", item_id=item_id))


def _speech(session: RealtimeVoiceSession, item_id: str) -> None:
    session._on_speech_started(FakeEvent(type="input_audio_buffer.speech_started", item_id=item_id))


def _created(
    session: RealtimeVoiceSession,
    response_id: object,
    *,
    item_id: str | None = None,
    metadata: object = _AUTO_METADATA,
) -> None:
    resolved = (
        _metadata_for(session, item_id) if metadata is _AUTO_METADATA else metadata
    )
    meta = resolved if isinstance(resolved, dict) or resolved is None else None
    session._on_response_created(
        FakeEvent(
            type="response.created",
            response=FakeResponse(id=response_id, metadata=meta),
        )
    )


def _done(session: RealtimeVoiceSession, response_id: object, *, status: str = "completed") -> None:
    session._on_response_done(
        FakeEvent(type="response.done", response=FakeResponse(id=response_id, status=status))
    )


def _pairing(session: RealtimeVoiceSession) -> dict[str, str]:
    return dict(session._assembler._response_user_item)


def test_deep_m25_stale_a_never_arrives_b_is_accepted() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _created(session, "resp_b")
    assert session._active_response_id == "resp_b"
    assert "resp_b" not in session._cancelled_set
    pairing = _pairing(session)
    assert pairing.get("resp_b") == "item_b"


def test_deep_m25_stale_a_arrives_before_b() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _created(session, "resp_a", item_id="item_a")
    _created(session, "resp_b", item_id="item_b")
    pairing = _pairing(session)
    assert session._active_response_id == "resp_b"
    assert "resp_a" in session._cancelled_set
    assert pairing.get("resp_a") != "item_b"
    assert pairing.get("resp_b") == "item_b"


def test_deep_m25_stale_a_arrives_after_b() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _created(session, "resp_b", item_id="item_b")
    session._on_audio_delta(
        FakeEvent(
            type="response.output_audio.delta",
            response_id="resp_b",
            delta="AAEC",
        )
    )
    _done(session, "resp_b")
    _created(session, "resp_a", item_id="item_a")
    assert "resp_a" in session._cancelled_set
    assert "resp_b" not in session._cancelled_set
    pairing = _pairing(session)
    assert pairing.get("resp_b") == "item_b"
    assert pairing.get("resp_a") != "item_b"


def test_deep_m25_triple_barge_in_before_any_created() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _speech(session, "item_b")
    _commit(session, "item_b")
    _speech(session, "item_c")
    _commit(session, "item_c")
    _created(session, "resp_c")
    assert session._active_response_id == "resp_c"
    pairing = _pairing(session)
    assert pairing.get("resp_c") == "item_c"


def test_deep_m25_duplicate_response_created() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _created(session, "resp_1")
    first_active = session._active_response_id
    _created(session, "resp_1")
    assert session._active_response_id == first_active
    pairing = _pairing(session)
    assert list(pairing.values()).count("item_a") <= 1


def test_deep_m25_done_before_created() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _done(session, "resp_early")
    assert session._active_response_id is None
    pairing = _pairing(session)
    assert "resp_early" not in pairing


def test_deep_m25_created_with_malformed_id() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _created(session, None)
    _created(session, 123)
    _created(session, "")
    assert session._active_response_id is None
    _created(session, "resp_ok")
    assert session._active_response_id == "resp_ok"


def test_deep_m25_done_for_unknown_id() -> None:
    session = _voice_session()
    _commit(session, "item_a")
    _created(session, "resp_ok")
    _done(session, "resp_unknown")
    assert session._active_response_id == "resp_ok"
    pairing = _pairing(session)
    assert pairing.get("resp_ok") == "item_a"


def test_deep_m25_extra_created_with_no_user_does_not_pair_later_unrelated() -> None:
    session = _voice_session()
    _created(session, "resp_orphan")
    _commit(session, "item_later")
    pairing = _pairing(session)
    assert pairing.get("resp_orphan") != "item_later"
    assert "resp_orphan" in session._cancelled_set or "resp_orphan" in session._tombstoned_set
    _created(session, "resp_later")
    pairing = _pairing(session)
    assert pairing.get("resp_later") == "item_later"
    assert pairing.get("resp_orphan") != "item_later"


def test_deep_m25_hundred_deterministic_sequence_variations() -> None:
    """100 deterministic provider-order variations; fail on wrong user pairing."""
    mispairs: list[str] = []
    map_growth: list[int] = []
    for index in range(100):
        session = _voice_session()
        kind = index % 10
        extra = index // 10
        try:
            if kind == 0:
                _commit(session, "item_a")
                _speech(session, "item_b")
                _commit(session, "item_b")
                _created(session, "resp_b", item_id="item_b")
                expected = ("resp_b", "item_b")
            elif kind == 1:
                _commit(session, "item_a")
                _speech(session, "item_b")
                _commit(session, "item_b")
                _created(session, "resp_stale", item_id="item_a")
                _created(session, "resp_b", item_id="item_b")
                expected = ("resp_b", "item_b")
            elif kind == 2:
                _commit(session, "item_a")
                _speech(session, "item_b")
                _commit(session, "item_b")
                _created(session, "resp_b", item_id="item_b")
                session._on_audio_delta(
                    FakeEvent(
                        type="response.output_audio.delta",
                        response_id="resp_b",
                        delta="AAEC",
                    )
                )
                _done(session, "resp_b")
                _created(session, "resp_stale", item_id="item_a")
                expected = ("resp_b", "item_b")
            elif kind == 3:
                _commit(session, "item_a")
                for label in ("b", "c", "d")[: 1 + (extra % 3)]:
                    _speech(session, f"item_{label}")
                    _commit(session, f"item_{label}")
                _created(session, "resp_final")
                expected = ("resp_final", session._assembler._current_user_item_id)
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
        except Exception as error:
            mispairs.append(f"index={index} kind={kind} raised {type(error).__name__}: {error}")
            continue

        pairing = _pairing(session)
        map_growth.append(len(pairing) + len(session._cancelled_set))
        if expected is not None:
            response_id, user_item = expected
            bound = pairing.get(response_id)
            if bound not in {user_item, None} and user_item is not None:
                mispairs.append(
                    f"index={index} kind={kind} bound {response_id} -> {bound!r}, "
                    f"expected {user_item!r}; pairing={pairing}"
                )
            if bound == "item_a" and user_item == "item_b":
                mispairs.append(
                    f"index={index} kind={kind} stale A paired onto B's response"
                )
            if kind == 1 and pairing.get("resp_stale") == "item_b":
                mispairs.append(
                    f"index={index} kind=1 stale resp_stale bound to live item_b; "
                    f"pairing={pairing} cancelled={set(session._cancelled_set)}"
                )
        if kind == 8 and pairing.get("resp_orphan") == "item_a":
            mispairs.append(
                f"index={index} orphan response bound to later user item; pairing={pairing}"
            )

    assert mispairs == [], "wrong pairing or handler crash:\n" + "\n".join(mispairs)
    assert all(size <= 48 for size in map_growth)


def test_deep_m25_cancelled_id_map_is_ring_bounded() -> None:
    session = _voice_session()
    for index in range(64):
        session._mark_response_cancelled(f"resp_{index}")
    assert len(session._cancelled_set) <= 16
    assert len(session._cancelled_response_ids) <= 16
