"""Tests for Milestone 25 realtime voice session engine."""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

from src.active_memory import ActiveMemoryContext
from src.config import (
    MAX_REALTIME_VOICE_SESSION_MINUTES,
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_INPUT_QUEUE_FRAMES,
    REALTIME_VOICE_OUTPUT_QUEUE_BYTES,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
)
from src.conversation import ConversationHistory
from src.realtime_voice import (
    REALTIME_CLEANUP_INCOMPLETE,
    REALTIME_INPUT_OVERFLOW,
    REALTIME_OUTPUT_OVERFLOW,
    REALTIME_POLICY_FAILURE,
    REALTIME_SDK_INCOMPATIBLE,
    REALTIME_SESSION_TIMEOUT,
    REALTIME_STOPPED_MESSAGE,
    RealtimeVoiceSession,
    _TurnAssembler,
    build_session_update_payload,
    run_realtime_voice_session,
)
from src.realtime_voice_input import (
    REALTIME_INPUT_OVERFLOW as INPUT_OVERFLOW_MESSAGE,
    RealtimeAudioFrame,
    RealtimeMicrophoneStream,
)
from src.settings import Settings


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
    id: str
    status: str = "completed"


class FakeSocket:
    def __init__(self) -> None:
        self._events: Queue[object] = Queue()
        self._map: dict[bytes, object] = {}
        self.closed = False

    def push(self, event: object) -> None:
        self._events.put(event)

    def recv(self, timeout: float | None = None, decode: bool | None = None) -> bytes:
        del decode
        try:
            item = self._events.get(timeout=0.05 if timeout is None else timeout)
        except Empty as error:
            raise TimeoutError from error
        if self.closed:
            raise ConnectionError("closed")
        key = str(id(item)).encode("ascii")
        self._map[key] = item
        return key


class FakeResource:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self.updates: list[Any] = []
        self.appends: list[str] = []
        self.cancels: list[str | None] = []
        self.created_items: list[Any] = []

    def update(self, *, session: Any, event_id: str | None = None) -> None:
        del event_id
        self.updates.append(session)
        self._connection.socket.push(FakeEvent(type="session.updated"))

    def append(self, *, audio: str, event_id: str | None = None) -> None:
        del event_id
        self.appends.append(audio)

    def cancel(
        self,
        *,
        response_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        del event_id
        self.cancels.append(response_id)

    def create(
        self,
        *,
        item: Any,
        event_id: str | None = None,
        previous_item_id: str | None = None,
    ) -> None:
        del event_id, previous_item_id
        self.created_items.append(item)


class FakeConversation:
    def __init__(self, connection: FakeConnection) -> None:
        self.item = FakeResource(connection)


class FakeConnection:
    def __init__(self) -> None:
        self.socket = FakeSocket()
        self._connection = self.socket
        self.session = FakeResource(self)
        self.response = FakeResource(self)
        self.input_audio_buffer = FakeResource(self)
        self.conversation = FakeConversation(self)
        self.closed = False
        self.socket.push(FakeEvent(type="session.created"))

    def send(self, event: object) -> None:
        del event

    def recv(self) -> object:
        raw = self.socket.recv(timeout=0.5)
        return self.parse_event(raw)

    def parse_event(self, data: str | bytes) -> object:
        key = data if isinstance(data, bytes) else data.encode("ascii")
        return self.socket._map[key]

    def close(self, *, code: int = 1000, reason: str = "") -> None:
        del code, reason
        self.closed = True
        self.socket.closed = True


class FakeManager:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def enter(self) -> FakeConnection:
        return self.connection

    def __enter__(self) -> FakeConnection:
        return self.connection

    def __exit__(self, *args: object) -> None:
        self.connection.close()


class FakePlaybackStream:
    instances: list[FakePlaybackStream] = []

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.abort_calls = 0
        self.stop_calls = 0
        self.start_calls = 0
        self.closed = False
        FakePlaybackStream.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def abort(self) -> None:
        self.abort_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.closed = True


class FakeMicrophone(RealtimeMicrophoneStream):
    def __init__(
        self,
        frame_queue: Queue[RealtimeAudioFrame],
        on_overflow: Callable[[], None] | None = None,
        on_capture_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        super().__init__(
            frame_queue=frame_queue,
            on_overflow=on_overflow,
            on_capture_error=on_capture_error,
        )
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self._active.set()

    def stop(self) -> None:
        self.stopped = True
        self._active.clear()


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _pcm_frame(sequence: int = 0) -> RealtimeAudioFrame:
    return RealtimeAudioFrame(
        pcm_bytes=b"\x00\x00" * (REALTIME_VOICE_FRAME_BYTES // 2),
        sample_rate=REALTIME_VOICE_SAMPLE_RATE_HZ,
        channels=1,
        sample_width_bytes=2,
        sequence=sequence,
    )


def _run_session(
    connection: FakeConnection,
    history: ConversationHistory,
    *,
    printed: list[str] | None = None,
    microphone_factory: Callable[..., FakeMicrophone] | None = None,
    playback_stream_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> tuple[threading.Thread, RealtimeVoiceSession, dict[str, str], list[str]]:
    FakePlaybackStream.instances.clear()
    lines = printed if printed is not None else []
    result_box: dict[str, str] = {}
    mic_factory = microphone_factory or (
        lambda q, o, e: FakeMicrophone(q, o, e)
    )
    session = RealtimeVoiceSession(
        settings=_settings(),
        client=object(),  # type: ignore[arg-type]
        conversation_history=history,
        active_memory_context=ActiveMemoryContext(),
        logger_=logging.getLogger("test"),
        connect_factory=lambda: FakeManager(connection),
        microphone_factory=mic_factory,
        playback_stream_factory=playback_stream_factory or FakePlaybackStream,
        print_fn=lines.append,
        monotonic_fn=monotonic_fn or time.monotonic,
        sleep_fn=sleep_fn or time.sleep,
    )

    def _target() -> None:
        result_box["message"] = session.run()

    thread = threading.Thread(target=_target)
    thread.start()
    assert _wait_until(
        lambda: session.state.value in {"listening", "responding"}
        or "message" in result_box
    )
    return thread, session, result_box, lines


def test_session_update_payload_disables_tools() -> None:
    payload = build_session_update_payload(
        settings=_settings(),
        instructions="test",
    )
    assert payload["tools"] == []
    assert payload["tool_choice"] == "none"
    audio = payload["audio"]
    assert isinstance(audio, dict)
    input_cfg = audio["input"]
    assert isinstance(input_cfg, dict)
    assert input_cfg["format"] == {"type": "audio/pcm", "rate": 24000}
    turn = input_cfg["turn_detection"]
    assert isinstance(turn, dict)
    assert turn["type"] == "server_vad"
    assert turn["interrupt_response"] is True
    assert "idle_timeout_ms" not in turn


def test_turn_assembler_commits_completed_pair() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assembler.store_user_transcript("item_u", "hello")
    assembler.store_assistant_transcript("resp_1", "hi there")
    result = assembler.on_response_done(response_id="resp_1", status="completed")
    assert result == "pair"
    assert history.turns[0].content == "hello"
    assert history.turns[1].content == "hi there"


def test_turn_assembler_cancelled_commits_user_only() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assembler.store_user_transcript("item_u", "interrupt me")
    assembler.store_assistant_transcript("resp_1", "partial answer")
    result = assembler.on_response_done(response_id="resp_1", status="cancelled")
    assert result == "user_only"
    assert len(history.turns) == 1
    assert history.turns[0].role == "user"
    assert history.turns[0].content == "interrupt me"


def test_realtime_session_happy_path_and_cleanup() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    history.add_user_message("prior")
    history.add_assistant_message("prior reply")
    thread, session, result_box, printed = _run_session(connection, history)

    assert _wait_until(lambda: bool(connection.conversation.item.created_items))
    assert connection.conversation.item.created_items
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="item_1")
    )
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.committed", item_id="item_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="item_1",
            transcript="hello cortana",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.created",
            response=FakeResponse(id="resp_1", status="in_progress"),
        )
    )
    pcm = b"\x01\x00" * 100
    connection.socket.push(
        FakeEvent(
            type="response.output_audio.delta",
            response_id="resp_1",
            delta=base64.b64encode(pcm).decode("ascii"),
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id="resp_1",
            transcript="hello human",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_1", status="completed"),
        )
    )
    assert _wait_until(
        lambda: any(t.content == "hello cortana" for t in history.turns)
        and any(t.content == "hello human" for t in history.turns)
    )
    assert any("Heard" in line for line in printed)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] == REALTIME_STOPPED_MESSAGE
    assert FakePlaybackStream.instances
    assert FakePlaybackStream.instances[0].closed is True
    assert connection.closed is True


def test_barge_in_aborts_and_rejects_stale_audio() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)

    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="item_a")
    )
    connection.socket.push(
        FakeEvent(
            type="response.created",
            response=FakeResponse(id="resp_a", status="in_progress"),
        )
    )
    pcm_a = b"\x02\x00" * 200
    connection.socket.push(
        FakeEvent(
            type="response.output_audio.delta",
            response_id="resp_a",
            delta=base64.b64encode(pcm_a).decode("ascii"),
        )
    )
    assert _wait_until(
        lambda: bool(FakePlaybackStream.instances)
        and len(FakePlaybackStream.instances[0].writes) >= 1
    )
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="item_b")
    )
    assert _wait_until(lambda: FakePlaybackStream.instances[0].abort_calls >= 1)
    writes_after = len(FakePlaybackStream.instances[0].writes)
    connection.socket.push(
        FakeEvent(
            type="response.output_audio.delta",
            response_id="resp_a",
            delta=base64.b64encode(b"\x03\x00" * 200).decode("ascii"),
        )
    )
    time.sleep(0.15)
    assert len(FakePlaybackStream.instances[0].writes) == writes_after
    assert connection.response.cancels == []
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id="resp_a",
            transcript="partial answer should not persist",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_a", status="cancelled"),
        )
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="item_a",
            transcript="first question",
        )
    )
    assert _wait_until(
        lambda: any(t.content == "first question" for t in history.turns)
    )
    assert all(
        t.content != "partial answer should not persist" for t in history.turns
    )
    assert all(t.role == "user" for t in history.turns if t.content == "first question")
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False


def test_tool_call_event_is_policy_failure() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, _session, result_box, _printed = _run_session(connection, history)
    connection.socket.push(FakeEvent(type="response.function_call_arguments.done"))
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] == REALTIME_POLICY_FAILURE


def test_connect_uses_max_retries_zero() -> None:
    seen: dict[str, Any] = {}
    connection = FakeConnection()

    class Client:
        class realtime:
            @staticmethod
            def connect(*, model: str, max_retries: int = 5) -> FakeManager:
                seen["model"] = model
                seen["max_retries"] = max_retries
                return FakeManager(connection)

    history = ConversationHistory()
    result_box: dict[str, str] = {}
    session = RealtimeVoiceSession(
        settings=_settings(),
        client=Client(),  # type: ignore[arg-type]
        conversation_history=history,
        active_memory_context=ActiveMemoryContext(),
        logger_=logging.getLogger("test"),
        microphone_factory=lambda q, o, e: FakeMicrophone(q, o, e),
        playback_stream_factory=FakePlaybackStream,
        print_fn=lambda _m: None,
    )

    def _target() -> None:
        result_box["message"] = session.run()

    thread = threading.Thread(target=_target)
    thread.start()
    assert _wait_until(lambda: seen.get("max_retries") == 0)
    assert seen["model"] == "gpt-realtime-mini"
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] == REALTIME_STOPPED_MESSAGE


def test_mic_opens_only_after_session_ready() -> None:
    connection = FakeConnection()
    original_update = FakeResource.update
    started = {"mic": False}

    class GatedMicrophone(FakeMicrophone):
        def start(self) -> None:
            started["mic"] = True
            super().start()

    def delayed_update(
        self: FakeResource,
        *,
        session: Any,
        event_id: str | None = None,
    ) -> None:
        del event_id
        self.updates.append(session)

    FakeResource.update = delayed_update  # type: ignore[method-assign]
    try:
        history = ConversationHistory()
        result_box: dict[str, str] = {}
        session = RealtimeVoiceSession(
            settings=_settings(),
            client=object(),  # type: ignore[arg-type]
            conversation_history=history,
            active_memory_context=ActiveMemoryContext(),
            logger_=logging.getLogger("test"),
            connect_factory=lambda: FakeManager(connection),
            microphone_factory=lambda q, o, e: GatedMicrophone(q, o, e),
            playback_stream_factory=FakePlaybackStream,
            print_fn=lambda _m: None,
        )

        def _target() -> None:
            result_box["message"] = session.run()

        thread = threading.Thread(target=_target)
        thread.start()
        assert _wait_until(lambda: bool(connection.session.updates))
        time.sleep(0.2)
        assert started["mic"] is False
        connection.socket.push(FakeEvent(type="session.updated"))
        assert _wait_until(lambda: started["mic"] is True)
        session.request_stop(error_type="cancelled")
        thread.join(timeout=5)
        assert thread.is_alive() is False
    finally:
        FakeResource.update = original_update  # type: ignore[method-assign]


def test_frame_constants_are_derived() -> None:
    assert REALTIME_VOICE_FRAME_BYTES == 960
    assert REALTIME_VOICE_SAMPLE_RATE_HZ == 24_000


def _assert_single_pair(history: ConversationHistory, user: str, assistant: str) -> None:
    assert len(history.turns) == 2
    assert history.turns[0].role == "user"
    assert history.turns[0].content == user
    assert history.turns[1].role == "assistant"
    assert history.turns[1].content == assistant


def test_turn_assembler_done_then_user_then_assistant() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assert assembler.on_response_done(response_id="resp_1", status="completed") == "none"
    assembler.store_user_transcript("item_u", "hello")
    assert len(history.turns) == 0
    assembler.store_assistant_transcript("resp_1", "hi there")
    _assert_single_pair(history, "hello", "hi there")


def test_turn_assembler_done_then_assistant_then_user() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assert assembler.on_response_done(response_id="resp_1", status="completed") == "none"
    assembler.store_assistant_transcript("resp_1", "hi there")
    assert len(history.turns) == 0
    assembler.store_user_transcript("item_u", "hello")
    _assert_single_pair(history, "hello", "hi there")


def test_turn_assembler_user_then_done_then_assistant() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assembler.store_user_transcript("item_u", "hello")
    assert assembler.on_response_done(response_id="resp_1", status="completed") == "none"
    assembler.store_assistant_transcript("resp_1", "hi there")
    _assert_single_pair(history, "hello", "hi there")


def test_turn_assembler_assistant_then_done_then_user() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assembler.store_assistant_transcript("resp_1", "hi there")
    assert assembler.on_response_done(response_id="resp_1", status="completed") == "none"
    assembler.store_user_transcript("item_u", "hello")
    _assert_single_pair(history, "hello", "hi there")


def test_turn_assembler_duplicate_finals_commit_once() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assembler.store_user_transcript("item_u", "hello")
    assembler.store_assistant_transcript("resp_1", "hi there")
    assert assembler.on_response_done(response_id="resp_1", status="completed") == "pair"
    assembler.store_user_transcript("item_u", "hello")
    assembler.store_assistant_transcript("resp_1", "hi there")
    assert assembler.on_response_done(response_id="resp_1", status="completed") == "none"
    _assert_single_pair(history, "hello", "hi there")


def test_turn_assembler_late_assistant_after_cancel_never_pairs() -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_u")
    assembler.bind_response("resp_1")
    assembler.store_user_transcript("item_u", "hello")
    assert assembler.on_response_done(response_id="resp_1", status="cancelled") == (
        "user_only"
    )
    assembler.store_assistant_transcript("resp_1", "should not appear")
    assert len(history.turns) == 1
    assert history.turns[0].content == "hello"


def test_incompatible_sdk_fails_before_microphone() -> None:
    class BadSocket:
        pass

    class BadConnection:
        def __init__(self) -> None:
            self.session = FakeResource(FakeConnection())
            self.response = FakeResource(FakeConnection())
            self.input_audio_buffer = FakeResource(FakeConnection())
            self.conversation = FakeConversation(FakeConnection())
            self.closed = False
            self.recv_calls = 0

        def recv(self) -> object:
            self.recv_calls += 1
            raise AssertionError("public blocking recv must not be used")

        def parse_event(self, data: str | bytes) -> object:
            del data
            raise AssertionError("parse_event should not run")

        def close(self, *, code: int = 1000, reason: str = "") -> None:
            del code, reason
            self.closed = True

    bad = BadConnection()
    started = {"mic": False}

    class GatedMicrophone(FakeMicrophone):
        def start(self) -> None:
            started["mic"] = True
            raise AssertionError("microphone must not open")

    message = run_realtime_voice_session(
        settings=_settings(),
        client=object(),  # type: ignore[arg-type]
        conversation_history=ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        logger_=logging.getLogger("test"),
        connect_factory=lambda: FakeManager(bad),  # type: ignore[arg-type]
        microphone_factory=lambda q, o, e: GatedMicrophone(q, o, e),
        playback_stream_factory=FakePlaybackStream,
        print_fn=lambda _m: None,
    )
    assert message == REALTIME_SDK_INCOMPATIBLE
    assert started["mic"] is False
    assert bad.recv_calls == 0


def test_keyboard_interrupt_mid_session_cleans_up() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    armed = threading.Event()
    sleeps = {"n": 0}

    def sleep_fn(_seconds: float) -> None:
        # Wait until the session has reached listening, then interrupt.
        if not armed.wait(timeout=2.0):
            return
        sleeps["n"] += 1
        if sleeps["n"] >= 1:
            raise KeyboardInterrupt

    thread, session, result_box, _printed = _run_session(
        connection,
        history,
        sleep_fn=sleep_fn,
    )
    assert session.state.value in {"listening", "responding"}
    armed.set()
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] == REALTIME_STOPPED_MESSAGE
    assert connection.closed is True
    assert FakePlaybackStream.instances
    assert FakePlaybackStream.instances[0].closed is True
    assert session.state.value in {"closed", "failed"}


def test_hard_session_timeout_lifecycle() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    clock = {"t": 1_000.0}
    armed = threading.Event()

    def monotonic_fn() -> float:
        return clock["t"]

    def sleep_fn(_seconds: float) -> None:
        if not armed.wait(timeout=2.0):
            return
        clock["t"] += (MAX_REALTIME_VOICE_SESSION_MINUTES * 60) + 1

    thread, session, result_box, _printed = _run_session(
        connection,
        history,
        monotonic_fn=monotonic_fn,
        sleep_fn=sleep_fn,
    )
    assert session.state.value in {"listening", "responding"}
    armed.set()
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] == REALTIME_SESSION_TIMEOUT
    assert connection.closed is True
    assert FakePlaybackStream.instances[0].closed is True
    assert session.state.value == "failed"


def test_input_queue_overflow_terminates_session() -> None:
    connection = FakeConnection()
    history = ConversationHistory()

    class OverflowMicrophone(FakeMicrophone):
        def start(self) -> None:
            super().start()
            for index in range(REALTIME_VOICE_INPUT_QUEUE_FRAMES):
                self._frame_queue.put_nowait(_pcm_frame(index))
            try:
                self._frame_queue.put_nowait(
                    _pcm_frame(REALTIME_VOICE_INPUT_QUEUE_FRAMES)
                )
            except Exception:
                if self._on_overflow is not None:
                    self._on_overflow()

    thread, session, result_box, _printed = _run_session(
        connection,
        history,
        microphone_factory=lambda q, o, e: OverflowMicrophone(q, o, e),
    )
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] in {REALTIME_INPUT_OVERFLOW, INPUT_OVERFLOW_MESSAGE}
    assert connection.closed is True
    assert FakePlaybackStream.instances[0].closed is True
    assert session.state.value == "failed"
    assert history.turns == []


def test_output_queue_overflow_terminates_session() -> None:
    connection = FakeConnection()
    history = ConversationHistory()

    class StallPlayback:
        instances: list[StallPlayback] = []

        def __init__(self) -> None:
            self.closed = False
            self.abort_calls = 0
            self._release = threading.Event()
            self.write_started = threading.Event()
            StallPlayback.instances.append(self)

        def start(self) -> None:
            return None

        def write(self, data: bytes) -> None:
            del data
            self.write_started.set()
            # Block until cleanup aborts/closes so the queue can back up.
            self._release.wait(timeout=30)

        def abort(self) -> None:
            self.abort_calls += 1
            self._release.set()

        def stop(self) -> None:
            self._release.set()

        def close(self) -> None:
            self._release.set()
            self.closed = True

    StallPlayback.instances.clear()
    thread, session, result_box, _printed = _run_session(
        connection,
        history,
        playback_stream_factory=StallPlayback,
    )
    chunk = b"\x01\x00" * 1_000
    encoded = base64.b64encode(chunk).decode("ascii")
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="item_o")
    )
    connection.socket.push(
        FakeEvent(
            type="response.created",
            response=FakeResponse(id="resp_o", status="in_progress"),
        )
    )
    # One chunk starts blocking write; remaining chunks fill the bounded queue.
    connection.socket.push(
        FakeEvent(
            type="response.output_audio.delta",
            response_id="resp_o",
            delta=encoded,
        )
    )
    assert _wait_until(
        lambda: bool(StallPlayback.instances) and StallPlayback.instances[0].write_started.is_set()
    )
    needed = (REALTIME_VOICE_OUTPUT_QUEUE_BYTES // len(chunk)) + 5
    for _ in range(needed):
        connection.socket.push(
            FakeEvent(
                type="response.output_audio.delta",
                response_id="resp_o",
                delta=encoded,
            )
        )
    assert _wait_until(
        lambda: result_box.get("message") == REALTIME_OUTPUT_OVERFLOW,
        timeout=8.0,
    )
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert result_box["message"] == REALTIME_OUTPUT_OVERFLOW
    assert connection.closed is True
    assert StallPlayback.instances[0].closed is True
    assert all(turn.role != "assistant" for turn in history.turns)
    assert session.state.value == "failed"


def test_quiet_recv_does_not_starve_microphone_appends() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    feed_stop = threading.Event()

    class FeedingMicrophone(FakeMicrophone):
        def start(self) -> None:
            super().start()

            def _feed() -> None:
                sequence = 0
                while not feed_stop.is_set() and self.is_active:
                    try:
                        self._frame_queue.put(_pcm_frame(sequence), timeout=0.05)
                        sequence += 1
                    except Exception:
                        return

            threading.Thread(target=_feed, daemon=True).start()

        def stop(self) -> None:
            feed_stop.set()
            super().stop()

    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        microphone_factory=lambda q, o, e: FeedingMicrophone(q, o, e),
    )
    # No server events after setup; timed recv should keep timing out while
    # microphone frames continue to be appended.
    assert _wait_until(lambda: len(connection.input_audio_buffer.appends) >= 5)
    first = len(connection.input_audio_buffer.appends)
    assert _wait_until(
        lambda: len(connection.input_audio_buffer.appends) >= first + 5
    )
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False


def test_join_timeout_surfaces_cleanup_incomplete() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    session = RealtimeVoiceSession(
        settings=_settings(),
        client=object(),  # type: ignore[arg-type]
        conversation_history=history,
        active_memory_context=ActiveMemoryContext(),
        logger_=logging.getLogger("test"),
        connect_factory=lambda: FakeManager(connection),
        microphone_factory=lambda q, o, e: FakeMicrophone(q, o, e),
        playback_stream_factory=FakePlaybackStream,
        print_fn=lambda _m: None,
    )

    class ZombieThread:
        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

    session._session_thread = ZombieThread()  # type: ignore[assignment]
    session._playback_thread = None
    session._cleanup()
    assert session._cleanup_incomplete is True
    assert session._failure_message == REALTIME_CLEANUP_INCOMPLETE
    assert session.state.value == "failed"
