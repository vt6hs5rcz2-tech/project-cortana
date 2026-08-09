"""Tests for Milestone 26 realtime multimodal session engine."""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

from PIL import Image

from src.active_memory import ActiveMemoryContext
from src.camera_capture import CameraCaptureSession, RealtimeVisualFrame
from src.config import (
    REALTIME_VISUAL_FIXED_LABEL,
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
)
from src.conversation import ConversationHistory
from src.realtime_multimodal import (
    MULTIMODAL_CAMERA_START_FAILED,
    MULTIMODAL_STARTED_MESSAGE,
    MULTIMODAL_STOPPED_MESSAGE,
    MULTIMODAL_VISUAL_DELETE_FAILED,
    RealtimeMultimodalSession,
    build_multimodal_session_update_payload,
    build_visual_conversation_item,
    run_realtime_multimodal_session,
)
from src.realtime_voice_input import RealtimeAudioFrame
from src.settings import Settings
from src.vision_normalize import encode_metadata_free_png


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
    item: Any | None = None


@dataclass
class FakeResponse:
    id: str
    status: str = "completed"


@dataclass
class FakeItem:
    id: str


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
        self.deleted_ids: list[str] = []
        self.response_creates = 0
        self.pending_response_ids: list[str] = []
        self._visual_seq = 0

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
        item: Any | None = None,
        event_id: str | None = None,
        previous_item_id: str | None = None,
        response: Any | None = None,
    ) -> None:
        del event_id, previous_item_id, response
        if item is not None:
            self.created_items.append(item)
            content = item.get("content") if isinstance(item, dict) else None
            is_visual = False
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        is_visual = True
                        break
            if is_visual:
                self._visual_seq += 1
                visual_id = f"visual_{self._visual_seq}"
                fake_item = FakeItem(id=visual_id)
                self._connection.socket.push(
                    FakeEvent(type="conversation.item.added", item=fake_item)
                )
                self._connection.socket.push(
                    FakeEvent(type="conversation.item.done", item=fake_item)
                )
            return
        # Bare response.create()
        self.response_creates += 1
        response_id = f"resp_{self.response_creates}"
        self.pending_response_ids.append(response_id)
        if self._connection.auto_ack_responses:
            self._connection.socket.push(
                FakeEvent(
                    type="response.created",
                    response=FakeResponse(id=response_id),
                )
            )

    def delete(self, *, item_id: str, event_id: str | None = None) -> None:
        del event_id
        self.deleted_ids.append(item_id)
        self._connection.socket.push(
            FakeEvent(type="conversation.item.deleted", item_id=item_id)
        )


class FakeConversation:
    def __init__(self, connection: FakeConnection) -> None:
        self.item = FakeResource(connection)


class FakeConnection:
    def __init__(self, *, auto_ack_responses: bool = True) -> None:
        self.auto_ack_responses = auto_ack_responses
        self.socket = FakeSocket()
        self._connection = self.socket
        self.session = FakeResource(self)
        self.response = FakeResource(self)
        self.input_audio_buffer = FakeResource(self)
        self.conversation = FakeConversation(self)
        self.closed = False
        self.socket.push(FakeEvent(type="session.created"))

    def ack_response_created(self, response_id: str) -> None:
        """Push a deferred response.created acknowledgement."""
        self.socket.push(
            FakeEvent(
                type="response.created",
                response=FakeResponse(id=response_id),
            )
        )

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


class FakeMicrophone:
    def __init__(
        self,
        frame_queue: Queue[RealtimeAudioFrame],
        on_overflow: Callable[[], None] | None = None,
        on_capture_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.frame_queue = frame_queue
        self.on_overflow = on_overflow
        self.on_capture_error = on_capture_error
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeCapture:
    def __init__(self, frames: list[Any]) -> None:
        self.frames = list(frames)
        self.opened = True
        self.release_calls = 0
        self.open_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop_id: int, value: float) -> bool:
        del prop_id, value
        return True

    def read(self) -> tuple[bool, Any]:
        if not self.frames:
            # Keep returning last solid frame for worker pacing.
            import numpy as np

            return True, np.zeros((32, 32, 3), dtype="uint8")
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.release_calls += 1
        self.opened = False


def _bgr(width: int = 64, height: int = 64) -> Any:
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)
    return frame


def _png_frame(sequence: int = 0, when: float = 1.0) -> RealtimeVisualFrame:
    image = Image.new("RGB", (32, 32), "red")
    png, width, height = encode_metadata_free_png(image)
    return RealtimeVisualFrame(
        image_bytes=png,
        mime_type="image/png",
        width=width,
        height=height,
        sequence=sequence,
        captured_at_monotonic=when,
    )


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
    camera_fail: bool = False,
) -> tuple[threading.Thread, RealtimeMultimodalSession, dict[str, str], list[str]]:
    FakePlaybackStream.instances.clear()
    lines = printed if printed is not None else []
    result_box: dict[str, str] = {}
    open_order: list[str] = []

    capture = FakeCapture(frames=[_bgr()])
    if camera_fail:
        from src.camera_capture import CAMERA_UNAVAILABLE, CameraCaptureError

        def bad_factory() -> FakeCapture:
            raise CameraCaptureError(
                CAMERA_UNAVAILABLE,
                error_type="camera_unavailable",
            )

        camera_factory = None
        capture_factory: Callable[[], Any] | None = bad_factory
    else:
        def camera_factory() -> CameraCaptureSession:
            session = CameraCaptureSession(
                capture_factory=lambda: capture,
                sample_interval_seconds=0.05,
                sleep_fn=lambda _s: time.sleep(0.01),
            )
            return session

        capture_factory = None

    mic_box: dict[str, FakeMicrophone] = {}

    def mic_factory(
        q: Queue[RealtimeAudioFrame],
        o: Callable[[], None],
        e: Callable[[BaseException], None],
    ) -> FakeMicrophone:
        mic = FakeMicrophone(q, o, e)
        mic_box["mic"] = mic
        return mic

    session = RealtimeMultimodalSession(
        settings=_settings(),
        client=object(),  # type: ignore[arg-type]
        conversation_history=history,
        active_memory_context=ActiveMemoryContext(),
        logger_=logging.getLogger("test"),
        connect_factory=lambda: FakeManager(connection),
        microphone_factory=mic_factory,
        playback_stream_factory=FakePlaybackStream,
        camera_factory=camera_factory,
        capture_factory=capture_factory,
        print_fn=lines.append,
    )

    # Patch open order tracking.
    original_make_camera = session._make_camera
    original_make_mic = session._make_microphone

    def tracked_camera() -> CameraCaptureSession:
        open_order.append("camera_make")
        cam = original_make_camera()
        original_open = cam.open_and_capture_first

        def tracked_open() -> RealtimeVisualFrame:
            open_order.append("camera_open")
            return original_open()

        cam.open_and_capture_first = tracked_open  # type: ignore[method-assign]
        return cam

    def tracked_mic() -> FakeMicrophone:
        open_order.append("mic_make")
        mic = original_make_mic()
        original_start = mic.start

        def tracked_start() -> None:
            open_order.append("mic_start")
            original_start()

        mic.start = tracked_start  # type: ignore[method-assign]
        return mic  # type: ignore[return-value]

    session._make_camera = tracked_camera  # type: ignore[method-assign]
    session._make_microphone = tracked_mic  # type: ignore[method-assign]
    session._test_open_order = open_order  # type: ignore[attr-defined]
    session._test_mic_box = mic_box  # type: ignore[attr-defined]

    def _target() -> None:
        result_box["message"] = session.run()

    thread = threading.Thread(target=_target)
    thread.start()
    assert _wait_until(
        lambda: session.state.value in {"listening", "responding", "failed", "closed"}
        or "message" in result_box
    )
    return thread, session, result_box, lines


def test_multimodal_session_update_payload_uses_manual_response() -> None:
    payload = build_multimodal_session_update_payload(
        settings=_settings(),
        instructions="test",
    )
    assert payload["tools"] == []
    assert payload["tool_choice"] == "none"
    audio = payload["audio"]
    assert isinstance(audio, dict)
    turn = audio["input"]["turn_detection"]  # type: ignore[index]
    assert turn["type"] == "server_vad"
    assert turn["create_response"] is False
    assert turn["interrupt_response"] is True


def test_visual_item_shape_uses_fixed_label_and_low_detail() -> None:
    frame = _png_frame()
    item = build_visual_conversation_item(frame)
    content = item["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == REALTIME_VISUAL_FIXED_LABEL
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "low"
    assert str(content[1]["image_url"]).startswith("data:image/png;base64,")


def test_startup_order_camera_before_mic_and_banner() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, result_box, lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    order = session._test_open_order  # type: ignore[attr-defined]
    assert order.index("camera_open") < order.index("mic_start")
    assert any(MULTIMODAL_STARTED_MESSAGE in line for line in lines) or (
        MULTIMODAL_STARTED_MESSAGE in result_box.get("message", "")
    )
    # Banner is printed after start; may already be in lines.
    assert _wait_until(lambda: any("camera are active" in line for line in lines))
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert connection.closed
    assert result_box["message"] == MULTIMODAL_STOPPED_MESSAGE


def test_camera_failure_before_mic() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, result_box, _lines = _run_session(
        connection,
        history,
        camera_fail=True,
    )
    thread.join(timeout=5)
    assert session.microphone_opened is False
    assert "camera" in result_box["message"].casefold() or result_box[
        "message"
    ] == MULTIMODAL_CAMERA_START_FAILED


def test_turn_binds_visual_creates_response_and_deletes() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    history.add_user_message("prior")
    history.add_assistant_message("prior-ack")
    thread, session, result_box, lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    assert _wait_until(lambda: len(connection.conversation.item.created_items) >= 2)

    connection.socket.push(FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1"))
    assert _wait_until(lambda: connection.response.response_creates >= 1, timeout=5)
    assert _wait_until(
        lambda: any(
            isinstance(item, dict)
            and any(
                isinstance(part, dict) and part.get("type") == "input_image"
                for part in item.get("content", [])
            )
            for item in connection.conversation.item.created_items
        ),
        timeout=5,
    )

    # Complete response so visual item is deleted.
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What am I looking at?",
        )
    )
    # response.created already auto-pushed by FakeResource.create
    assert _wait_until(lambda: session.state.value == "responding", timeout=5)
    active = session._active_response_id
    assert isinstance(active, str)
    pcm = base64.b64encode(b"\x00\x00" * 80).decode("ascii")
    connection.socket.push(
        FakeEvent(type="response.output_audio.delta", response_id=active, delta=pcm)
    )
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id=active,
            transcript="A red square on a white background.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id=active, status="completed"),
        )
    )
    assert _wait_until(lambda: len(connection.conversation.item.deleted_ids) >= 1, timeout=5)

    # Second turn gets a new visual item; first should already be deleted.
    connection.socket.push(FakeEvent(type="input_audio_buffer.committed", item_id="user_2"))
    assert _wait_until(lambda: connection.response.response_creates >= 2, timeout=5)
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_2",
            transcript="What is this?",
        )
    )
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    active2 = session._active_response_id
    assert isinstance(active2, str)
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id=active2,
            transcript="Still a red square.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id=active2, status="completed"),
        )
    )
    assert _wait_until(lambda: len(connection.conversation.item.deleted_ids) >= 2, timeout=5)

    # History remains text only.
    for turn in history.turns:
        assert "base64" not in turn.content
        assert "data:image" not in turn.content
        assert REALTIME_VISUAL_FIXED_LABEL not in turn.content

    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert "message" in result_box


def test_barge_in_does_not_wait_on_vision() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u1"))
    assert _wait_until(lambda: connection.response.response_creates >= 1)
    assert _wait_until(lambda: session._active_response_id is not None)
    response_id = session._active_response_id
    assert isinstance(response_id, str)
    pcm = base64.b64encode(b"\x01\x00" * 80).decode("ascii")
    connection.socket.push(
        FakeEvent(type="response.output_audio.delta", response_id=response_id, delta=pcm)
    )
    assert _wait_until(lambda: len(FakePlaybackStream.instances[0].writes) >= 1)
    connection.socket.push(FakeEvent(type="input_audio_buffer.speech_started", item_id="u2"))
    assert _wait_until(lambda: FakePlaybackStream.instances[0].abort_calls >= 1)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_visual_delete_failure_fails_session() -> None:
    connection = FakeConnection()

    def failing_delete(*, item_id: str, event_id: str | None = None) -> None:
        del item_id, event_id
        raise RuntimeError("delete failed")

    connection.conversation.item.delete = failing_delete  # type: ignore[method-assign]
    history = ConversationHistory()
    thread, session, result_box, _lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u1"))
    assert _wait_until(lambda: connection.response.response_creates >= 1)
    assert _wait_until(lambda: session._active_response_id is not None)
    active = session._active_response_id
    assert isinstance(active, str)
    # Ensure visual remote id was acked.
    assert _wait_until(
        lambda: any(
            turn.remote_visual_item_id is not None
            for turn in session._visual_turns.values()
        ),
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id=active, status="completed"),
        )
    )
    thread.join(timeout=5)
    assert result_box["message"] == MULTIMODAL_VISUAL_DELETE_FAILED


def test_rapid_double_turn_response_linkage_fifo() -> None:
    """M26-F2: two outstanding response.create calls link FIFO without cross-talk."""
    connection = FakeConnection(auto_ack_responses=False)
    history = ConversationHistory()
    thread, session, _result_box, _lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)

    # Turn A then Turn B. Allow visual item acks to settle, but keep both
    # response.create calls outstanding before any response.created.
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_a")
    )
    assert _wait_until(lambda: connection.response.response_creates >= 1, timeout=5)
    assert _wait_until(
        lambda: session._visual_turns.get("user_a") is not None
        and session._visual_turns["user_a"].response_create_sent
        and session._visual_turns["user_a"].remote_visual_item_id is not None,
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.committed", item_id="user_b")
    )
    assert _wait_until(lambda: connection.response.response_creates >= 2, timeout=5)
    assert _wait_until(
        lambda: session._visual_turns.get("user_b") is not None
        and session._visual_turns["user_b"].response_create_sent
        and session._visual_turns["user_b"].remote_visual_item_id is not None,
        timeout=5,
    )
    # Both creates outstanding; no response linkage yet.
    assert session._visual_turns["user_a"].response_id is None
    assert session._visual_turns["user_b"].response_id is None
    assert connection.response.pending_response_ids[:2] == ["resp_1", "resp_2"]

    # Provider-valid order: acknowledgements arrive in create order.
    connection.ack_response_created("resp_1")
    assert _wait_until(
        lambda: session._visual_turns["user_a"].response_id == "resp_1",
        timeout=5,
    )
    assert session._response_to_user_item.get("resp_1") == "user_a"
    assert session._visual_turns["user_b"].response_id is None

    connection.ack_response_created("resp_2")
    assert _wait_until(
        lambda: session._visual_turns["user_b"].response_id == "resp_2",
        timeout=5,
    )
    assert session._response_to_user_item.get("resp_2") == "user_b"
    assert session._visual_turns["user_a"].response_id == "resp_1"

    visual_a = session._visual_turns["user_a"].remote_visual_item_id
    visual_b = session._visual_turns["user_b"].remote_visual_item_id
    assert isinstance(visual_a, str) and isinstance(visual_b, str)
    assert visual_a != visual_b

    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_a",
            transcript="What is on the left?",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id="resp_1",
            transcript="A red object.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_1", status="completed"),
        )
    )
    assert _wait_until(lambda: visual_a in connection.conversation.item.deleted_ids, timeout=5)
    assert visual_b not in connection.conversation.item.deleted_ids

    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_b",
            transcript="What is this?",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id="resp_2",
            transcript="A red square.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_2", status="completed"),
        )
    )
    assert _wait_until(lambda: visual_b in connection.conversation.item.deleted_ids, timeout=5)
    assert connection.conversation.item.deleted_ids.count(visual_a) == 1
    assert connection.conversation.item.deleted_ids.count(visual_b) == 1

    # No cross-turn history contamination / no image material in history.
    for turn in history.turns:
        assert "data:image" not in turn.content
        assert "base64" not in turn.content
        assert REALTIME_VISUAL_FIXED_LABEL not in turn.content
    assert any(t.content == "What is on the left?" for t in history.turns)
    assert any(t.content == "What is this?" for t in history.turns)
    assert any(t.content == "A red object." for t in history.turns)
    assert any(t.content == "A red square." for t in history.turns)

    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
