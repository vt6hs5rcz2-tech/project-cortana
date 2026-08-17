"""Tests for Milestone 26 realtime multimodal session engine."""

from __future__ import annotations

import base64
import inspect
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from queue import Empty, Queue
from typing import Any

from PIL import Image

from src.active_memory import ActiveMemoryContext
from src.camera_capture import CameraCaptureSession, RealtimeVisualFrame
from src.config import (
    MAX_CANCELLED_REALTIME_RESPONSE_IDS,
    MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    REALTIME_VISUAL_FIXED_LABEL,
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
)
from src.conversation import ConversationHistory
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.realtime_conversation_plan import REALTIME_PLAN_BEGIN
from src.speech_delivery import SPEECH_DELIVERY_BEGIN
from src.realtime_idle import REALTIME_IDLE_TIMEOUT_MESSAGE
from src.realtime_multimodal import (
    MULTIMODAL_CAMERA_START_FAILED,
    MULTIMODAL_CONVERSATION_INSTRUCTIONS,
    MULTIMODAL_STARTED_MESSAGE,
    MULTIMODAL_STOPPED_MESSAGE,
    MULTIMODAL_VISUAL_DELETE_FAILED,
    MULTIMODAL_VISUAL_UNUSABLE,
    RealtimeMultimodalSession,
    _MAX_INTERRUPTED_ITEM_IDS,
    _MAX_VISUAL_TURNS,
    build_multimodal_instructions,
    build_multimodal_session_update_payload,
    build_visual_conversation_item,
    run_realtime_multimodal_session,
)
from src.realtime_voice import PlaybackChunk
from src.realtime_voice_input import RealtimeAudioFrame
from src.settings import Settings
from src.vision_normalize import encode_metadata_free_png
from tests.test_visual_policy import (
    LIVE_M26_FORBIDDEN_VISUAL_PHRASES,
    _assert_no_live_visual_refusal_language,
)


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
        self._connection.call_order.append("session.update")
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
                if self._connection.auto_ack_visual:
                    self._connection.socket.push(
                        FakeEvent(type="conversation.item.added", item=fake_item)
                    )
                    self._connection.socket.push(
                        FakeEvent(type="conversation.item.done", item=fake_item)
                    )
            return
        # Bare response.create()
        self.response_creates += 1
        self._connection.call_order.append("response.create")
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
    def __init__(
        self,
        *,
        auto_ack_responses: bool = True,
        auto_ack_visual: bool = True,
    ) -> None:
        self.auto_ack_responses = auto_ack_responses
        self.auto_ack_visual = auto_ack_visual
        self.socket = FakeSocket()
        self._connection = self.socket
        self.session = FakeResource(self)
        self.response = FakeResource(self)
        self.input_audio_buffer = FakeResource(self)
        self.conversation = FakeConversation(self)
        self.closed = False
        self.call_order: list[str] = []
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
        self._last: Any | None = None

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop_id: int, value: float) -> bool:
        del prop_id, value
        return True

    def read(self) -> tuple[bool, Any]:
        if self.frames:
            self._last = self.frames.pop(0)
            return True, self._last
        if self._last is not None:
            return True, self._last
        import numpy as np

        return True, np.zeros((32, 32, 3), dtype="uint8")

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
    conversation_state: ConversationState | None = None,
    transcript_wait_seconds: float | None = None,
    capture_frames: list[Any] | None = None,
    capture: Any | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> tuple[threading.Thread, RealtimeMultimodalSession, dict[str, str], list[str]]:
    FakePlaybackStream.instances.clear()
    lines = printed if printed is not None else []
    result_box: dict[str, str] = {}
    open_order: list[str] = []

    capture_device = (
        capture
        if capture is not None
        else FakeCapture(frames=list(capture_frames) if capture_frames else [_bgr()])
    )
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
                capture_factory=lambda: capture_device,
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
        conversation_state=conversation_state,
        transcript_wait_seconds=transcript_wait_seconds,
        monotonic_fn=monotonic_fn or time.monotonic,
        sleep_fn=sleep_fn or time.sleep,
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


def test_controlled_image_survives_m26_encoding_path() -> None:
    """Obvious local PNG stays valid PNG/data-URI through the M26 insert shape."""
    image = Image.new("RGB", (64, 48), "navy")
    for x in range(20, 44):
        for y in range(12, 36):
            image.putpixel((x, y), (255, 0, 0))
    png, width, height = encode_metadata_free_png(image)
    frame = RealtimeVisualFrame(
        image_bytes=png,
        mime_type="image/png",
        width=width,
        height=height,
        sequence=7,
        captured_at_monotonic=1.0,
    )
    item = build_visual_conversation_item(frame)
    content = item["content"]
    assert isinstance(content, list)
    image_part = content[1]
    assert isinstance(image_part, dict)
    data_uri = str(image_part["image_url"])
    assert data_uri.startswith("data:image/png;base64,")
    encoded = data_uri.split(",", 1)[1]
    decoded = base64.b64decode(encoded)
    assert decoded == png
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    opened = Image.open(BytesIO(decoded))
    assert opened.size == (64, 48)
    assert opened.mode == "RGB"
    extrema = opened.getextrema()
    assert extrema[0][0] < extrema[0][1] or extrema[2][0] < extrema[2][1]


def test_response_create_waits_for_visual_ack() -> None:
    connection = FakeConnection(auto_ack_visual=False)
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What object is visible?",
        )
    )
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
    time.sleep(0.15)
    assert connection.response.response_creates == 0
    connection.socket.push(
        FakeEvent(type="conversation.item.created", item=FakeItem(id="visual_live"))
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    turn = session._visual_turns["user_1"]
    assert turn.remote_visual_item_id == "visual_live"
    assert turn.response_create_sent is True
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_visual_ack_timeout_still_issues_one_response_create() -> None:
    connection = FakeConnection(auto_ack_visual=False)
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    session._visual_ack_wait_seconds = 0.2
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_visual_ack_timeout_late_ack_then_follow_up_turn() -> None:
    """Worker-loop withhold: timeout, late ack, then a valid second visual turn."""
    connection = FakeConnection(auto_ack_visual=False)
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    session._visual_ack_wait_seconds = 0.2
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What object is visible?",
        )
    )
    assert _wait_until(
        lambda: bool(connection.conversation.item.created_items),
        timeout=5,
    )
    assert connection.response.response_creates == 0
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1

    connection.socket.push(
        FakeEvent(type="conversation.item.created", item=FakeItem(id="visual_1_late"))
    )
    time.sleep(0.15)
    first = session._visual_turns.get("user_1")
    if first is not None:
        assert first.remote_visual_item_id != "visual_1_late"
    assert session._current_remote_visual_item_id != "visual_1_late"
    assert state.visual_context_ref_id != "visual_1_late"
    assert connection.response.response_creates == 1

    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_2")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_2",
            transcript="What color is that object?",
        )
    )
    assert _wait_until(
        lambda: len(connection.conversation.item.created_items) >= 2,
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(type="conversation.item.created", item=FakeItem(id="visual_2"))
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    second = session._visual_turns.get("user_2")
    assert second is not None
    assert second.remote_visual_item_id == "visual_2"
    assert session._current_remote_visual_item_id == "visual_2"
    assert connection.response.response_creates == 2
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


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
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What am I looking at?",
        )
    )
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
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_2",
            transcript="What is this?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates >= 2, timeout=5)
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
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u1",
            transcript="Describe this",
        )
    )
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
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u1",
            transcript="What is that?",
        )
    )
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

    # Turn A then Turn B. Bind visuals first; M28 waits for each finalized
    # transcript before the existing manual response.create.
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_a")
    )
    assert _wait_until(
        lambda: session._visual_turns.get("user_a") is not None,
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.committed", item_id="user_b")
    )
    assert _wait_until(
        lambda: session._visual_turns.get("user_b") is not None,
        timeout=5,
    )
    assert connection.response.response_creates == 0
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_a",
            transcript="What is on the left?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates >= 1, timeout=5)
    assert _wait_until(
        lambda: session._visual_turns["user_a"].response_create_sent
        and session._visual_turns["user_a"].remote_visual_item_id is not None,
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_b",
            transcript="What is this?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates >= 2, timeout=5)
    assert _wait_until(
        lambda: session._visual_turns["user_b"].response_create_sent
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


# ---------------------------------------------------------------------------
# M27-F1: lightweight per-completed-turn conversational-intelligence tests.
# ---------------------------------------------------------------------------


def test_finalized_multimodal_turn_updates_state_without_extra_response() -> None:
    """C + H: a completed multimodal turn observes state locally and issues

    exactly one response.create() — the local hook never competes with the
    provider for response generation.
    """
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)

    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="How should I organize my desk photos?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates >= 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    active = session._active_response_id
    assert isinstance(active, str)
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id=active,
            transcript="You could group them by date or by event.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id=active, status="completed"),
        )
    )
    assert _wait_until(lambda: state.active_goal is not None, timeout=5)

    assert state.recent_interaction_mode == "multimodal"
    assert "desk photos" in (state.active_goal or "").casefold()
    # Exactly one response.create() call occurred for this one turn — the
    # local observe hook did not trigger a second/competing response.
    assert connection.response.response_creates == 1
    assert connection.response.cancels == []

    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False


# ---------------------------------------------------------------------------
# M27-F2: stale visual-referent cleanup race regression.
# ---------------------------------------------------------------------------


def test_late_visual_ack_during_shutdown_cannot_resurrect_stale_ref() -> None:
    """A late in-flight visual item write during the session-worker join

    window must not leave ``visual_context_ref_id`` populated after
    ``_cleanup()`` returns.
    """
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    session = RealtimeMultimodalSession(
        settings=_settings(),
        client=object(),  # type: ignore[arg-type]
        conversation_history=history,
        active_memory_context=ActiveMemoryContext(),
        logger_=logging.getLogger("test"),
        connect_factory=lambda: FakeManager(connection),
        conversation_state=state,
    )
    # Simulate: a visual item was authorized and acked before shutdown began.
    session._current_remote_visual_item_id = "visual_x"
    state.set_visual_context_ref("visual_x")
    session._connection = connection

    class LateWritingThread:
        """Simulates the session worker processing one more in-flight ack

        during its own shutdown window, resurrecting a (now stale) ref,
        before the join call observes it has actually finished.
        """

        def join(self, timeout: float | None = None) -> None:
            del timeout
            state.set_visual_context_ref("late_resurrected_item")

        def is_alive(self) -> bool:
            return False

    session._session_thread = LateWritingThread()  # type: ignore[assignment]
    session._playback_thread = None
    session._cleanup()

    # The final unconditional clear runs only after the session worker join,
    # so it always wins over any late in-flight write during that window.
    assert state.visual_context_ref_id is None


# ---------------------------------------------------------------------------
# Milestone 28 realtime conversational planning
# ---------------------------------------------------------------------------


def test_m26_does_not_create_response_until_final_transcript() -> None:
    """I: speech_stopped binds visual but waits for the finalized transcript."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: session._visual_turns.get("user_1") is not None, timeout=5)
    time.sleep(0.15)
    assert connection.response.response_creates == 0
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What is that?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(
        lambda: any(
            REALTIME_PLAN_BEGIN in str(update.get("instructions", ""))
            for update in connection.session.updates
        ),
        timeout=5,
    )
    latest = connection.session.updates[-1]
    turn = latest["audio"]["input"]["turn_detection"]
    assert turn["create_response"] is False
    assert turn["interrupt_response"] is True
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_what_is_that_uses_live_visual_and_does_not_authorize() -> None:
    """G + I: visual referent is planned before the single response.create."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What is that?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    plan = session._plans_by_item.get("user_1")
    assert plan is not None
    assert plan.visual_referent_resolved is True
    assert plan.authorizes_privileged_action is False
    assert connection.response.response_creates == 1
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_planning_failure_still_creates_exactly_one_response() -> None:
    """L: planning exceptions do not skip or duplicate the existing response.create."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)

    class Broken(ConversationIntelligence):
        def interpret(self, *args: object, **kwargs: object) -> Any:
            raise RuntimeError("boom")

    session._conversation_intelligence = Broken()
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="How should I organize my desk photos?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False


def test_m26_partial_transcript_does_not_plan_or_create() -> None:
    """M: ignored transcription deltas never alter planning or trigger response.create."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.delta",
            item_id="user_1",
            transcript="what is th",
        )
    )
    time.sleep(0.15)
    assert connection.response.response_creates == 0
    assert "user_1" not in session._plans_by_item
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_has_exactly_one_response_create_site() -> None:
    source = inspect.getsource(RealtimeMultimodalSession)
    assert source.count("connection.response.create()") == 1


def test_m26_transcript_before_timeout_applies_plan_then_one_create() -> None:
    """F1-A: finalized transcript before the deadline plans, injects, then creates once."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="How should I organize my desk photos?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1
    create_at = connection.call_order.index("response.create")
    assert connection.call_order[create_at - 1] == "session.update"
    plan_updates = [
        update
        for update in connection.session.updates
        if REALTIME_PLAN_BEGIN in str(update.get("instructions", ""))
    ]
    assert plan_updates
    assert "desk photos" in (state.active_goal or "").casefold()
    turn = session._visual_turns["user_1"]
    assert turn.transcript_fallback is False
    assert turn.response_create_sent is True
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_missing_transcript_falls_back_without_fabricating_plan() -> None:
    """F1-B: bounded fallback still creates exactly once with no transcript-derived plan."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    started = time.monotonic()
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert connection.response.response_creates == 1
    assert "user_1" not in session._plans_by_item
    assert state.active_goal is None
    assert session._visual_turns["user_1"].transcript_fallback is True
    create_at = connection.call_order.index("response.create")
    updates_before = connection.call_order[:create_at].count("session.update")
    fallback_update = connection.session.updates[updates_before - 1]
    assert REALTIME_PLAN_BEGIN not in str(fallback_update.get("instructions", ""))
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_transcript_wait_is_bounded_by_configured_timeout() -> None:
    """Response-start wait is bounded by the configured fallback, not unbounded."""
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    assert session._transcript_wait_seconds == MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    started = time.monotonic()
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.0
    assert elapsed < 2.0
    assert connection.response.response_creates == 1
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_late_transcript_after_fallback_does_not_create_or_overwrite() -> None:
    """F1-C: late transcript cannot create a second response or overwrite a newer goal."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_2")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_2",
            transcript="How do backups work?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert _wait_until(lambda: "backup" in (state.active_goal or "").casefold(), timeout=5)
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="schedule the briefing for Monday",
        )
    )
    time.sleep(0.2)
    assert connection.response.response_creates == 2
    assert "backup" in (state.active_goal or "").casefold()
    assert "briefing" not in (state.active_goal or "").casefold()
    assert "user_1" not in session._plans_by_item
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_interrupted_turn_before_timeout_does_not_create() -> None:
    """F1-D: a stale waiting turn ignores timeout/fallback and creates no response."""
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: session._visual_turns.get("user_1") is not None, timeout=5)
    session._visual_turns["user_1"].stale = True
    session._clear_transcript_wait("user_1")
    time.sleep(MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS + 0.4)
    assert connection.response.response_creates == 0
    turn = session._visual_turns.get("user_1")
    assert turn is None or (
        turn.prepare_enqueued is False and turn.response_create_sent is False
    )
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_cleanup_before_timeout_creates_no_response_and_clears_deadlines() -> None:
    """F1-E: shutdown cancels pending transcript waits; no late create or leak."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: session._visual_turns.get("user_1") is not None, timeout=5)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)
    assert thread.is_alive() is False
    time.sleep(MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS + 0.4)
    assert connection.response.response_creates == 0
    assert session._transcript_deadlines == {}
    assert state.active_goal is None


def test_m26_rapid_double_turn_timeouts_are_independent() -> None:
    """F1-F: each turn has its own deadline; first timeout cannot create for the second."""
    connection = FakeConnection(auto_ack_responses=False)
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_a")
    )
    assert _wait_until(lambda: session._visual_turns.get("user_a") is not None, timeout=5)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.committed", item_id="user_b")
    )
    assert _wait_until(lambda: session._visual_turns.get("user_b") is not None, timeout=5)
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert session._visual_turns["user_a"].response_create_sent is True
    assert session._visual_turns["user_b"].response_create_sent is True
    assert session._visual_turns["user_a"].transcript_fallback is True
    assert session._visual_turns["user_b"].transcript_fallback is True
    assert connection.response.pending_response_ids[:2] == ["resp_1", "resp_2"]
    connection.ack_response_created("resp_1")
    assert _wait_until(
        lambda: session._visual_turns["user_a"].response_id == "resp_1",
        timeout=5,
    )
    assert session._visual_turns["user_b"].response_id is None
    connection.ack_response_created("resp_2")
    assert _wait_until(
        lambda: session._visual_turns["user_b"].response_id == "resp_2",
        timeout=5,
    )
    assert session._response_to_user_item.get("resp_1") == "user_a"
    assert session._response_to_user_item.get("resp_2") == "user_b"
    assert connection.response.response_creates == 2
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_planning_failure_after_transcript_still_one_create() -> None:
    """F1-G: planning failure after a normal transcript still uses one response.create."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)

    class Broken(ConversationIntelligence):
        def interpret(self, *args: object, **kwargs: object) -> Any:
            raise RuntimeError("boom")

    session._conversation_intelligence = Broken()
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="How should I organize my desk photos?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1
    assert "user_1" not in session._plans_by_item
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_timeout_fallback_does_not_authorize_privileged_action() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert session._plans_by_item == {}
    assert state.active_goal is None
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_duplicate_text_plans_associate_by_item_id() -> None:
    """F2-A/D: two multimodal 'continue' turns keep distinct plans and FIFO linkage."""
    connection = FakeConnection(auto_ack_responses=False)
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_a")
    )
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.committed", item_id="user_b")
    )
    assert _wait_until(
        lambda: session._visual_turns.get("user_a") is not None
        and session._visual_turns.get("user_b") is not None,
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_a",
            transcript="continue",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_b",
            transcript="continue",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert _wait_until(
        lambda: "user_a" in session._plans_by_item and "user_b" in session._plans_by_item,
        timeout=5,
    )
    assert session._plans_by_item["user_a"] is not session._plans_by_item["user_b"]
    connection.ack_response_created("resp_1")
    assert _wait_until(
        lambda: session._visual_turns["user_a"].response_id == "resp_1",
        timeout=5,
    )
    assert session._visual_turns["user_b"].response_id is None
    connection.ack_response_created("resp_2")
    assert _wait_until(
        lambda: session._visual_turns["user_b"].response_id == "resp_2",
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id="resp_2",
            transcript="Later continue.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_2", status="completed"),
        )
    )
    assert _wait_until(lambda: "user_b" not in session._plans_by_item, timeout=5)
    assert "user_a" in session._plans_by_item
    connection.socket.push(
        FakeEvent(
            type="response.output_audio_transcript.done",
            response_id="resp_1",
            transcript="Earlier continue.",
        )
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_1", status="completed"),
        )
    )
    assert _wait_until(lambda: "user_a" not in session._plans_by_item, timeout=5)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_interrupted_duplicate_text_plan_cannot_be_consumed() -> None:
    """F2-C: interrupting one identical-text multimodal turn drops only that plan."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_a")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_a",
            transcript="continue",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_b")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_b",
            transcript="continue",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert _wait_until(
        lambda: "user_a" in session._plans_by_item and "user_b" in session._plans_by_item,
        timeout=5,
    )
    connection.socket.push(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id="resp_1", status="cancelled"),
        )
    )
    assert _wait_until(lambda: "user_a" not in session._plans_by_item, timeout=5)
    assert "user_b" in session._plans_by_item
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_no_transcript_equality_plan_identity() -> None:
    """F2-F: no leftover transcript-equality identity mechanism."""
    assert not hasattr(RealtimeMultimodalSession, "_take_plan_for_user_text")
    source = inspect.getsource(RealtimeMultimodalSession)
    assert "_take_plan_for_user_text" not in source
    assert "original_user_text == " not in source


def test_m26_speech_delivery_is_included_before_single_response_create() -> None:
    """M29 delivery guidance rides the existing M28 session.update path."""
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="user_1",
            transcript="What time is the meeting?",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1
    latest = connection.session.updates[-1]
    turn = latest["audio"]["input"]["turn_detection"]
    assert turn["create_response"] is False
    assert turn["interrupt_response"] is True
    instructions = str(latest["instructions"])
    assert REALTIME_PLAN_BEGIN in instructions
    assert SPEECH_DELIVERY_BEGIN in instructions
    assert "never authorizes" in instructions.casefold()
    create_at = connection.call_order.index("response.create")
    assert connection.call_order.count("response.create") == 1
    assert create_at == connection.call_order.index("response.create")
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_m26_fallback_still_omits_transcript_derived_delivery_guidance() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        transcript_wait_seconds=MIN_REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS,
    )
    assert _wait_until(lambda: session.microphone_opened)
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="user_1")
    )
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    create_at = connection.call_order.index("response.create")
    updates_before = connection.call_order[:create_at].count("session.update")
    fallback_update = connection.session.updates[updates_before - 1]
    instructions = str(fallback_update.get("instructions", ""))
    assert REALTIME_PLAN_BEGIN not in instructions
    assert SPEECH_DELIVERY_BEGIN not in instructions
    assert connection.response.response_creates == 1
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_multimodal_instructions_allow_ordinary_objects_keep_person_id_ban() -> None:
    text = build_multimodal_instructions(active_memory_context=None)
    folded = text.casefold()
    assert "face recognition" in folded
    assert "person identification" in folded or "identify real people" in folded
    assert "camera images are valid perceptual input" in folded
    assert "cannot authorize actions" in folded
    assert "untrusted" not in folded
    assert "cannot trust" not in folded
    assert "not confirming from the visual" not in folded
    assert REALTIME_VISUAL_FIXED_LABEL == (
        "Current camera image for this spoken user turn."
    )
    assert "authority" not in REALTIME_VISUAL_FIXED_LABEL.casefold()
    assert "not interpret" not in MULTIMODAL_CONVERSATION_INSTRUCTIONS.casefold()


def test_speech_started_during_playback_clears_old_audio() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    pcm = b"\x01\x00" * 80
    session._responding = True
    session._active_response_id = "resp_old"
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id="resp_old", pcm_bytes=pcm)
    )
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id="resp_old", pcm_bytes=pcm)
    )
    with session._playback_bytes_lock:
        session._playback_bytes_queued = len(pcm) * 2
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_echo")
    )
    assert session._playback_bytes_queued == 0
    assert session._is_cancelled("resp_old")
    leftover: list[PlaybackChunk] = []
    while True:
        try:
            item = session._playback_queue.get_nowait()
        except Empty:
            break
        if item is not None:
            leftover.append(item)
    assert leftover == []
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_cancelled_deltas_do_not_regrow_playback_queue() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    pcm = b"\x02\x00" * 80
    session._responding = True
    session._active_response_id = "resp_live"
    session._mark_response_cancelled("resp_live")
    encoded = base64.b64encode(pcm).decode("ascii")
    for _ in range(8):
        session._on_audio_delta(
            FakeEvent(
                type="response.output_audio.delta",
                response_id="resp_live",
                delta=encoded,
            )
        )
    assert session._playback_bytes_queued == 0
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_repeated_false_interruptions_stay_bounded() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _lines = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    pcm = b"\x03\x00" * 40
    encoded = base64.b64encode(pcm).decode("ascii")
    for index in range(40):
        response_id = f"resp_loop_{index}"
        session._responding = True
        session._active_response_id = response_id
        session._playback_queue.put_nowait(
            PlaybackChunk(response_id=response_id, pcm_bytes=pcm)
        )
        with session._playback_bytes_lock:
            session._playback_bytes_queued = len(pcm)
        session._on_speech_started(
            FakeEvent(
                type="input_audio_buffer.speech_started",
                item_id=f"u_loop_{index}",
            )
        )
        session._on_audio_delta(
            FakeEvent(
                type="response.output_audio.delta",
                response_id=response_id,
                delta=encoded,
            )
        )
    assert session._playback_bytes_queued == 0
    assert len(session._cancelled_set) <= MAX_CANCELLED_REALTIME_RESPONSE_IDS
    assert len(session._interrupted_item_ids) <= _MAX_INTERRUPTED_ITEM_IDS
    assert len(session._visual_turns) <= _MAX_VISUAL_TURNS
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_accidental_fragment_during_playback_does_not_create_response() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    lines: list[str] = []
    thread, session, _result_box, printed = _run_session(
        connection, history, printed=lines
    )
    assert _wait_until(lambda: session.microphone_opened)
    creates_before = connection.response.response_creates
    session._responding = True
    session._active_response_id = "resp_live"
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_noise")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_noise")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_noise",
            transcript="Acaba.",
        )
    )
    time.sleep(0.3)
    assert connection.response.response_creates == creates_before
    assert not any("Acaba" in line for line in printed)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_short_barge_in_stop_is_not_suppressed() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    lines: list[str] = []
    thread, session, _result_box, printed = _run_session(
        connection, history, printed=lines
    )
    assert _wait_until(lambda: session.microphone_opened)
    creates_before = connection.response.response_creates
    session._responding = True
    session._active_response_id = "resp_live"
    session._assistant_audio_started_at = session._monotonic()
    pcm = b"\x01\x00" * 80
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id="resp_live", pcm_bytes=pcm)
    )
    with session._playback_bytes_lock:
        session._playback_bytes_queued = len(pcm)
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_stop")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_stop")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_stop",
            transcript="Stop.",
        )
    )
    assert _wait_until(
        lambda: any("Heard) Stop" in line for line in printed),
        timeout=5,
    )
    time.sleep(0.3)
    assert connection.response.response_creates == creates_before
    assert session._playback_bytes_queued == 0
    assert session._responding is False
    assert session._is_cancelled("resp_live")
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_privacy_placeholder_does_not_create_visual_item() -> None:
    from tests.test_visual_frame_quality import _bgr_from_image, _privacy_placeholder

    connection = FakeConnection()
    history = ConversationHistory()
    placeholder = _bgr_from_image(_privacy_placeholder())
    thread, session, _result_box, lines = _run_session(
        connection,
        history,
        capture_frames=[placeholder, placeholder, placeholder],
    )
    assert _wait_until(lambda: session.microphone_opened)
    camera = session._camera
    assert camera is not None
    assert _wait_until(lambda: camera.get_usable_fresh_frame() is None, timeout=5)
    _drive_visual_turn(connection, "user_ph", "What object is visible?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    image_items = [
        item
        for item in connection.conversation.item.created_items
        if isinstance(item, dict)
        and any(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in (item.get("content") or [])
            if isinstance(part, dict)
        )
    ]
    assert image_items == []
    assert any(MULTIMODAL_VISUAL_UNUSABLE in line for line in lines)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_current_visual_required_does_not_reuse_stale_visual_ref() -> None:
    from tests.test_visual_frame_quality import _bgr_from_image, _privacy_placeholder

    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    capture = FakeCapture(frames=[_bgr(), _bgr(), _bgr(), _bgr()])
    thread, session, _result_box, lines = _run_session(
        connection,
        history,
        conversation_state=state,
        capture=capture,
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What object am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    prior_ref = state.visual_context_ref_id
    assert prior_ref is not None

    placeholder = _bgr_from_image(_privacy_placeholder())
    capture.frames = [placeholder, placeholder, placeholder, placeholder]
    capture._last = placeholder
    camera = session._camera
    assert camera is not None
    assert _wait_until(lambda: camera.get_usable_fresh_frame() is None, timeout=5)

    _drive_visual_turn(connection, "u_now", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    assert any(MULTIMODAL_VISUAL_UNUSABLE in line for line in lines)
    plan = session._plans_by_item.get("u_now")
    assert plan is not None
    assert plan.visual_referent_resolved is False
    assert plan.resolved_follow_up is not None
    follow_up = plan.resolved_follow_up.casefold()
    assert "no usable camera image" in follow_up
    assert "current camera image is relevant" not in follow_up
    assert state.visual_context_ref_id == prior_ref
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_prior_reference_after_unusable_frame_keeps_previous_visual() -> None:
    from tests.test_visual_frame_quality import _bgr_from_image, _privacy_placeholder

    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    capture = FakeCapture(frames=[_bgr(), _bgr(), _bgr(), _bgr()])
    thread, session, _result_box, _lines = _run_session(
        connection,
        history,
        conversation_state=state,
        capture=capture,
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What object am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    prior_ref = state.visual_context_ref_id
    assert prior_ref is not None

    placeholder = _bgr_from_image(_privacy_placeholder())
    capture.frames = [placeholder, placeholder, placeholder, placeholder]
    capture._last = placeholder
    camera = session._camera
    assert camera is not None
    assert _wait_until(lambda: camera.get_usable_fresh_frame() is None, timeout=5)

    _drive_visual_turn(connection, "u_was", "What color was it?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    plan = session._plans_by_item.get("u_was")
    assert plan is not None
    assert plan.visual_referent_resolved is True
    assert plan.visual_context_ref_id == prior_ref
    assert plan.resolved_follow_up is not None
    assert "previous camera image is relevant" in plan.resolved_follow_up.casefold()
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def _created_visual_images(connection: FakeConnection) -> list[Any]:
    images: list[Any] = []
    for item in connection.conversation.item.created_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in content
        ):
            images.append(item)
    return images


def _drive_visual_turn(
    connection: FakeConnection,
    item_id: str,
    transcript: str,
) -> None:
    connection.socket.push(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id=item_id)
    )
    connection.socket.push(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            transcript=transcript,
        )
    )


def test_explicit_stop_variants_do_not_create_response() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u1", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    first_creates = connection.response.response_creates
    pcm = b"\x01\x00" * 80
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id=session._active_response_id or "resp_1", pcm_bytes=pcm)
    )
    with session._playback_bytes_lock:
        session._playback_bytes_queued = len(pcm)

    for index, phrase in enumerate(("Stop talking.", "Never mind.")):
        item_id = f"u_stop_{index}"
        session._on_speech_started(
            FakeEvent(type="input_audio_buffer.speech_started", item_id=item_id)
        )
        session._on_user_turn_boundary(
            FakeEvent(type="input_audio_buffer.speech_stopped", item_id=item_id)
        )
        session._on_user_transcript(
            FakeEvent(
                type="conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                transcript=phrase,
            )
        )
        time.sleep(0.15)
        assert connection.response.response_creates == first_creates, phrase
        assert session._playback_bytes_queued == 0
        assert session._responding is False
        assert any("Heard)" in line and phrase.rstrip(".") in line for line in printed)

    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_non_command_stop_phrases_still_create() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_dont", "Don't stop.")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    _drive_visual_turn(connection, "u_sign", "What is a stop sign?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_after_stop_next_visual_turn_still_creates_once() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u1", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_stop")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_stop")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_stop",
            transcript="Stop.",
        )
    )
    time.sleep(0.2)
    assert connection.response.response_creates == 1
    assert session._playback_bytes_queued == 0
    _drive_visual_turn(connection, "u_next", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert connection.response.response_creates == 2
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_stale_audio_does_not_resume_after_stop() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    session._responding = True
    session._active_response_id = "resp_live"
    pcm = b"\x04\x00" * 40
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id="resp_live", pcm_bytes=pcm)
    )
    with session._playback_bytes_lock:
        session._playback_bytes_queued = len(pcm)
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_stop")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_stop",
            transcript="Stop!",
        )
    )
    assert session._playback_bytes_queued == 0
    leftover: list[PlaybackChunk] = []
    while True:
        try:
            item = session._playback_queue.get_nowait()
        except Empty:
            break
        if item is not None:
            leftover.append(item)
    assert leftover == []
    encoded = base64.b64encode(pcm).decode("ascii")
    session._on_audio_delta(
        FakeEvent(
            type="response.output_audio.delta",
            response_id="resp_live",
            delta=encoded,
        )
    )
    assert session._playback_bytes_queued == 0
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_live_m26_payload_for_object_question_is_positive_visual_policy() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
    )
    assert _wait_until(lambda: session.microphone_opened)
    assert history.turns == []
    initial = str(connection.session.updates[0].get("instructions", ""))
    assert "camera images are valid perceptual input" in initial.casefold()
    assert "untrusted" not in initial.casefold()
    _assert_no_live_visual_refusal_language(initial)

    _drive_visual_turn(connection, "u_obj", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert connection.response.response_creates == 1

    image_items = [
        item
        for item in connection.conversation.item.created_items
        if isinstance(item, dict)
        and any(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in (item.get("content") or [])
            if isinstance(part, dict)
        )
    ]
    assert image_items
    label_parts = [
        part.get("text")
        for part in image_items[0]["content"]
        if isinstance(part, dict) and part.get("type") == "input_text"
    ]
    assert label_parts == [REALTIME_VISUAL_FIXED_LABEL]
    label = str(label_parts[0]).casefold()
    for word in ("untrusted", "authority", "policy", "trust", "rely"):
        assert word not in label, word

    planned = str(connection.session.updates[-1].get("instructions", ""))
    planned_folded = planned.casefold()
    assert "camera images are valid perceptual input" in planned_folded
    assert "current camera image is relevant" in planned_folded
    assert "do not identify people" in planned_folded
    assert "cannot authorize actions" in planned_folded
    assert "untrusted" not in planned_folded
    for phrase in LIVE_M26_FORBIDDEN_VISUAL_PHRASES:
        assert phrase not in planned_folded, phrase

    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def _live_visual_user_item(session: RealtimeMultimodalSession) -> str:
    for user_item_id, turn in session._visual_turns.items():
        if turn.response_create_sent and not turn.stale:
            return user_item_id
    raise AssertionError("no live visual turn")


def test_accidental_fragment_recovers_paused_visual_response() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_obj", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    live_item = _live_visual_user_item(session)
    remote_id = session._visual_turns[live_item].remote_visual_item_id
    first_response = session._active_response_id
    assert first_response is not None
    pcm = b"\x01\x00" * 80
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id=first_response, pcm_bytes=pcm)
    )
    with session._playback_bytes_lock:
        session._playback_bytes_queued = len(pcm)
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_echo")
    )
    assert session._playback_bytes_queued == 0
    assert session._is_cancelled(first_response)
    assert session._visual_turns[live_item].stale is False
    session._on_response_done(
        FakeEvent(
            type="response.done",
            response=FakeResponse(id=first_response, status="cancelled"),
        )
    )
    assert remote_id not in connection.conversation.item.deleted_ids
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_echo")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_echo",
            transcript="Acaba.",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert not any("Acaba" in line for line in printed)
    recovered = session._visual_turns.get(live_item)
    assert recovered is not None
    assert recovered.stale is False
    assert recovered.response_create_sent is True
    echo = session._visual_turns.get("u_echo")
    assert echo is None or echo.stale is True
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_stop_after_speech_started_keeps_assistant_silent() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_obj", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    first_creates = connection.response.response_creates
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_stop")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_stop")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_stop",
            transcript="Stop.",
        )
    )
    time.sleep(0.3)
    assert connection.response.response_creates == first_creates
    assert session._responding is False
    assert session._pending_barge is None
    assert any("Heard) Stop" in line for line in printed)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_genuine_barge_in_replaces_paused_visual_response() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_obj", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    live_item = _live_visual_user_item(session)
    first_response = session._active_response_id
    assert first_response is not None
    pcm = b"\x02\x00" * 40
    session._playback_queue.put_nowait(
        PlaybackChunk(response_id=first_response, pcm_bytes=pcm)
    )
    with session._playback_bytes_lock:
        session._playback_bytes_queued = len(pcm)
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_ram")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_ram")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_ram",
            transcript="Tell me about RAM.",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    old = session._visual_turns.get(live_item)
    assert old is None or old.stale is True
    encoded = base64.b64encode(pcm).decode("ascii")
    session._on_audio_delta(
        FakeEvent(
            type="response.output_audio.delta",
            response_id=first_response,
            delta=encoded,
        )
    )
    assert session._playback_bytes_queued == 0
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_false_speech_started_without_transcript_recovers() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(
        connection, history, transcript_wait_seconds=0.25
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_obj", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    live_item = _live_visual_user_item(session)
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_ghost")
    )
    assert session._pending_barge is not None
    session._pending_barge.deadline_monotonic = session._monotonic() - 0.01
    session._expire_pending_barge_waits()
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    recovered = session._visual_turns.get(live_item)
    assert recovered is not None
    assert recovered.stale is False
    assert session._pending_barge is None
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_recovered_visual_response_stays_on_same_turn() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_obj", "What object am I holding up?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert _wait_until(lambda: session._active_response_id is not None, timeout=5)
    live_item = _live_visual_user_item(session)
    remote_before = session._visual_turns[live_item].remote_visual_item_id
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_echo")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_echo")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_echo",
            transcript="um",
        )
    )
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    recovered = session._visual_turns[live_item]
    assert recovered.remote_visual_item_id == remote_before
    assert recovered.stale is False
    image_creates = [
        item
        for item in connection.conversation.item.created_items
        if isinstance(item, dict)
        and any(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in (item.get("content") or [])
            if isinstance(part, dict)
        )
    ]
    assert len(image_creates) == 1
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_visual_question_inserts_image() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What object am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    plan = session._plans_by_item.get("u_hold")
    assert plan is not None
    assert plan.visual_relevant is True
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_color_follow_up_keeps_visual_context() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What object am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    _drive_visual_turn(connection, "u_color", "What color is it?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert len(_created_visual_images(connection)) == 2
    plan = session._plans_by_item.get("u_color")
    assert plan is not None
    assert plan.visual_relevant is True
    assert "repeating the full previous object description" in plan.avoid_phrases
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_non_visual_questions_do_not_create_image() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_mon", "What day comes after Monday?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    _drive_visual_turn(connection, "u_math", "What is 2 + 2?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    _drive_visual_turn(connection, "u_py", "Tell me about Python.")
    assert _wait_until(lambda: connection.response.response_creates == 3, timeout=5)
    assert _created_visual_images(connection) == []
    monday = session._plans_by_item.get("u_mon")
    assert monday is not None
    assert monday.visual_relevant is False
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_visual_then_nonvisual_does_not_inherit_image_insert() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    prior_ref = state.visual_context_ref_id
    _drive_visual_turn(connection, "u_mon", "What day comes after Monday?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    monday = session._plans_by_item.get("u_mon")
    assert monday is not None
    assert monday.visual_relevant is False
    assert state.visual_context_ref_id == prior_ref
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_scene_change_this_one_inserts_new_image() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    thread, session, _result_box, _printed = _run_session(connection, history)
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    _drive_visual_turn(connection, "u_now", "What am I showing you now?")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    _drive_visual_turn(connection, "u_this", "What color is this one?")
    assert _wait_until(lambda: connection.response.response_creates == 3, timeout=5)
    assert len(_created_visual_images(connection)) == 3
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_truncated_its_during_playback_does_not_create_response() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    lines: list[str] = []
    thread, session, _result_box, printed = _run_session(
        connection, history, printed=lines
    )
    assert _wait_until(lambda: session.microphone_opened)
    creates_before = connection.response.response_creates
    session._responding = True
    session._active_response_id = "resp_live"
    session._assistant_audio_started_at = session._monotonic()
    session._on_speech_started(
        FakeEvent(type="input_audio_buffer.speech_started", item_id="u_its")
    )
    session._on_user_turn_boundary(
        FakeEvent(type="input_audio_buffer.speech_stopped", item_id="u_its")
    )
    session._on_user_transcript(
        FakeEvent(
            type="conversation.item.input_audio_transcription.completed",
            item_id="u_its",
            transcript="It's...",
        )
    )
    time.sleep(0.3)
    assert connection.response.response_creates == creates_before
    assert not any("It's" in line for line in printed)
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


def test_legitimate_its_turns_are_not_suppressed() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    thread, session, _result_box, _printed = _run_session(
        connection, history, conversation_state=state
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_broken", "It's broken.")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    _drive_visual_turn(connection, "u_tue", "It's Tuesday.")
    assert _wait_until(lambda: connection.response.response_creates == 2, timeout=5)
    assert _created_visual_images(connection) == []
    session.request_stop(error_type="cancelled")
    thread.join(timeout=5)


class _FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _yield_sleep(_seconds: float) -> None:
    time.sleep(0.001)


def test_multimodal_rejected_noise_does_not_upload_or_reset_idle() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    clock = _FakeClock()
    thread, session, result_box, printed = _run_session(
        connection,
        history,
        monotonic_fn=clock,
        sleep_fn=_yield_sleep,
    )
    assert _wait_until(lambda: session.microphone_opened)
    creates_before = connection.response.response_creates
    clock.advance(9.0)
    for item_id, transcript in (
        ("u_empty", ""),
        ("u_space", "   "),
        ("u_punct", "..."),
        ("u_its", "It's..."),
    ):
        session._on_user_transcript(
            FakeEvent(
                type="conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                transcript=transcript,
            )
        )
    time.sleep(0.2)
    assert connection.response.response_creates == creates_before
    assert _created_visual_images(connection) == []
    assert not any("Heard" in line for line in printed)
    clock.advance(1.1)
    thread.join(timeout=5)
    assert result_box["message"] == REALTIME_IDLE_TIMEOUT_MESSAGE


def test_multimodal_visual_question_resets_idle_and_inserts_image() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    clock = _FakeClock()
    thread, session, result_box, _printed = _run_session(
        connection,
        history,
        monotonic_fn=clock,
        sleep_fn=_yield_sleep,
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    assert len(_created_visual_images(connection)) == 1
    clock.advance(9.0)
    time.sleep(0.05)
    assert session._stop.is_set() is False
    clock.advance(1.1)
    thread.join(timeout=5)
    assert result_box["message"] == REALTIME_IDLE_TIMEOUT_MESSAGE


def test_multimodal_idle_timeout_stops_camera_and_clears_visual_state() -> None:
    connection = FakeConnection()
    history = ConversationHistory()
    state = ConversationState()
    clock = _FakeClock()
    thread, session, result_box, _printed = _run_session(
        connection,
        history,
        conversation_state=state,
        monotonic_fn=clock,
        sleep_fn=_yield_sleep,
    )
    assert _wait_until(lambda: session.microphone_opened)
    _drive_visual_turn(connection, "u_hold", "What am I holding?")
    assert _wait_until(lambda: connection.response.response_creates == 1, timeout=5)
    cleanup_calls = {"n": 0}
    original_cleanup = session._cleanup

    def tracked_cleanup() -> None:
        cleanup_calls["n"] += 1
        original_cleanup()

    session._cleanup = tracked_cleanup  # type: ignore[method-assign]
    clock.advance(10.1)
    thread.join(timeout=5)
    assert result_box["message"] == REALTIME_IDLE_TIMEOUT_MESSAGE
    assert cleanup_calls["n"] == 1
    assert connection.closed is True
    assert session._camera is None
    assert session._visual_turns == {}
    assert state.visual_context_ref_id is None
    assert session.state.value == "closed"


def test_multimodal_restart_after_idle_timeout() -> None:
    clock = _FakeClock()
    connection = FakeConnection()
    history = ConversationHistory()
    thread, _session, result_box, _printed = _run_session(
        connection,
        history,
        monotonic_fn=clock,
        sleep_fn=_yield_sleep,
    )
    assert _wait_until(lambda: _session.microphone_opened)
    clock.advance(10.1)
    thread.join(timeout=5)
    assert result_box["message"] == REALTIME_IDLE_TIMEOUT_MESSAGE

    connection2 = FakeConnection()
    thread2, session2, result_box2, _printed2 = _run_session(
        connection2,
        history,
        monotonic_fn=clock,
        sleep_fn=_yield_sleep,
    )
    assert _wait_until(lambda: session2.microphone_opened)
    _drive_visual_turn(connection2, "u_hold", "What am I holding?")
    assert _wait_until(lambda: connection2.response.response_creates == 1, timeout=5)
    assert len(_created_visual_images(connection2)) == 1
    session2.request_stop(error_type="cancelled")
    thread2.join(timeout=5)
    assert result_box2["message"] == MULTIMODAL_STOPPED_MESSAGE
