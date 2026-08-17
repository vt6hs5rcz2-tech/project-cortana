"""Live M26 camera-salience isolation matrix.

Investigation only. Not imported by pytest or application startup.
Does not change production M26 routing, Stop, or visual-policy wording.
Never prints OPENAI_API_KEY, image bytes, or audio bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tempfile
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_capture import CameraCaptureSession, RealtimeVisualFrame
from src.config import (
    REALTIME_VISUAL_FIXED_LABEL,
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
)
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.openai_client import create_openai_client
from src.realtime_conversation_plan import (
    format_realtime_plan_instructions,
    plan_realtime_turn,
    speech_delivery_plan_from_realtime,
)
from src.realtime_multimodal import (
    MULTIMODAL_CONVERSATION_INSTRUCTIONS,
    build_visual_conversation_item,
)
from src.settings import load_settings
from src.speech_delivery import SpeechDeliveryState
from src.vision_normalize import encode_metadata_free_png
from src.visual_frame_quality import assess_visual_frame_quality

QUESTION = "What object is visible?"
IMAGE_DETAIL = "low"
SYNTHETIC_DESCRIPTION = "red square on white (128x128 PNG, 64x64 red square)"
ACK_EVENT_TYPES = frozenset(
    {
        "conversation.item.added",
        "conversation.item.created",
        "conversation.item.done",
    }
)
TEXT_DELTA_TYPES = frozenset(
    {
        "response.output_text.delta",
        "response.text.delta",
        "response.output_audio_transcript.delta",
        "response.audio_transcript.delta",
    }
)
REFUSAL_MARKERS = (
    "focusing on your description",
    "describe the object",
    "share verbally",
    "details you share",
    "not relying on visual",
    "not relying on visuals",
    "i'm not relying on visuals",
    "cannot see",
    "can't see",
    "do not have an image",
    "don't have an image",
    "no image",
    "unable to see",
    "can't view",
    "cannot view",
    "no camera",
    "without a photo",
    "without an image",
    "describe the object verbally",
)
REMOTE_MARKERS = (
    "remote",
    "controller",
    "clicker",
)
RED_SQUARE_MARKERS_COLOR = ("red",)
RED_SQUARE_MARKERS_SHAPE = ("square", "rectangle", "box")
_ARM_TIMEOUT_SECONDS = 45.0
_CONNECT_MAX_RETRIES = 0
_TRANSCRIPT_WAIT_SECONDS = 8.0


Verdict = Literal["IDENTIFIED", "PARTIAL", "REFUSED", "ERROR", "EMPTY", "OTHER"]
QuestionMode = Literal["text", "audio"]
ImageMode = Literal["synthetic", "camera"]


@dataclass
class FrameReport:
    """Secret-free local inspection of one camera or synthetic frame."""

    source: str
    width: int
    height: int
    encoded_bytes: int
    sequence: int
    age_ms: int
    mean_luminance: float
    dynamic_range: int
    variance: float
    quality_usable: bool
    quality_reason: str


@dataclass
class ItemStructure:
    """Secret-free Realtime conversation item summary."""

    source_event: str
    item_type: str
    role: str
    status: str
    content_types: list[str]
    has_transcript: bool
    transcript_chars: int
    previous_item_id_present: bool
    id_present: bool


@dataclass
class ArmSpec:
    """One live matrix arm."""

    name: str
    question_mode: QuestionMode
    image_mode: ImageMode
    wait_for_transcript: bool
    inject_m28_plan: bool


@dataclass
class ArmResult:
    """Secret-free result of one live matrix arm."""

    name: str
    model: str
    question_mode: QuestionMode
    image_mode: ImageMode
    wait_for_transcript: bool
    inject_m28_plan: bool
    instructions_fingerprint: str
    question_item: ItemStructure | None = None
    image_item: ItemStructure | None = None
    transcript: str = ""
    error: str = ""
    event_types: list[str] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    verdict: Verdict = "EMPTY"
    visual_identification: str = "FAIL"
    refusal: str = "no"
    user_item_committed_before_image: bool | None = None
    transcript_before_image_create: bool | None = None
    transcript_before_response_create: bool | None = None
    image_ack_before_response_create: bool | None = None
    observed_user_transcript: str = ""


def instructions_fingerprint(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"sha256={digest} len={len(text)}"


def _synthetic_png() -> bytes:
    image = Image.new("RGB", (128, 128), (255, 255, 255))
    for y in range(32, 96):
        for x in range(32, 96):
            image.putpixel((x, y), (220, 16, 16))
    png_bytes, _width, _height = encode_metadata_free_png(image)
    return png_bytes


def _frame_from_png(
    png_bytes: bytes,
    *,
    sequence: int,
    captured_at: float,
) -> RealtimeVisualFrame:
    with Image.open(io.BytesIO(png_bytes)) as image:
        width, height = image.size
    return RealtimeVisualFrame(
        image_bytes=png_bytes,
        mime_type="image/png",
        width=width,
        height=height,
        sequence=sequence,
        captured_at_monotonic=captured_at,
    )


def report_frame(frame: RealtimeVisualFrame, *, source: str, now: float) -> FrameReport:
    quality = assess_visual_frame_quality(frame, now=now)
    age_ms = int(max(0.0, (now - frame.captured_at_monotonic) * 1000))
    return FrameReport(
        source=source,
        width=frame.width,
        height=frame.height,
        encoded_bytes=len(frame.image_bytes),
        sequence=frame.sequence,
        age_ms=age_ms,
        mean_luminance=round(quality.mean_luminance, 1),
        dynamic_range=quality.dynamic_range,
        variance=round(quality.luminance_variance, 1),
        quality_usable=quality.usable,
        quality_reason=quality.reason,
    )


def classify_transcript(
    transcript: str,
    *,
    image_mode: ImageMode,
) -> Verdict:
    text = transcript.strip()
    if not text:
        return "EMPTY"
    lowered = text.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return "REFUSED"
    if image_mode == "synthetic":
        color = any(token in lowered for token in RED_SQUARE_MARKERS_COLOR)
        shape = any(token in lowered for token in RED_SQUARE_MARKERS_SHAPE)
        if color and shape:
            return "IDENTIFIED"
        if color or shape:
            return "PARTIAL"
        return "OTHER"
    if any(marker in lowered for marker in REMOTE_MARKERS):
        return "IDENTIFIED"
    object_words = (
        "phone",
        "keyboard",
        "mouse",
        "cup",
        "bottle",
        "book",
        "laptop",
        "monitor",
        "hand",
        "person",
        "desk",
        "chair",
        "cable",
        "black",
    )
    if any(word in lowered for word in object_words):
        return "PARTIAL"
    return "OTHER"


def _safe_error_text(value: object) -> str:
    text = str(value)
    if "sk-" in text or "api_key" in text.lower():
        return type(value).__name__
    if "base64" in text.lower() or "data:image" in text:
        return "provider error (payload omitted)"
    return text[:240]


def _event_type(event: object) -> str:
    return str(getattr(event, "type", "") or "")


def _event_error_message(event: object) -> str:
    error = getattr(event, "error", None)
    if error is None:
        return "error event"
    message = getattr(error, "message", None)
    code = getattr(error, "code", None)
    parts = [str(part) for part in (code, message) if part]
    return _safe_error_text(" ".join(parts) or "error event")


def _item_id_from_event(event: object) -> str | None:
    item = getattr(event, "item", None)
    item_id = getattr(item, "id", None) if item is not None else None
    if isinstance(item_id, str) and item_id:
        return item_id
    raw = getattr(event, "item_id", None)
    if isinstance(raw, str) and raw:
        return raw
    return None


def _summarize_item(event: object) -> ItemStructure:
    item = getattr(event, "item", None)
    content = getattr(item, "content", None) if item is not None else None
    content_types: list[str] = []
    has_transcript = False
    transcript_chars = 0
    if isinstance(content, list):
        for part in content:
            part_type = str(getattr(part, "type", "") or "")
            if part_type:
                content_types.append(part_type)
            transcript = getattr(part, "transcript", None)
            if isinstance(transcript, str) and transcript.strip():
                has_transcript = True
                transcript_chars = len(transcript.strip())
    previous = getattr(event, "previous_item_id", None)
    return ItemStructure(
        source_event=_event_type(event),
        item_type=str(getattr(item, "type", "") or ""),
        role=str(getattr(item, "role", "") or ""),
        status=str(getattr(item, "status", "") or ""),
        content_types=content_types,
        has_transcript=has_transcript,
        transcript_chars=transcript_chars,
        previous_item_id_present=bool(previous),
        id_present=bool(_item_id_from_event(event)),
    )


def _collect_text(event: object) -> str:
    event_type = _event_type(event)
    if event_type in TEXT_DELTA_TYPES:
        delta = getattr(event, "delta", None)
        return delta if isinstance(delta, str) else ""
    if event_type != "response.done":
        return ""
    response = getattr(event, "response", None)
    output = getattr(response, "output", None) if response is not None else None
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            for attr in ("text", "transcript"):
                value = getattr(part, attr, None)
                if isinstance(value, str) and value:
                    chunks.append(value)
    return "".join(chunks)


def _recv_event(connection: object, timeout: float) -> object | None:
    raw_connection = getattr(connection, "_connection", None)
    recv = getattr(raw_connection, "recv", None)
    parse_event = getattr(connection, "parse_event", None)
    if callable(recv) and callable(parse_event):
        try:
            raw = recv(timeout=timeout, decode=False)
        except TypeError:
            raw = recv()
        except TimeoutError:
            return None
        return parse_event(raw)
    recv_direct = getattr(connection, "recv", None)
    if not callable(recv_direct):
        return None
    try:
        return recv_direct(timeout=timeout)
    except TypeError:
        return recv_direct()
    except TimeoutError:
        return None


def _session_payload(settings: object, instructions: str) -> dict[str, object]:
    audio_format: dict[str, object] = {"type": "audio/pcm", "rate": 24000}
    transcription_model = getattr(settings, "transcription_model", "gpt-4o-mini-transcribe")
    return {
        "type": "realtime",
        "instructions": instructions,
        "tools": [],
        "tool_choice": "none",
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "format": audio_format,
                "transcription": {"model": transcription_model},
                "turn_detection": None,
            }
        },
    }


def capture_camera_frame() -> tuple[RealtimeVisualFrame, Path]:
    """Capture one production-normalized camera frame and write a temp preview."""
    camera = CameraCaptureSession()
    first = camera.open_and_capture_first()
    camera.start_worker()
    try:
        frame = camera.wait_for_usable_fresh_frame(wait_seconds=2.0) or first
    finally:
        camera.stop()
    preview = Path(tempfile.gettempdir()) / "cortana_m26_salience_preview.png"
    preview.write_bytes(frame.image_bytes)
    return frame, preview


def load_frame_file(path: Path) -> RealtimeVisualFrame:
    png_bytes = path.read_bytes()
    return _frame_from_png(png_bytes, sequence=1, captured_at=time.monotonic())


def _wav_to_pcm24k(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or sample_width != 2:
        raise RuntimeError("tts_wav_format")
    if rate == REALTIME_VOICE_SAMPLE_RATE_HZ:
        return frames
    raise RuntimeError(f"tts_wav_rate_{rate}")


def synthesize_question_pcm(client: Any, settings: object) -> bytes:
    """Return 24 kHz mono PCM for the isolation question. Bytes are not logged."""
    response = client.audio.speech.create(
        input=QUESTION,
        model=getattr(settings, "tts_model"),
        voice=getattr(settings, "realtime_voice", None) or getattr(settings, "tts_voice"),
        response_format="pcm",
    )
    content = getattr(response, "content", None)
    if isinstance(content, bytes) and content:
        return content
    wav_response = client.audio.speech.create(
        input=QUESTION,
        model=getattr(settings, "tts_model"),
        voice=getattr(settings, "tts_voice"),
        response_format="wav",
    )
    wav_bytes = getattr(wav_response, "content", None)
    if not isinstance(wav_bytes, bytes) or not wav_bytes:
        raise RuntimeError("tts_empty")
    return _wav_to_pcm24k(wav_bytes)


def _append_pcm(connection: Any, pcm: bytes) -> None:
    silence = b"\x00" * REALTIME_VOICE_FRAME_BYTES
    payload = silence * 5 + pcm + silence * 5
    offset = 0
    while offset < len(payload):
        chunk = payload[offset : offset + REALTIME_VOICE_FRAME_BYTES]
        if len(chunk) < REALTIME_VOICE_FRAME_BYTES:
            chunk = chunk + (b"\x00" * (REALTIME_VOICE_FRAME_BYTES - len(chunk)))
        encoded = base64.b64encode(chunk).decode("ascii")
        connection.input_audio_buffer.append(audio=encoded)
        offset += REALTIME_VOICE_FRAME_BYTES
    connection.input_audio_buffer.commit()


def _m28_instructions(base: str, transcript: str, visual_item_id: str | None) -> str:
    state = ConversationState()
    state.set_interaction_mode("multimodal")
    if visual_item_id:
        state.set_visual_context_ref(visual_item_id)
    intelligence = ConversationIntelligence()
    plan = plan_realtime_turn(
        intelligence,
        transcript or QUESTION,
        state,
        interaction_mode="multimodal",
        visual_context_authorized=True,
    )
    return format_realtime_plan_instructions(
        base,
        plan,
        state,
        delivery_plan=speech_delivery_plan_from_realtime(
            plan,
            SpeechDeliveryState(),
            delivery_mode="multimodal",
        ),
    )


def _mark(result: ArmResult, started: float, label: str) -> None:
    offset_ms = int((time.monotonic() - started) * 1000)
    result.timeline.append(f"{offset_ms}ms {label}")


def run_arm(
    *,
    spec: ArmSpec,
    settings: object,
    client: Any,
    instructions: str,
    synthetic_frame: RealtimeVisualFrame,
    camera_frame: RealtimeVisualFrame | None,
    question_pcm: bytes | None,
) -> ArmResult:
    fingerprint = instructions_fingerprint(instructions)
    result = ArmResult(
        name=spec.name,
        model=str(getattr(settings, "realtime_model")),
        question_mode=spec.question_mode,
        image_mode=spec.image_mode,
        wait_for_transcript=spec.wait_for_transcript,
        inject_m28_plan=spec.inject_m28_plan,
        instructions_fingerprint=fingerprint,
    )
    frame = synthetic_frame if spec.image_mode == "synthetic" else camera_frame
    if frame is None:
        result.error = "camera frame missing"
        result.verdict = "ERROR"
        return result
    if spec.question_mode == "audio" and not question_pcm:
        result.error = "question pcm missing"
        result.verdict = "ERROR"
        return result

    started = time.monotonic()
    question_item_id: str | None = None
    image_item_id: str | None = None
    transcript_ready = False
    image_acked = False
    user_committed = False
    session_update_sent = False

    def handle(event: object) -> None:
        nonlocal question_item_id, image_item_id, transcript_ready, image_acked
        nonlocal user_committed
        event_type = _event_type(event)
        if event_type and event_type not in result.event_types:
            result.event_types.append(event_type)
        if event_type == "error":
            result.error = _event_error_message(event)
            result.verdict = "ERROR"
            return
        if event_type == "input_audio_buffer.committed":
            user_committed = True
            _mark(result, started, "user_item_committed")
            item_id = getattr(event, "item_id", None)
            if isinstance(item_id, str) and item_id and question_item_id is None:
                question_item_id = item_id
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript_ready = True
            text = getattr(event, "transcript", None)
            if isinstance(text, str):
                result.observed_user_transcript = text.strip()[:80]
            _mark(result, started, "transcript_finalized")
            return
        if event_type in ACK_EVENT_TYPES:
            structure = _summarize_item(event)
            item_id = _item_id_from_event(event)
            if structure.role == "user" and "input_audio" in structure.content_types:
                if result.question_item is None:
                    result.question_item = structure
                if item_id and question_item_id is None:
                    question_item_id = item_id
                if not user_committed:
                    user_committed = True
                    _mark(result, started, "user_item_committed")
                return
            if structure.role == "user" and "input_text" in structure.content_types:
                if "input_image" in structure.content_types:
                    if result.image_item is None:
                        result.image_item = structure
                    if item_id:
                        image_item_id = item_id
                    image_acked = True
                    _mark(result, started, "image_ack_received")
                    return
                if result.question_item is None:
                    result.question_item = structure
                if item_id:
                    question_item_id = item_id
                if not user_committed:
                    user_committed = True
                    _mark(result, started, "user_item_committed")
                return
            if structure.role == "user" and "input_image" in structure.content_types:
                if result.image_item is None:
                    result.image_item = structure
                if item_id:
                    image_item_id = item_id
                image_acked = True
                _mark(result, started, "image_ack_received")

    try:
        manager = client.realtime.connect(
            model=getattr(settings, "realtime_model"),
            max_retries=_CONNECT_MAX_RETRIES,
        )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"
        return result

    try:
        with manager as connection:
            connection.session.update(session=_session_payload(settings, instructions))
            deadline = time.monotonic() + 15.0
            session_ready = False
            while time.monotonic() < deadline:
                event = _recv_event(connection, 0.5)
                if event is None:
                    continue
                handle(event)
                if result.verdict == "ERROR":
                    return result
                if _event_type(event) in {"session.updated", "session.created"}:
                    session_ready = True
                    break
            if not session_ready:
                result.error = "session.update timeout"
                result.verdict = "ERROR"
                return result

            if spec.question_mode == "text":
                connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": QUESTION}],
                    }
                )
                _mark(result, started, "text_question_item.create")
            else:
                _mark(result, started, "audio_append_begin")
                _append_pcm(connection, question_pcm or b"")
                _mark(result, started, "audio_commit_sent")

            question_deadline = time.monotonic() + 15.0
            while time.monotonic() < question_deadline and question_item_id is None:
                event = _recv_event(connection, 0.5)
                if event is None:
                    continue
                handle(event)
                if result.verdict == "ERROR":
                    return result
            if question_item_id is None:
                result.error = "question item ack timeout"
                result.verdict = "ERROR"
                return result

            if spec.wait_for_transcript and spec.question_mode == "audio":
                transcript_deadline = time.monotonic() + _TRANSCRIPT_WAIT_SECONDS
                while time.monotonic() < transcript_deadline and not transcript_ready:
                    event = _recv_event(connection, 0.5)
                    if event is None:
                        continue
                    handle(event)
                    if result.verdict == "ERROR":
                        return result

            result.transcript_before_image_create = transcript_ready
            result.user_item_committed_before_image = user_committed
            _mark(result, started, "camera_frame_selected")
            connection.conversation.item.create(
                item=build_visual_conversation_item(frame)
            )
            _mark(result, started, "image_item.create_sent")

            ack_deadline = time.monotonic() + 15.0
            while time.monotonic() < ack_deadline and not image_acked:
                event = _recv_event(connection, 0.5)
                if event is None:
                    continue
                handle(event)
                if result.verdict == "ERROR":
                    return result
            if not image_acked:
                result.error = "image ack timeout"
                result.verdict = "ERROR"
                return result

            if spec.wait_for_transcript and spec.question_mode == "audio" and not transcript_ready:
                transcript_deadline = time.monotonic() + _TRANSCRIPT_WAIT_SECONDS
                while time.monotonic() < transcript_deadline and not transcript_ready:
                    event = _recv_event(connection, 0.5)
                    if event is None:
                        continue
                    handle(event)
                    if result.verdict == "ERROR":
                        return result

            if spec.inject_m28_plan:
                plan_text = _m28_instructions(
                    instructions,
                    result.observed_user_transcript or QUESTION,
                    image_item_id,
                )
                connection.session.update(session=_session_payload(settings, plan_text))
                session_update_sent = True
                _mark(result, started, "per_turn_session.update_sent")
                update_deadline = time.monotonic() + 10.0
                while time.monotonic() < update_deadline:
                    event = _recv_event(connection, 0.5)
                    if event is None:
                        continue
                    handle(event)
                    if result.verdict == "ERROR":
                        return result
                    if _event_type(event) == "session.updated":
                        break

            result.transcript_before_response_create = transcript_ready
            result.image_ack_before_response_create = image_acked
            connection.response.create()
            _mark(result, started, "response.create_sent")

            response_deadline = time.monotonic() + _ARM_TIMEOUT_SECONDS
            chunks: list[str] = []
            done = False
            while time.monotonic() < response_deadline and not done:
                event = _recv_event(connection, 0.5)
                if event is None:
                    continue
                handle(event)
                if result.verdict == "ERROR":
                    return result
                event_type = _event_type(event)
                if event_type == "response.created":
                    _mark(result, started, "response.created")
                if event_type in TEXT_DELTA_TYPES and not any(
                    row.endswith("first_response_text_delta") for row in result.timeline
                ):
                    _mark(result, started, "first_response_text_delta")
                piece = _collect_text(event)
                if event_type in TEXT_DELTA_TYPES and piece:
                    chunks.append(piece)
                if event_type == "response.done":
                    done = True
                    if piece and not chunks:
                        chunks.append(piece)
            if not done:
                result.error = "response.done timeout"
                result.verdict = "ERROR"
                return result
            result.transcript = "".join(chunks).strip()
            result.verdict = classify_transcript(
                result.transcript,
                image_mode=spec.image_mode,
            )
            if not session_update_sent:
                result.timeline.append("session.update omitted")
            for item_id in (question_item_id, image_item_id):
                if not item_id:
                    continue
                try:
                    connection.conversation.item.delete(item_id=item_id)
                except Exception:
                    continue
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"

    if result.verdict == "IDENTIFIED":
        result.visual_identification = "PASS"
        result.refusal = "no"
    elif result.verdict == "REFUSED":
        result.visual_identification = "FAIL"
        result.refusal = "yes"
    elif result.verdict == "ERROR":
        result.visual_identification = "FAIL"
        result.refusal = "no"
    else:
        result.visual_identification = "FAIL"
        result.refusal = "yes" if result.verdict == "REFUSED" else "no"
    return result


MATRIX_ARMS = (
    ArmSpec("A", "text", "synthetic", False, False),
    ArmSpec("B", "text", "camera", False, False),
    ArmSpec("C", "audio", "synthetic", False, False),
    ArmSpec("D", "audio", "camera", False, False),
    ArmSpec("C_wait_transcript", "audio", "synthetic", True, False),
    ArmSpec("D_wait_transcript", "audio", "camera", True, False),
    ArmSpec("D_with_m28_plan", "audio", "camera", True, True),
)


def format_frame_report(report: FrameReport) -> str:
    return "\n".join(
        [
            f"source: {report.source}",
            f"width_height: {report.width}x{report.height}",
            f"encoded_png_bytes: {report.encoded_bytes}",
            f"sequence: {report.sequence}",
            f"age_ms: {report.age_ms}",
            f"mean_luminance: {report.mean_luminance}",
            f"dynamic_range: {report.dynamic_range}",
            f"variance: {report.variance}",
            f"quality_gate: {'PASS' if report.quality_usable else 'FAIL'} ({report.quality_reason})",
        ]
    )


def format_item(item: ItemStructure | None) -> str:
    if item is None:
        return "(none)"
    return (
        f"event={item.source_event} type={item.item_type} role={item.role} "
        f"status={item.status} content={','.join(item.content_types) or '(none)'} "
        f"transcript={item.has_transcript} transcript_chars={item.transcript_chars} "
        f"previous_item_id_present={item.previous_item_id_present} "
        f"id_present={item.id_present}"
    )


def format_report(
    *,
    instructions_fp: str,
    synthetic_report: FrameReport,
    camera_report: FrameReport | None,
    results: list[ArmResult],
) -> str:
    lines = [
        "M26 camera-salience isolation matrix",
        f"question: {QUESTION}",
        f"session_instructions: {instructions_fp}",
        f"synthetic_image: {SYNTHETIC_DESCRIPTION}",
        "output_modalities: text (echo avoided)",
        "stores/tools/memory: none",
        "",
        "SYNTHETIC FRAME",
        format_frame_report(synthetic_report),
        "",
        "CAMERA FRAME",
        format_frame_report(camera_report) if camera_report else "unavailable",
        "",
        "RESULTS",
    ]
    for result in results:
        lines.extend(
            [
                f"ARM {result.name}",
                f"  question_mode: {result.question_mode}",
                f"  image_mode: {result.image_mode}",
                f"  wait_for_transcript: {result.wait_for_transcript}",
                f"  inject_m28_plan: {result.inject_m28_plan}",
                f"  instructions: {result.instructions_fingerprint}",
                f"  question_item: {format_item(result.question_item)}",
                f"  image_item: {format_item(result.image_item)}",
                f"  observed_user_transcript: {result.observed_user_transcript or '(none)'}",
                f"  transcript_before_image_create: {result.transcript_before_image_create}",
                f"  transcript_before_response_create: {result.transcript_before_response_create}",
                f"  image_ack_before_response_create: {result.image_ack_before_response_create}",
                f"  timeline: {' | '.join(result.timeline) or '(none)'}",
                f"  event_types: {', '.join(result.event_types) or '(none)'}",
                f"  model_transcript: {result.transcript or '(empty)'}",
                f"  error: {result.error or '(none)'}",
                f"  verdict: {result.verdict}",
                f"  VISUAL_IDENTIFICATION: {result.visual_identification}",
                f"  REFUSAL: {result.refusal}",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M26 camera salience live matrix")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--frame-file", type=Path)
    parser.add_argument("--skip-camera", action="store_true")
    args = parser.parse_args(argv)

    if args.preview_only:
        try:
            _frame, preview = capture_camera_frame()
        except Exception as error:
            print(f"CAMERA_CAPTURE_ERROR: {_safe_error_text(error)}")
            return 1
        print(f"PREVIEW_PATH={preview}")
        print(f"PREVIEW_BYTES={preview.stat().st_size}")
        return 0

    settings = load_settings()
    instructions = MULTIMODAL_CONVERSATION_INSTRUCTIONS
    fingerprint = instructions_fingerprint(instructions)
    now = time.monotonic()
    synthetic_png = _synthetic_png()
    synthetic_frame = _frame_from_png(synthetic_png, sequence=0, captured_at=now)
    synthetic_report = report_frame(synthetic_frame, source="synthetic", now=now)

    camera_frame: RealtimeVisualFrame | None = None
    camera_report: FrameReport | None = None
    preview_path: Path | None = None
    if not args.skip_camera:
        try:
            if args.frame_file is not None:
                camera_frame = load_frame_file(args.frame_file)
                preview_path = args.frame_file
            else:
                camera_frame, preview_path = capture_camera_frame()
            camera_report = report_frame(
                camera_frame,
                source="camera",
                now=time.monotonic(),
            )
        except Exception as error:
            print(f"CAMERA_CAPTURE_ERROR: {_safe_error_text(error)}")
            camera_frame = None

    client = create_openai_client(settings)
    question_pcm: bytes | None = None
    try:
        question_pcm = synthesize_question_pcm(client, settings)
    except Exception as error:
        print(f"TTS_ERROR: {_safe_error_text(error)}")

    results: list[ArmResult] = []
    for spec in MATRIX_ARMS:
        if spec.image_mode == "camera" and camera_frame is None:
            failed = ArmResult(
                name=spec.name,
                model=str(settings.realtime_model),
                question_mode=spec.question_mode,
                image_mode=spec.image_mode,
                wait_for_transcript=spec.wait_for_transcript,
                inject_m28_plan=spec.inject_m28_plan,
                instructions_fingerprint=fingerprint,
                error="camera frame unavailable",
                verdict="ERROR",
                visual_identification="FAIL",
            )
            results.append(failed)
            continue
        if spec.question_mode == "audio" and question_pcm is None:
            failed = ArmResult(
                name=spec.name,
                model=str(settings.realtime_model),
                question_mode=spec.question_mode,
                image_mode=spec.image_mode,
                wait_for_transcript=spec.wait_for_transcript,
                inject_m28_plan=spec.inject_m28_plan,
                instructions_fingerprint=fingerprint,
                error="question pcm unavailable",
                verdict="ERROR",
                visual_identification="FAIL",
            )
            results.append(failed)
            continue
        results.append(
            run_arm(
                spec=spec,
                settings=settings,
                client=client,
                instructions=instructions,
                synthetic_frame=synthetic_frame,
                camera_frame=camera_frame,
                question_pcm=question_pcm,
            )
        )

    print(
        format_report(
            instructions_fp=fingerprint,
            synthetic_report=synthetic_report,
            camera_report=camera_report,
            results=results,
        )
    )
    if preview_path is not None and args.frame_file is None:
        try:
            preview_path.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
