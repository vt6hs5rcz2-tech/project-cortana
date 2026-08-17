"""Live M26 implicit vs explicit visual-response-context spike.

Manual investigation only. Not imported by pytest or application startup.
Never prints OPENAI_API_KEY or image bytes. Does not write Cortana stores.
Does not change production M26 routing.
"""

from __future__ import annotations

import base64
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REALTIME_VISUAL_FIXED_LABEL
from src.openai_client import create_openai_client
from src.realtime_multimodal import MULTIMODAL_CONVERSATION_INSTRUCTIONS
from src.settings import load_settings
from src.vision_normalize import encode_metadata_free_png

QUESTION = "What color and shape are visible?"
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
TEXT_DONE_TYPES = frozenset(
    {
        "response.output_text.done",
        "response.text.done",
        "response.output_audio_transcript.done",
        "response.audio_transcript.done",
    }
)
REFUSAL_MARKERS = (
    "focusing on your description",
    "describe the object",
    "share verbally",
    "details you share",
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
)
_ARM_TIMEOUT_SECONDS = 45.0
_CONNECT_MAX_RETRIES = 0


Verdict = Literal["IDENTIFIED", "PARTIAL", "REFUSED", "ERROR", "EMPTY", "OTHER"]


@dataclass
class ArmResult:
    """Secret-free result of one live spike arm."""

    name: str
    model: str
    create_mode: str
    question_item_id: str | None = None
    image_item_id: str | None = None
    transcript: str = ""
    continuity_transcript: str = ""
    error: str = ""
    event_types: list[str] = field(default_factory=list)
    verdict: Verdict = "EMPTY"
    continuity_verdict: Verdict | None = None
    secret_color_mentioned: bool | None = None


def _synthetic_png() -> bytes:
    image = Image.new("RGB", (128, 128), (255, 255, 255))
    for y in range(32, 96):
        for x in range(32, 96):
            image.putpixel((x, y), (220, 16, 16))
    png_bytes, _width, _height = encode_metadata_free_png(image)
    return png_bytes


def _data_uri(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _question_item(*, item_id: str | None) -> dict[str, object]:
    item: dict[str, object] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": QUESTION}],
    }
    if item_id:
        item["id"] = item_id
    return item


def _image_item(*, item_id: str | None, data_uri: str) -> dict[str, object]:
    item: dict[str, object] = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": REALTIME_VISUAL_FIXED_LABEL},
            {
                "type": "input_image",
                "image_url": data_uri,
                "detail": IMAGE_DETAIL,
            },
        ],
    }
    if item_id:
        item["id"] = item_id
    return item


def _inline_user_item(data_uri: str) -> dict[str, object]:
    return {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": QUESTION},
            {
                "type": "input_image",
                "image_url": data_uri,
                "detail": IMAGE_DETAIL,
            },
        ],
    }


def _session_payload() -> dict[str, object]:
    audio_format: dict[str, object] = {"type": "audio/pcm", "rate": 24000}
    return {
        "type": "realtime",
        "instructions": MULTIMODAL_CONVERSATION_INSTRUCTIONS,
        "tools": [],
        "tool_choice": "none",
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "format": audio_format,
                "turn_detection": None,
            },
        },
    }


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


def _collect_text(event: object) -> str:
    event_type = _event_type(event)
    if event_type in TEXT_DELTA_TYPES or event_type in TEXT_DONE_TYPES:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str) and event_type in TEXT_DELTA_TYPES:
            return delta
        text = getattr(event, "text", None)
        if isinstance(text, str) and event_type in TEXT_DONE_TYPES:
            return ""
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


def classify_transcript(transcript: str) -> Verdict:
    """Classify a secret-free model transcript against the synthetic image."""
    text = transcript.strip()
    if not text:
        return "EMPTY"
    lowered = text.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return "REFUSED"
    mentions_red = "red" in lowered
    mentions_shape = any(
        token in lowered for token in ("square", "rectangle", "box")
    )
    if mentions_red and mentions_shape:
        return "IDENTIFIED"
    if mentions_red or mentions_shape:
        return "PARTIAL"
    return "OTHER"


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


def _drain_until(
    connection: object,
    *,
    deadline: float,
    should_stop: Callable[[object, str], bool],
    result: ArmResult,
) -> object | None:
    while time.monotonic() < deadline:
        remaining = min(2.0, max(0.1, deadline - time.monotonic()))
        try:
            event = _recv_event(connection, remaining)
        except Exception as error:
            result.error = _safe_error_text(error)
            result.verdict = "ERROR"
            return None
        if event is None:
            continue
        event_type = _event_type(event)
        if event_type and event_type not in result.event_types:
            result.event_types.append(event_type)
        if event_type == "error":
            result.error = _event_error_message(event)
            result.verdict = "ERROR"
            return event
        if should_stop(event, event_type):
            return event
    return None


def _wait_session_ready(connection: object, result: ArmResult) -> bool:
    deadline = time.monotonic() + 15.0
    event = _drain_until(
        connection,
        deadline=deadline,
        should_stop=lambda _event, event_type: event_type
        in {"session.updated", "session.created"},
        result=result,
    )
    return event is not None and result.verdict != "ERROR"


def _wait_item_ack(
    connection: object,
    result: ArmResult,
    *,
    preferred_id: str,
) -> str | None:
    deadline = time.monotonic() + 15.0
    found: str | None = None

    def _stop(event: object, event_type: str) -> bool:
        nonlocal found
        if event_type not in ACK_EVENT_TYPES:
            return False
        item_id = _item_id_from_event(event)
        if item_id is None:
            return False
        if not preferred_id or item_id == preferred_id or found is None:
            found = item_id
            return True
        return False

    event = _drain_until(
        connection,
        deadline=deadline,
        should_stop=_stop,
        result=result,
    )
    if event is None or result.verdict == "ERROR":
        return None
    return found


def _wait_response_text(connection: object, result: ArmResult) -> None:
    deadline = time.monotonic() + _ARM_TIMEOUT_SECONDS
    chunks: list[str] = []
    done_text = ""

    def _stop(event: object, event_type: str) -> bool:
        nonlocal done_text
        piece = _collect_text(event)
        if event_type in TEXT_DELTA_TYPES and piece:
            chunks.append(piece)
        if event_type == "response.done":
            done_text = piece
            return True
        return False

    event = _drain_until(
        connection,
        deadline=deadline,
        should_stop=_stop,
        result=result,
    )
    if result.verdict == "ERROR":
        return
    if event is None:
        result.error = "response.done timeout"
        result.verdict = "ERROR"
        return
    result.transcript = "".join(chunks).strip() or done_text.strip()
    result.verdict = classify_transcript(result.transcript)


def _create_and_ack(
    connection: Any,
    result: ArmResult,
    item: dict[str, object],
    *,
    preferred_id: str,
) -> str | None:
    connection.conversation.item.create(item=item)
    return _wait_item_ack(connection, result, preferred_id=preferred_id)


def _delete_items(connection: Any, item_ids: list[str | None]) -> None:
    for item_id in item_ids:
        if not item_id:
            continue
        try:
            connection.conversation.item.delete(item_id=item_id)
        except Exception:
            continue


def _probe_continuity(connection: Any, result: ArmResult) -> None:
    """Ask a follow-up with bare response.create to test default-conversation memory."""
    saved_verdict = result.verdict
    saved_error = result.error
    saved_transcript = result.transcript
    follow_id = f"c_{result.name}"[:32]
    follow_item = {
        "id": follow_id,
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "What color did you just mention?",
            }
        ],
    }
    acked = _create_and_ack(
        connection,
        result,
        follow_item,
        preferred_id=follow_id,
    )
    if result.verdict == "ERROR" or acked is None:
        result.continuity_transcript = result.error or "follow-up item ack timeout"
        result.continuity_verdict = "ERROR"
        result.verdict = saved_verdict
        result.error = saved_error
        result.transcript = saved_transcript
        return
    try:
        connection.response.create()
    except Exception as error:
        result.continuity_transcript = _safe_error_text(error)
        result.continuity_verdict = "ERROR"
        result.verdict = saved_verdict
        result.error = saved_error
        result.transcript = saved_transcript
        _delete_items(connection, [acked])
        return
    _wait_response_text(connection, result)
    result.continuity_transcript = result.transcript
    result.continuity_verdict = classify_transcript(result.continuity_transcript)
    result.verdict = saved_verdict
    result.error = saved_error
    result.transcript = saved_transcript
    _delete_items(connection, [acked])


def _run_connected_arm(
    *,
    name: str,
    model: str,
    create_mode: str,
    png_bytes: bytes,
    issue_create: Callable[[Any, ArmResult, str, str, str], None],
    assign_client_ids: bool = True,
) -> ArmResult:
    result = ArmResult(name=name, model=model, create_mode=create_mode)
    settings = load_settings()
    client = create_openai_client(settings)
    data_uri = _data_uri(png_bytes)
    question_id = "q1" if assign_client_ids else None
    image_id = "i1" if assign_client_ids else None
    try:
        manager = client.realtime.connect(
            model=model,
            max_retries=_CONNECT_MAX_RETRIES,
        )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"
        return result
    try:
        with manager as connection:
            connection.session.update(session=_session_payload())
            if not _wait_session_ready(connection, result):
                if result.verdict != "ERROR":
                    result.error = "session.update timeout"
                    result.verdict = "ERROR"
                return result
            acked_question = _create_and_ack(
                connection,
                result,
                _question_item(item_id=question_id),
                preferred_id=question_id or "",
            )
            if result.verdict == "ERROR":
                return result
            if acked_question is None:
                result.error = "question item ack timeout"
                result.verdict = "ERROR"
                return result
            result.question_item_id = acked_question
            acked_image = _create_and_ack(
                connection,
                result,
                _image_item(item_id=image_id, data_uri=data_uri),
                preferred_id=image_id or "",
            )
            if result.verdict == "ERROR":
                return result
            if acked_image is None:
                result.error = "image item ack timeout"
                result.verdict = "ERROR"
                return result
            result.image_item_id = acked_image
            issue_create(connection, result, acked_question, acked_image, data_uri)
            if result.verdict == "ERROR":
                return result
            _wait_response_text(connection, result)
            if result.verdict != "ERROR" and name in {
                "implicit",
                "explicit_inline",
            }:
                _probe_continuity(connection, result)
            _delete_items(
                connection,
                [result.question_item_id, result.image_item_id],
            )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"
    return result


def _issue_bare_create(
    connection: Any,
    result: ArmResult,
    _question_id: str,
    _image_id: str,
    _data_uri: str,
) -> None:
    try:
        connection.response.create()
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"


def _item_reference_payload(
    *,
    question_id: str,
    image_id: str,
    conversation: Literal["auto", "none"],
) -> dict[str, object]:
    return {
        "conversation": conversation,
        "output_modalities": ["text"],
        "input": [
            {"type": "item_reference", "id": question_id},
            {"type": "item_reference", "id": image_id},
        ],
    }


def _issue_item_reference_create(
    connection: Any,
    result: ArmResult,
    question_id: str,
    image_id: str,
    _data_uri: str,
) -> None:
    try:
        connection.response.create(
            response=_item_reference_payload(
                question_id=question_id,
                image_id=image_id,
                conversation="auto",
            )
        )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"


def _issue_item_reference_none_create(
    connection: Any,
    result: ArmResult,
    question_id: str,
    image_id: str,
    _data_uri: str,
) -> None:
    try:
        connection.response.create(
            response=_item_reference_payload(
                question_id=question_id,
                image_id=image_id,
                conversation="none",
            )
        )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"


def _issue_inline_create(
    connection: Any,
    result: ArmResult,
    _question_id: str,
    _image_id: str,
    data_uri: str,
) -> None:
    try:
        connection.response.create(
            response={
                "conversation": "auto",
                "output_modalities": ["text"],
                "input": [_inline_user_item(data_uri)],
            }
        )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"


def _run_decoy_merge_arm(*, model: str, png_bytes: bytes) -> ArmResult:
    """Test whether response.input replaces or merges default conversation."""
    result = ArmResult(
        name="input_vs_default_conversation",
        model=model,
        create_mode="decoy in conversation; inline image-only response.input",
    )
    settings = load_settings()
    client = create_openai_client(settings)
    data_uri = _data_uri(png_bytes)
    decoy = {
        "id": "d1",
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "The secret color word is purple.",
            }
        ],
    }
    mixed_question = (
        "What color and shape are visible? "
        "If a secret color word was stated earlier, repeat that word."
    )
    inline_item = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": mixed_question},
            {
                "type": "input_image",
                "image_url": data_uri,
                "detail": IMAGE_DETAIL,
            },
        ],
    }
    try:
        manager = client.realtime.connect(
            model=model,
            max_retries=_CONNECT_MAX_RETRIES,
        )
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"
        return result
    try:
        with manager as connection:
            connection.session.update(session=_session_payload())
            if not _wait_session_ready(connection, result):
                if result.verdict != "ERROR":
                    result.error = "session.update timeout"
                    result.verdict = "ERROR"
                return result
            acked = _create_and_ack(connection, result, decoy, preferred_id="d1")
            if result.verdict == "ERROR":
                return result
            if acked is None:
                result.error = "decoy item ack timeout"
                result.verdict = "ERROR"
                return result
            connection.response.create(
                response={
                    "conversation": "auto",
                    "output_modalities": ["text"],
                    "input": [inline_item],
                }
            )
            _wait_response_text(connection, result)
            lowered = result.transcript.lower()
            result.secret_color_mentioned = "purple" in lowered
            _delete_items(connection, [acked])
    except Exception as error:
        result.error = _safe_error_text(error)
        result.verdict = "ERROR"
    return result


def run_live_spike() -> list[ArmResult]:
    """Run implicit then explicit live arms against gpt-realtime-mini."""
    settings = load_settings()
    model = settings.realtime_model
    png_bytes = _synthetic_png()
    return [
        _run_connected_arm(
            name="implicit",
            model=model,
            create_mode="bare response.create()",
            png_bytes=png_bytes,
            issue_create=_issue_bare_create,
        ),
        _run_connected_arm(
            name="explicit_item_reference_auto",
            model=model,
            create_mode="response.input item_reference x2 conversation=auto",
            png_bytes=png_bytes,
            issue_create=_issue_item_reference_create,
        ),
        _run_connected_arm(
            name="explicit_item_reference_none",
            model=model,
            create_mode="response.input item_reference x2 conversation=none",
            png_bytes=png_bytes,
            issue_create=_issue_item_reference_none_create,
        ),
        _run_connected_arm(
            name="explicit_item_reference_server_ids",
            model=model,
            create_mode="item_reference x2 conversation=none server-generated ids",
            png_bytes=png_bytes,
            issue_create=_issue_item_reference_none_create,
            assign_client_ids=False,
        ),
        _run_connected_arm(
            name="explicit_inline",
            model=model,
            create_mode="response.input inline input_text+input_image",
            png_bytes=png_bytes,
            issue_create=_issue_inline_create,
        ),
        _run_decoy_merge_arm(model=model, png_bytes=png_bytes),
    ]


def format_report(results: list[ArmResult]) -> str:
    """Return a secret-free comparison report."""
    lines = [
        "M26 visual-response-context live spike",
        f"synthetic_image: {SYNTHETIC_DESCRIPTION}",
        f"question: {QUESTION}",
        "session_instructions: production MULTIMODAL_CONVERSATION_INSTRUCTIONS",
        "stores/tools/memory/audio persistence: none",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"ARM {result.name}",
                f"  model: {result.model}",
                f"  create_mode: {result.create_mode}",
                f"  question_item_id_present: {bool(result.question_item_id)}",
                f"  image_item_id_present: {bool(result.image_item_id)}",
                f"  event_types: {', '.join(result.event_types) or '(none)'}",
                f"  transcript: {result.transcript or '(empty)'}",
                f"  error: {result.error or '(none)'}",
                f"  verdict: {result.verdict}",
                f"  continuity_transcript: {result.continuity_transcript or '(none)'}",
                f"  continuity_verdict: {result.continuity_verdict or '(none)'}",
                f"  secret_color_mentioned: {result.secret_color_mentioned}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    try:
        results = run_live_spike()
    except Exception as error:
        print(f"SPIKE_ERROR: {_safe_error_text(error)}")
        return 1
    print(format_report(results))
    if any(result.verdict == "ERROR" and result.name == "implicit" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
