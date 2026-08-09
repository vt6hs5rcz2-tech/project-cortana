"""Tests for Milestone 23 visual analysis service and AI path."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO

import pytest
from PIL import Image

from src.ai_service import (
    AIResponse,
    ResponsesApiInput,
    ResponsesClient,
    VISUAL_ANALYSIS_INSTRUCTIONS,
    generate_visual_analysis_response,
)
from src.settings import Settings
from src.vision_input import NormalizedVisualInput
from src.vision_service import (
    VISION_AI_FAILURE,
    VISION_QUESTION_REQUIRED,
    VISION_QUESTION_TOO_LONG,
    VisualAnalysisResult,
    VisualAnalysisService,
    VisionAnalysisValidationError,
    parse_visual_model_output,
)


def _png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "green").save(buffer, format="PNG")
    return buffer.getvalue()


def _normalized(size: tuple[int, int] = (8, 8)) -> NormalizedVisualInput:
    png = _png_bytes(size)
    return NormalizedVisualInput(
        mime_type="image/png",
        image_bytes=png,
        width=size[0],
        height=size[1],
        source_kind="local_file",
    )


@dataclass
class FakeAIResponse:
    output_text: str


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.model: str | None = None
        self.input: ResponsesApiInput | None = None
        self.instructions: str | None = None
        self._output_text = output_text

    def create(
        self,
        *,
        model: str,
        input: ResponsesApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        self.model = model
        self.input = input
        self.instructions = instructions
        return FakeAIResponse(output_text=self._output_text)


class FakeClient:
    responses: ResponsesClient

    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)

    @property
    def fake_responses(self) -> FakeResponses:
        assert isinstance(self.responses, FakeResponses)
        return self.responses


def _valid_json(
    *,
    answer: str = "A green square is visible.",
    visibility: str = "observed",
    warning: str | None = None,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "visibility": visibility,
            "warning": warning,
        }
    )


def test_generate_visual_analysis_response_payload_shape() -> None:
    client = FakeClient(_valid_json())
    settings = Settings(openai_api_key="test-key", openai_model="test-model")
    image = _normalized()

    result = generate_visual_analysis_response(
        client=client,
        settings=settings,
        task_text="  Describe this image  ",
        image=image,
    )

    assert result == _valid_json()
    assert client.fake_responses.model == "test-model"
    assert client.fake_responses.instructions == VISUAL_ANALYSIS_INSTRUCTIONS
    assert isinstance(client.fake_responses.input, list)
    message = client.fake_responses.input[0]
    assert message["role"] == "user"
    content = message["content"]
    assert content[0] == {"type": "input_text", "text": "Describe this image"}
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "auto"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    encoded = content[1]["image_url"].removeprefix("data:image/png;base64,")
    assert base64.b64decode(encoded) == image.image_bytes


def test_visual_payload_has_no_path_history_or_memory_sentinels() -> None:
    client = FakeClient(_valid_json())
    settings = Settings(openai_api_key="test-key", openai_model="test-model")
    image = _normalized()
    generate_visual_analysis_response(
        client=client,
        settings=settings,
        task_text="What color is visible?",
        image=image,
    )
    payload = repr(client.fake_responses.input)
    for sentinel in (
        r"C:\Secret\path.png",
        "ConversationHistory",
        "ActiveMemoryContext",
        "document_vault",
        "study_state",
        "reminder",
        "calendar",
        "incident",
        "tool_executor",
        "workflow_executor",
        "GPS",
        "Exif",
    ):
        assert sentinel not in payload


def test_visual_instructions_require_authority_and_uncertainty_bounds() -> None:
    text = VISUAL_ANALYSIS_INSTRUCTIONS.casefold()
    for phrase in (
        "untrusted",
        "tool",
        "workflow",
        "security",
        "calendar",
        "reminder",
        "memory-write",
        "network",
        "shell",
        "qr",
        "identity recognition",
        "visibility",
        "observed",
        "undetermined",
    ):
        assert phrase in text


def test_parse_visual_model_output_valid() -> None:
    result = parse_visual_model_output(
        _valid_json(visibility="mixed", warning="Low contrast")
    )
    assert result == VisualAnalysisResult(
        answer="A green square is visible.",
        visibility="mixed",
        warning="Low contrast",
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not-json",
        json.dumps({"answer": "x", "visibility": "observed"}),
        json.dumps(
            {
                "answer": "x",
                "visibility": "observed",
                "warning": None,
                "extra": 1,
            }
        ),
        json.dumps({"answer": "", "visibility": "observed", "warning": None}),
        json.dumps({"answer": "x", "visibility": "high", "warning": None}),
        json.dumps({"answer": "x", "visibility": "observed", "warning": 1}),
    ],
)
def test_parse_visual_model_output_fail_closed(raw: str) -> None:
    with pytest.raises(VisionAnalysisValidationError, match="could not complete"):
        parse_visual_model_output(raw)
    assert VISION_AI_FAILURE


def test_service_describe_and_ask() -> None:
    client = FakeClient(_valid_json(answer="Seen."))
    settings = Settings(openai_api_key="test-key", openai_model="test-model")
    service = VisualAnalysisService(settings=settings, client=client)
    image = _normalized()

    described = service.describe(image)
    assert described.answer == "Seen."
    assert client.fake_responses.input is not None
    assert "Describe what is visibly present" in repr(client.fake_responses.input)

    asked = service.ask(image, "What color?")
    assert asked.answer == "Seen."
    assert client.fake_responses.input is not None
    content = client.fake_responses.input[0]["content"]
    assert content[0]["text"] == "What color?"


def test_service_rejects_blank_and_long_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(_valid_json())
    settings = Settings(openai_api_key="test-key", openai_model="test-model")
    service = VisualAnalysisService(settings=settings, client=client)
    image = _normalized()

    with pytest.raises(VisionAnalysisValidationError, match="Please provide a question"):
        service.ask(image, "   ")
    assert VISION_QUESTION_REQUIRED

    monkeypatch.setattr("src.vision_service.MAX_VISION_QUESTION_CHARS", 5)
    with pytest.raises(VisionAnalysisValidationError, match="exceeds the maximum"):
        service.ask(image, "123456")
    assert VISION_QUESTION_TOO_LONG


def test_service_maps_provider_errors() -> None:
    class BoomResponses:
        def create(
            self,
            *,
            model: str,
            input: ResponsesApiInput,
            instructions: str | None = None,
        ) -> AIResponse:
            raise RuntimeError("provider exploded")

    class BoomClient:
        responses: ResponsesClient = BoomResponses()

    service = VisualAnalysisService(
        settings=Settings(openai_api_key="k", openai_model="m"),
        client=BoomClient(),
    )
    with pytest.raises(Exception, match="could not complete"):
        service.describe(_normalized())
