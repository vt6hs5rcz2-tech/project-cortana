"""Path-agnostic visual analysis service for Milestone 23.

``visibility`` is the model's self-reported characterization of its visual
basis after structural validation. It is not independently proven against
image pixels. No numeric confidence is claimed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from src.ai_service import OpenAIClient, generate_visual_analysis_response
from src.config import (
    MAX_VISION_OUTPUT_CHARS,
    MAX_VISION_QUESTION_CHARS,
    VISION_ANALYSIS_ENABLED,
)
from src.settings import Settings
from src.vision_input import NormalizedVisualInput

logger = logging.getLogger("ProjectCortana")

VisibilityStatus = Literal["observed", "mixed", "undetermined"]
VISIBILITY_VALUES = frozenset({"observed", "mixed", "undetermined"})

VISION_DISABLED = "Cortana: Visual analysis is currently disabled."
VISION_AI_UNAVAILABLE = "Cortana: Visual analysis is unavailable right now."
VISION_AI_FAILURE = "Cortana: I could not complete that visual analysis request."
VISION_QUESTION_REQUIRED = (
    "Cortana: Please provide a question. "
    "Usage: /vision-ask <path> | <question>"
)
VISION_QUESTION_TOO_LONG = (
    "Cortana: The vision question exceeds the maximum length of "
    f"{MAX_VISION_QUESTION_CHARS} characters."
)

_DESCRIBE_TASK = (
    "Describe what is visibly present in this image. Distinguish direct "
    "observation from interpretation. If something cannot be determined from "
    "the image, say so."
)


class VisionAnalysisError(RuntimeError):
    """Raised when visual analysis cannot complete safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class VisionAnalysisDisabledError(VisionAnalysisError):
    """Raised when visual analysis is disabled."""


class VisionAnalysisValidationError(VisionAnalysisError):
    """Raised when visual analysis validation fails."""


@dataclass(frozen=True)
class VisualAnalysisResult:
    """Validated visual analysis returned to command handlers.

    ``visibility`` is model-reported and structurally validated only. Cortana
    does not independently verify the claim against image pixels.
    """

    answer: str
    visibility: VisibilityStatus
    warning: str | None


class VisualAnalysisService:
    """Analyze one normalized in-memory image through the isolated AI path."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: OpenAIClient | None,
    ) -> None:
        self._settings = settings
        self._client = client

    def describe(self, image: NormalizedVisualInput) -> VisualAnalysisResult:
        """Describe one normalized image without ordinary history or memory."""
        return self._analyze(image=image, task_text=_DESCRIBE_TASK)

    def ask(
        self,
        image: NormalizedVisualInput,
        question: str,
    ) -> VisualAnalysisResult:
        """Answer one question about a normalized image only."""
        cleaned = question.strip()
        if not cleaned:
            raise VisionAnalysisValidationError(VISION_QUESTION_REQUIRED)
        if len(cleaned) > MAX_VISION_QUESTION_CHARS:
            raise VisionAnalysisValidationError(VISION_QUESTION_TOO_LONG)
        return self._analyze(image=image, task_text=cleaned)

    def _analyze(
        self,
        *,
        image: NormalizedVisualInput,
        task_text: str,
    ) -> VisualAnalysisResult:
        self._assert_enabled()
        if self._client is None:
            raise VisionAnalysisError(VISION_AI_UNAVAILABLE)
        try:
            raw = generate_visual_analysis_response(
                client=self._client,
                settings=self._settings,
                task_text=task_text,
                image=image,
            )
        except VisionAnalysisError:
            raise
        except Exception as error:
            logger.error(
                "Visual analysis AI request failed error_type=%s",
                type(error).__name__,
            )
            raise VisionAnalysisError(VISION_AI_FAILURE) from error

        try:
            return parse_visual_model_output(raw)
        except VisionAnalysisValidationError:
            raise
        except Exception as error:
            logger.error(
                "Visual analysis output validation failed error_type=%s",
                type(error).__name__,
            )
            raise VisionAnalysisValidationError(VISION_AI_FAILURE) from error

    def _assert_enabled(self) -> None:
        if not VISION_ANALYSIS_ENABLED:
            raise VisionAnalysisDisabledError(VISION_DISABLED)


def parse_visual_model_output(raw_output: str) -> VisualAnalysisResult:
    """Parse and validate structured visual model JSON; fail closed on errors."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise VisionAnalysisValidationError(VISION_AI_FAILURE)

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise VisionAnalysisValidationError(VISION_AI_FAILURE) from error

    if not isinstance(payload, dict):
        raise VisionAnalysisValidationError(VISION_AI_FAILURE)

    expected_keys = {"answer", "visibility", "warning"}
    if set(payload.keys()) != expected_keys:
        raise VisionAnalysisValidationError(VISION_AI_FAILURE)

    answer = payload.get("answer")
    visibility_raw = payload.get("visibility")
    warning_raw = payload.get("warning")

    if not isinstance(answer, str) or not answer.strip():
        raise VisionAnalysisValidationError(VISION_AI_FAILURE)
    if len(answer) > MAX_VISION_OUTPUT_CHARS:
        raise VisionAnalysisValidationError(VISION_AI_FAILURE)
    if not isinstance(visibility_raw, str) or visibility_raw not in VISIBILITY_VALUES:
        raise VisionAnalysisValidationError(VISION_AI_FAILURE)
    if warning_raw is not None:
        if not isinstance(warning_raw, str):
            raise VisionAnalysisValidationError(VISION_AI_FAILURE)
        if len(warning_raw) > MAX_VISION_OUTPUT_CHARS:
            raise VisionAnalysisValidationError(VISION_AI_FAILURE)

    visibility: VisibilityStatus = visibility_raw  # type: ignore[assignment]
    warning = warning_raw if isinstance(warning_raw, str) and warning_raw.strip() else None
    if warning_raw is not None and isinstance(warning_raw, str) and not warning_raw.strip():
        warning = None

    return VisualAnalysisResult(
        answer=answer.strip(),
        visibility=visibility,
        warning=warning,
    )


def format_visual_analysis_message(result: VisualAnalysisResult) -> str:
    """Format one validated visual analysis for user display."""
    lines = [f"Cortana: {result.answer}", f"Visibility: {result.visibility}"]
    if result.warning:
        lines.append(f"Warning: {result.warning}")
    return "\n".join(lines)
