"""Local slash-command handlers for Milestone 23 visual understanding."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.ai_service import OpenAIClient
from src.config import VISION_ANALYSIS_ENABLED
from src.command_argument_utils import extract_command_argument
from src.path_argument_utils import strip_path_argument_quotes
from src.settings import Settings
from src.vision_input import (
    VISION_DISABLED,
    LoadedVisualFile,
    VisionInputError,
    VisualInputLoader,
)
from src.vision_service import (
    VISION_QUESTION_REQUIRED,
    VisualAnalysisService,
    VisionAnalysisError,
    format_visual_analysis_message,
)

logger = logging.getLogger("ProjectCortana")

FIELD_DELIMITER = " | "

COMMAND_VISION_DESCRIBE = "vision-describe"
COMMAND_VISION_ASK = "vision-ask"
COMMAND_VISION_INFO = "vision-info"

VISION_COMMAND_NAMES = frozenset(
    {
        COMMAND_VISION_DESCRIBE,
        COMMAND_VISION_ASK,
        COMMAND_VISION_INFO,
    }
)

VISION_DESCRIBE_USAGE = "Cortana: Usage: /vision-describe <path>"
VISION_ASK_USAGE = "Cortana: Usage: /vision-ask <path> | <question>"
VISION_INFO_USAGE = "Cortana: Usage: /vision-info <path>"


@dataclass(frozen=True)
class VisionCommandContext:
    """Inputs available to vision slash-command handlers."""

    message: str
    settings: Settings
    client: OpenAIClient | None
    loader: VisualInputLoader | None = None
    analysis_service: VisualAnalysisService | None = None


@dataclass(frozen=True)
class VisionCommandResult:
    """User-facing result from a vision command."""

    message: str


VisionCommandHandler = Callable[[VisionCommandContext], VisionCommandResult]


def create_default_vision_services(
    *,
    settings: Settings,
    client: OpenAIClient | None,
) -> tuple[VisualInputLoader, VisualAnalysisService]:
    """Create the default ephemeral vision loader and analysis service."""
    loader = VisualInputLoader()
    service = VisualAnalysisService(settings=settings, client=client)
    return loader, service


def handle_vision_command(
    command_name: str,
    context: VisionCommandContext,
) -> VisionCommandResult | None:
    """Dispatch one vision command, or return None when unknown."""
    if not VISION_ANALYSIS_ENABLED:
        return VisionCommandResult(message=VISION_DISABLED)

    handler = VISION_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None

    try:
        return handler(context)
    except VisionInputError as error:
        return VisionCommandResult(message=error.user_message)
    except VisionAnalysisError as error:
        return VisionCommandResult(message=error.user_message)


def _require_loader(context: VisionCommandContext) -> VisualInputLoader:
    if context.loader is not None:
        return context.loader
    return VisualInputLoader()


def _require_service(context: VisionCommandContext) -> VisualAnalysisService:
    if context.analysis_service is not None:
        return context.analysis_service
    return VisualAnalysisService(settings=context.settings, client=context.client)


def _load_path(context: VisionCommandContext, path_argument: str) -> LoadedVisualFile:
    return _require_loader(context).load_local_file(path_argument)


def _handle_vision_describe(context: VisionCommandContext) -> VisionCommandResult:
    path_argument = extract_command_argument(context.message)
    cleaned = strip_path_argument_quotes(path_argument).strip()
    if not cleaned:
        return VisionCommandResult(message=VISION_DESCRIBE_USAGE)
    loaded = _load_path(context, cleaned)
    result = _require_service(context).describe(loaded.normalized)
    return VisionCommandResult(message=format_visual_analysis_message(result))


def _handle_vision_ask(context: VisionCommandContext) -> VisionCommandResult:
    argument = extract_command_argument(context.message)
    fields = _split_two_fields(argument)
    if fields is None:
        return VisionCommandResult(message=VISION_ASK_USAGE)
    path_raw, question = fields
    cleaned_path = strip_path_argument_quotes(path_raw).strip()
    if not cleaned_path:
        return VisionCommandResult(message=VISION_ASK_USAGE)
    if not question.strip():
        return VisionCommandResult(message=VISION_QUESTION_REQUIRED)
    loaded = _load_path(context, cleaned_path)
    result = _require_service(context).ask(loaded.normalized, question)
    return VisionCommandResult(message=format_visual_analysis_message(result))


def _handle_vision_info(context: VisionCommandContext) -> VisionCommandResult:
    path_argument = extract_command_argument(context.message)
    cleaned = strip_path_argument_quotes(path_argument).strip()
    if not cleaned:
        return VisionCommandResult(message=VISION_INFO_USAGE)
    loaded = _load_path(context, cleaned)
    message = (
        "Cortana: Vision image info\n"
        f"  Filename: {loaded.display_basename}\n"
        f"  Detected format: {loaded.detected_source_format}\n"
        f"  Source bytes: {loaded.source_bytes}\n"
        f"  Normalized width: {loaded.normalized.width}\n"
        f"  Normalized height: {loaded.normalized.height}\n"
        f"  Normalized bytes: {len(loaded.normalized.image_bytes)}"
    )
    return VisionCommandResult(message=message)


def _split_two_fields(argument: str) -> list[str] | None:
    parts = argument.split(FIELD_DELIMITER)
    if len(parts) != 2:
        return None
    return [part.strip() for part in parts]


VISION_COMMAND_HANDLERS: dict[str, VisionCommandHandler] = {
    COMMAND_VISION_DESCRIBE: _handle_vision_describe,
    COMMAND_VISION_ASK: _handle_vision_ask,
    COMMAND_VISION_INFO: _handle_vision_info,
}
