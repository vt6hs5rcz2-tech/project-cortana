"""Local slash-command handlers for Milestone 26 realtime multimodal mode."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.camera_capture import (
    MULTIMODAL_DISABLED,
    realtime_multimodal_features_enabled,
)
from src.config import (
    REALTIME_MULTIMODAL_ENABLED,
    REALTIME_VOICE_ENABLED,
    VISION_ANALYSIS_ENABLED,
    VOICE_INTERACTION_ENABLED,
)
from src.conversation import ConversationHistory
from src.realtime_multimodal import (
    MULTIMODAL_UNAVAILABLE,
    format_multimodal_status_lines,
    run_realtime_multimodal_session,
)
from src.realtime_voice import RealtimeVoiceClient
from src.realtime_voice_input import REALTIME_UNSUPPORTED_PLATFORM
from src.settings import Settings

logger = logging.getLogger("ProjectCortana")

COMMAND_MULTIMODAL_REALTIME = "multimodal-realtime"

REALTIME_MULTIMODAL_COMMAND_NAMES = frozenset(
    {
        COMMAND_MULTIMODAL_REALTIME,
    }
)


@dataclass(frozen=True)
class RealtimeMultimodalCommandContext:
    """Inputs available to multimodal slash-command handlers."""

    message: str
    settings: Settings
    client: Any | None
    conversation_history: ConversationHistory
    active_memory_context: ActiveMemoryContext
    logger: logging.Logger
    connect_factory: Callable[[], Any] | None = None
    microphone_factory: Callable[..., Any] | None = None
    playback_stream_factory: Callable[[], Any] | None = None
    camera_factory: Callable[[], Any] | None = None
    print_fn: Callable[[str], None] = print


@dataclass(frozen=True)
class RealtimeMultimodalCommandResult:
    """User-facing result from a multimodal command."""

    message: str


def handle_realtime_multimodal_command(
    command_name: str,
    context: RealtimeMultimodalCommandContext,
) -> RealtimeMultimodalCommandResult | None:
    """Dispatch one multimodal command, or return None when unknown."""
    if command_name not in REALTIME_MULTIMODAL_COMMAND_NAMES:
        return None
    if not realtime_multimodal_features_enabled():
        return RealtimeMultimodalCommandResult(message=MULTIMODAL_DISABLED)
    if command_name == COMMAND_MULTIMODAL_REALTIME:
        return _handle_multimodal_realtime(context)
    return None


def append_multimodal_status_lines(message: str, settings: Settings) -> str:
    """Append multimodal status lines to an existing /voice-status message."""
    if not message:
        return format_multimodal_status_lines(settings)
    return f"{message.rstrip()}\n{format_multimodal_status_lines(settings)}"


def _handle_multimodal_realtime(
    context: RealtimeMultimodalCommandContext,
) -> RealtimeMultimodalCommandResult:
    if sys.platform != "win32":
        return RealtimeMultimodalCommandResult(message=REALTIME_UNSUPPORTED_PLATFORM)
    if not (
        VOICE_INTERACTION_ENABLED
        and REALTIME_VOICE_ENABLED
        and VISION_ANALYSIS_ENABLED
        and REALTIME_MULTIMODAL_ENABLED
    ):
        return RealtimeMultimodalCommandResult(message=MULTIMODAL_DISABLED)
    if context.client is None:
        return RealtimeMultimodalCommandResult(message=MULTIMODAL_UNAVAILABLE)

    realtime_client = cast(RealtimeVoiceClient, context.client)
    try:
        getattr(realtime_client, "realtime")
    except Exception:
        return RealtimeMultimodalCommandResult(message=MULTIMODAL_UNAVAILABLE)

    message = run_realtime_multimodal_session(
        settings=context.settings,
        client=realtime_client,
        conversation_history=context.conversation_history,
        active_memory_context=context.active_memory_context,
        logger_=context.logger,
        connect_factory=context.connect_factory,
        microphone_factory=context.microphone_factory,
        playback_stream_factory=context.playback_stream_factory,
        camera_factory=context.camera_factory,
        print_fn=context.print_fn,
    )
    return RealtimeMultimodalCommandResult(message=message)
