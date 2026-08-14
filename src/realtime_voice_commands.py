"""Local slash-command handlers for Milestone 25 realtime voice."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.config import (
    REALTIME_VOICE_ENABLED,
    VOICE_INTERACTION_ENABLED,
)
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.speech_delivery import SpeechDeliveryState
from src.realtime_voice import (
    REALTIME_UNAVAILABLE,
    RealtimeVoiceClient,
    format_realtime_voice_status_lines,
    run_realtime_voice_session,
)
from src.realtime_voice_input import (
    REALTIME_UNSUPPORTED_PLATFORM,
    REALTIME_VOICE_DISABLED,
    realtime_voice_features_enabled,
)
from src.settings import Settings

logger = logging.getLogger("ProjectCortana")

COMMAND_VOICE_REALTIME = "voice-realtime"

REALTIME_VOICE_COMMAND_NAMES = frozenset(
    {
        COMMAND_VOICE_REALTIME,
    }
)


@dataclass(frozen=True)
class RealtimeVoiceCommandContext:
    """Inputs available to realtime voice slash-command handlers."""

    message: str
    settings: Settings
    client: Any | None
    conversation_history: ConversationHistory
    active_memory_context: ActiveMemoryContext
    logger: logging.Logger
    connect_factory: Callable[[], Any] | None = None
    microphone_factory: Callable[..., Any] | None = None
    playback_stream_factory: Callable[[], Any] | None = None
    print_fn: Callable[[str], None] = print
    conversation_state: ConversationState | None = None
    speech_delivery_state: SpeechDeliveryState | None = None


@dataclass(frozen=True)
class RealtimeVoiceCommandResult:
    """User-facing result from a realtime voice command."""

    message: str


def handle_realtime_voice_command(
    command_name: str,
    context: RealtimeVoiceCommandContext,
) -> RealtimeVoiceCommandResult | None:
    """Dispatch one realtime voice command, or return None when unknown."""
    if command_name not in REALTIME_VOICE_COMMAND_NAMES:
        return None
    if not VOICE_INTERACTION_ENABLED or not REALTIME_VOICE_ENABLED:
        return RealtimeVoiceCommandResult(message=REALTIME_VOICE_DISABLED)
    if command_name == COMMAND_VOICE_REALTIME:
        return _handle_voice_realtime(context)
    return None


def append_realtime_status_lines(message: str, settings: Settings) -> str:
    """Append realtime status lines to an existing /voice-status message."""
    if not message:
        return format_realtime_voice_status_lines(settings)
    return f"{message.rstrip()}\n{format_realtime_voice_status_lines(settings)}"


def _handle_voice_realtime(
    context: RealtimeVoiceCommandContext,
) -> RealtimeVoiceCommandResult:
    if sys.platform != "win32":
        return RealtimeVoiceCommandResult(message=REALTIME_UNSUPPORTED_PLATFORM)
    if not realtime_voice_features_enabled():
        return RealtimeVoiceCommandResult(message=REALTIME_VOICE_DISABLED)
    if context.client is None:
        return RealtimeVoiceCommandResult(message=REALTIME_UNAVAILABLE)

    realtime_client = cast(RealtimeVoiceClient, context.client)
    try:
        getattr(realtime_client, "realtime")
    except Exception:
        return RealtimeVoiceCommandResult(message=REALTIME_UNAVAILABLE)

    message = run_realtime_voice_session(
        settings=context.settings,
        client=realtime_client,
        conversation_history=context.conversation_history,
        active_memory_context=context.active_memory_context,
        logger_=context.logger,
        connect_factory=context.connect_factory,
        microphone_factory=context.microphone_factory,
        playback_stream_factory=context.playback_stream_factory,
        print_fn=context.print_fn,
        conversation_state=context.conversation_state,
        speech_delivery_state=context.speech_delivery_state,
    )
    return RealtimeVoiceCommandResult(message=message)
