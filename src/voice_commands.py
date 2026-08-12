"""Local slash-command handlers for Milestone 24 natural voice conversation."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.config import (
    MAX_VOICE_UTTERANCE_SECONDS,
    VOICE_INTERACTION_ENABLED,
    VOICE_SAMPLE_RATE_HZ,
)
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.settings import Settings
from src.voice_input import (
    VOICE_CANCELLED,
    VOICE_DISABLED,
    VOICE_LISTENING_PROMPT,
    MicrophoneCaptureAdapter,
    VoiceCaptureCancelledError,
    VoiceInputError,
)
from src.voice_service import (
    VoiceAudioClient,
    VoiceService,
    VoiceServiceError,
)

logger = logging.getLogger("ProjectCortana")

COMMAND_VOICE_TURN = "voice-turn"
COMMAND_VOICE_STATUS = "voice-status"

VOICE_COMMAND_NAMES = frozenset(
    {
        COMMAND_VOICE_TURN,
        COMMAND_VOICE_STATUS,
    }
)

VOICE_CHAT_FAILED = "Cortana: I could not complete that request."
VOICE_PLAYBACK_FAILED = (
    "Cortana: I could not play the spoken response."
)
VOICE_UNSUPPORTED_PLATFORM = (
    "Cortana: Voice interaction requires Windows in this milestone."
)


@dataclass(frozen=True)
class VoiceCommandContext:
    """Inputs available to voice slash-command handlers."""

    message: str
    settings: Settings
    client: OpenAIClient | None
    conversation_history: ConversationHistory
    active_memory_context: ActiveMemoryContext
    logger: logging.Logger
    stop_signal: Callable[[], bool]
    capture: MicrophoneCaptureAdapter | None = None
    voice_service: VoiceService | None = None
    conversation_state: ConversationState | None = None


@dataclass(frozen=True)
class VoiceCommandResult:
    """User-facing result from a voice command."""

    message: str


VoiceCommandHandler = Callable[[VoiceCommandContext], VoiceCommandResult]


def create_default_voice_services(
    *,
    settings: Settings,
    client: OpenAIClient | None,
) -> tuple[MicrophoneCaptureAdapter, VoiceService]:
    """Create the default capture adapter and voice service.

    Construction never opens a microphone.
    """
    capture = MicrophoneCaptureAdapter()
    audio_client = cast(VoiceAudioClient | None, client)
    service = VoiceService(settings=settings, client=audio_client)
    return capture, service


def handle_voice_command(
    command_name: str,
    context: VoiceCommandContext,
) -> VoiceCommandResult | None:
    """Dispatch one voice command, or return None when unknown."""
    if not VOICE_INTERACTION_ENABLED:
        return VoiceCommandResult(message=VOICE_DISABLED)

    handler = VOICE_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None

    try:
        return handler(context)
    except VoiceCaptureCancelledError:
        return VoiceCommandResult(message=VOICE_CANCELLED)
    except VoiceInputError as error:
        return VoiceCommandResult(message=error.user_message)
    except VoiceServiceError as error:
        return VoiceCommandResult(message=error.user_message)


def _require_capture(context: VoiceCommandContext) -> MicrophoneCaptureAdapter:
    if context.capture is not None:
        return context.capture
    return MicrophoneCaptureAdapter()


def _require_service(context: VoiceCommandContext) -> VoiceService:
    if context.voice_service is not None:
        return context.voice_service
    audio_client = cast(VoiceAudioClient | None, context.client)
    return VoiceService(settings=context.settings, client=audio_client)


def _handle_voice_status(context: VoiceCommandContext) -> VoiceCommandResult:
    """Report safe voice configuration without opening the microphone."""
    platform_supported = "yes" if sys.platform == "win32" else "no"
    enabled = "enabled" if VOICE_INTERACTION_ENABLED else "disabled"
    message = (
        "Cortana: Voice status\n"
        f"  Voice interaction: {enabled}\n"
        f"  Platform supported: {platform_supported}\n"
        "  Session state: idle\n"
        f"  Transcription model: {context.settings.transcription_model}\n"
        f"  TTS model: {context.settings.tts_model}\n"
        f"  TTS voice: {context.settings.tts_voice}\n"
        f"  Max utterance seconds: {MAX_VOICE_UTTERANCE_SECONDS}\n"
        f"  Sample rate Hz: {VOICE_SAMPLE_RATE_HZ}"
    )
    return VoiceCommandResult(message=message)


def _play_wav_synchronously(wav_bytes: bytes) -> None:
    """Play one WAV buffer on the Windows default output device."""
    if sys.platform != "win32":
        raise VoiceServiceError(VOICE_UNSUPPORTED_PLATFORM)
    try:
        import winsound
    except Exception as error:
        logger.error(
            "Voice winsound import failed error_type=%s",
            type(error).__name__,
        )
        raise VoiceServiceError(VOICE_PLAYBACK_FAILED) from error
    try:
        winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
    except Exception as error:
        logger.error(
            "Voice playback failed error_type=%s",
            type(error).__name__,
        )
        raise VoiceServiceError(VOICE_PLAYBACK_FAILED) from error


def _handle_voice_turn(context: VoiceCommandContext) -> VoiceCommandResult:
    """Capture one utterance, chat ordinarily, then speak the reply."""
    # Lazy import avoids a module-level cycle:
    # conversation_loop -> commands -> voice_commands -> conversation_loop
    from src.conversation_loop import process_conversation_turn

    if sys.platform != "win32":
        return VoiceCommandResult(message=VOICE_UNSUPPORTED_PLATFORM)
    if context.client is None:
        return VoiceCommandResult(message=VOICE_CHAT_FAILED)

    print(VOICE_LISTENING_PROMPT)
    capture = _require_capture(context)
    try:
        audio = capture.capture(stop_signal=context.stop_signal)
    except VoiceCaptureCancelledError:
        return VoiceCommandResult(message=VOICE_CANCELLED)

    service = _require_service(context)
    transcript = service.transcribe(audio)
    print(f"Cortana: (Heard) {transcript}")

    answer = process_conversation_turn(
        client=context.client,
        settings=context.settings,
        user_message=transcript,
        logger=context.logger,
        conversation_history=context.conversation_history,
        active_memory_context=context.active_memory_context,
        conversation_state=context.conversation_state,
        interaction_mode="voice",
    )
    if answer is None:
        return VoiceCommandResult(message=VOICE_CHAT_FAILED)

    print(f"Cortana: {answer}")

    try:
        speech_bytes = service.synthesize(answer)
    except VoiceServiceError as error:
        return VoiceCommandResult(message=error.user_message)

    try:
        _play_wav_synchronously(speech_bytes)
    except VoiceServiceError as error:
        return VoiceCommandResult(message=error.user_message)
    finally:
        del speech_bytes

    return VoiceCommandResult(message="")


VOICE_COMMAND_HANDLERS: dict[str, VoiceCommandHandler] = {
    COMMAND_VOICE_TURN: _handle_voice_turn,
    COMMAND_VOICE_STATUS: _handle_voice_status,
}
