"""Speech recognition and synthesis for Milestone 24 voice transport.

Uses narrow voice-only client protocols. Does not widen the shared
``OpenAIClient`` responses protocol. Does not own capture, history, memory,
slash routing, or operational services.
"""

from __future__ import annotations

import logging
import struct
from io import BytesIO
from typing import Protocol, runtime_checkable

from src.config import (
    ALLOWED_TRANSCRIPTION_MODELS,
    ALLOWED_TTS_MODELS,
    ALLOWED_TTS_VOICES,
    MAX_TTS_CHARS,
    MAX_VOICE_TRANSCRIPT_CHARS,
    VOICE_INTERACTION_ENABLED,
)
from src.local_speech_transcriber import LocalSpeechTranscriber
from src.settings import Settings
from src.voice_input import NormalizedAudioInput

logger = logging.getLogger("ProjectCortana")

VOICE_DISABLED = "Cortana: Voice interaction is currently disabled."
VOICE_TRANSCRIPTION_UNAVAILABLE = (
    "Cortana: Speech transcription is unavailable right now."
)
VOICE_TRANSCRIPTION_FAILED = (
    "Cortana: I could not transcribe that speech."
)
VOICE_EMPTY_TRANSCRIPT = "Cortana: No speech was recognized."
VOICE_TRANSCRIPT_TOO_LONG = (
    "Cortana: The transcript exceeds the maximum length of "
    f"{MAX_VOICE_TRANSCRIPT_CHARS} characters."
)
VOICE_TTS_UNAVAILABLE = "Cortana: Speech synthesis is unavailable right now."
VOICE_TTS_FAILED = "Cortana: I could not generate speech for that response."
VOICE_TTS_TOO_LONG = (
    "Cortana: I could not generate speech because the response exceeds "
    f"the maximum speech length of {MAX_TTS_CHARS} characters."
)


class VoiceServiceError(RuntimeError):
    """Raised when speech recognition or synthesis cannot complete safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class VoiceServiceDisabledError(VoiceServiceError):
    """Raised when voice interaction is disabled."""


class VoiceServiceValidationError(VoiceServiceError):
    """Raised when STT/TTS validation fails."""


@runtime_checkable
class TranscriptionResult(Protocol):
    """Minimum transcription response shape required by Cortana."""

    text: str


@runtime_checkable
class SpeechBinaryResponse(Protocol):
    """Minimum speech synthesis binary response shape."""

    @property
    def content(self) -> bytes:
        """Return synthesized audio bytes."""


class TranscriptionsClient(Protocol):
    """Narrow transcriptions.create surface used by Milestone 24."""

    def create(
        self,
        *,
        file: tuple[str, BytesIO, str],
        model: str,
    ) -> TranscriptionResult:
        """Create one non-streaming transcription."""


class SpeechClient(Protocol):
    """Narrow speech.create surface used by Milestone 24."""

    def create(
        self,
        *,
        input: str,
        model: str,
        voice: str,
        response_format: str = "wav",
    ) -> SpeechBinaryResponse:
        """Create one non-streaming speech synthesis response."""


class VoiceAudioResources(Protocol):
    """Audio resource bundle exposing only transcriptions and speech."""

    transcriptions: TranscriptionsClient
    speech: SpeechClient


class VoiceAudioClient(Protocol):
    """Voice-only OpenAI client protocol separate from ``OpenAIClient``."""

    audio: VoiceAudioResources


_UNSIZED_WAV_CHUNK = 0xFFFFFFFF


def canonicalize_wav_bytes(wav_bytes: bytes) -> bytes:
    """Rewrite streamed/unknown RIFF and data sizes for local playback.

    Some TTS responses return a valid PCM WAV body with ``0xFFFFFFFF`` RIFF
    and data-chunk sizes. Windows ``PlaySound(SND_MEMORY)`` rejects that
    header even though the audio bytes are present.
    """
    if len(wav_bytes) < 44:
        return wav_bytes
    if wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return wav_bytes

    out = bytearray(wav_bytes)
    struct.pack_into("<I", out, 4, len(out) - 8)
    position = 12
    while position + 8 <= len(out):
        chunk_id = bytes(out[position : position + 4])
        declared = struct.unpack_from("<I", out, position + 4)[0]
        start = position + 8
        if chunk_id == b"data":
            actual = len(out) - start
            if declared != actual:
                struct.pack_into("<I", out, position + 4, actual)
            break
        if declared == _UNSIZED_WAV_CHUNK:
            break
        position = start + declared + (declared % 2)
    return bytes(out)


class VoiceService:
    """Transcribe canonical audio and synthesize WAV speech."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: VoiceAudioClient | None,
        local_transcriber: LocalSpeechTranscriber | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._local_transcriber = local_transcriber

    def transcribe(self, audio: NormalizedAudioInput) -> str:
        """Return a validated transcript for one normalized WAV utterance."""
        self._assert_enabled()

        if self._local_transcriber is not None:
            try:
                local_text = self._local_transcriber.transcribe(audio)
            except Exception as error:
                logger.error(
                    "Local voice transcription failed error_type=%s",
                    type(error).__name__,
                )
                raise VoiceServiceError(VOICE_TRANSCRIPTION_FAILED) from error

            cleaned = local_text.strip()
            if not cleaned:
                raise VoiceServiceValidationError(VOICE_EMPTY_TRANSCRIPT)
            if len(cleaned) > MAX_VOICE_TRANSCRIPT_CHARS:
                raise VoiceServiceValidationError(VOICE_TRANSCRIPT_TOO_LONG)
            return cleaned

        if self._client is None:
            raise VoiceServiceError(VOICE_TRANSCRIPTION_UNAVAILABLE)
        model = self._settings.transcription_model
        if model not in ALLOWED_TRANSCRIPTION_MODELS:
            raise VoiceServiceError(VOICE_TRANSCRIPTION_UNAVAILABLE)
        if audio.format != "wav" or not audio.audio_bytes:
            raise VoiceServiceValidationError(VOICE_EMPTY_TRANSCRIPT)

        file_payload = (
            "speech.wav",
            BytesIO(audio.audio_bytes),
            "audio/wav",
        )
        try:
            response = self._client.audio.transcriptions.create(
                file=file_payload,
                model=model,
            )
        except VoiceServiceError:
            raise
        except Exception as error:
            logger.error(
                "Voice transcription request failed error_type=%s",
                type(error).__name__,
            )
            raise VoiceServiceError(VOICE_TRANSCRIPTION_FAILED) from error

        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise VoiceServiceValidationError(VOICE_EMPTY_TRANSCRIPT)
        cleaned = text.strip()
        if not cleaned:
            raise VoiceServiceValidationError(VOICE_EMPTY_TRANSCRIPT)
        if len(cleaned) > MAX_VOICE_TRANSCRIPT_CHARS:
            raise VoiceServiceValidationError(VOICE_TRANSCRIPT_TOO_LONG)
        return cleaned

    def synthesize(self, text: str) -> bytes:
        """Return ephemeral WAV bytes for one plain-text Cortana response."""
        self._assert_enabled()
        if self._client is None:
            raise VoiceServiceError(VOICE_TTS_UNAVAILABLE)
        model = self._settings.tts_model
        voice = self._settings.tts_voice
        if model not in ALLOWED_TTS_MODELS or voice not in ALLOWED_TTS_VOICES:
            raise VoiceServiceError(VOICE_TTS_UNAVAILABLE)

        cleaned = text.strip()
        if not cleaned:
            raise VoiceServiceValidationError(VOICE_TTS_FAILED)
        if len(cleaned) > MAX_TTS_CHARS:
            raise VoiceServiceValidationError(VOICE_TTS_TOO_LONG)

        try:
            response = self._client.audio.speech.create(
                input=cleaned,
                model=model,
                voice=voice,
                response_format="wav",
            )
        except VoiceServiceError:
            raise
        except Exception as error:
            logger.error(
                "Voice speech synthesis request failed error_type=%s",
                type(error).__name__,
            )
            raise VoiceServiceError(VOICE_TTS_FAILED) from error

        content = getattr(response, "content", None)
        if not isinstance(content, bytes) or not content:
            raise VoiceServiceValidationError(VOICE_TTS_FAILED)
        return canonicalize_wav_bytes(content)

    def _assert_enabled(self) -> None:
        if not VOICE_INTERACTION_ENABLED:
            raise VoiceServiceDisabledError(VOICE_DISABLED)
