"""Bounded microphone capture and normalized audio for Milestone 24.

Capture opens the default microphone only inside ``MicrophoneCaptureAdapter.capture``.
Construction and import never open a device. NumPy is not required: capture uses
``sounddevice.RawInputStream`` buffer callbacks.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.config import (
    MAX_VOICE_AUDIO_BYTES,
    MAX_VOICE_PCM_BYTES,
    MAX_VOICE_UTTERANCE_SECONDS,
    MIN_VOICE_UTTERANCE_MS,
    VOICE_CAPTURE_BLOCKSIZE,
    VOICE_CHANNELS,
    VOICE_INTERACTION_ENABLED,
    VOICE_SAMPLE_RATE_HZ,
    VOICE_SAMPLE_WIDTH_BYTES,
)

# Main-thread console poll interval while listening for Enter / timeout.
VOICE_STOP_POLL_INTERVAL_SECONDS = 0.05

logger = logging.getLogger("ProjectCortana")

VoiceAudioFormat = Literal["wav"]
VoiceSourceKind = Literal["microphone"]

VOICE_DISABLED = "Cortana: Voice interaction is currently disabled."
VOICE_UNSUPPORTED_PLATFORM = (
    "Cortana: Voice capture requires Windows in this milestone."
)
VOICE_MICROPHONE_UNAVAILABLE = (
    "Cortana: No usable microphone is available."
)
VOICE_CAPTURE_FAILED = "Cortana: Microphone capture failed."
VOICE_EMPTY_AUDIO = "Cortana: No usable speech was captured."
VOICE_CANCELLED = "Cortana: Voice turn cancelled."
VOICE_LISTENING_PROMPT = (
    "Cortana: Listening... press Enter to finish "
    f"(max {MAX_VOICE_UTTERANCE_SECONDS} seconds)."
)


class VoiceInputError(RuntimeError):
    """Raised when voice capture cannot complete safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class VoiceInputDisabledError(VoiceInputError):
    """Raised when voice interaction is disabled."""


class VoiceInputValidationError(VoiceInputError):
    """Raised when captured audio fails validation."""


class VoiceCaptureCancelledError(VoiceInputError):
    """Raised when the user cancels the listen phase."""


@dataclass(frozen=True)
class NormalizedAudioInput:
    """Ephemeral canonical WAV accepted by speech recognition.

    Contains no device name, device index, filesystem path, or provider file ID.
    """

    audio_bytes: bytes
    format: VoiceAudioFormat
    sample_rate: int
    channels: int
    sample_width_bytes: int
    duration_ms: int
    source_kind: VoiceSourceKind

    def __post_init__(self) -> None:
        if self.format != "wav":
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if self.source_kind != "microphone":
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if self.sample_rate != VOICE_SAMPLE_RATE_HZ:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if self.channels != VOICE_CHANNELS:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if self.sample_width_bytes != VOICE_SAMPLE_WIDTH_BYTES:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if not isinstance(self.audio_bytes, bytes) or not self.audio_bytes:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if len(self.audio_bytes) > MAX_VOICE_AUDIO_BYTES:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < MIN_VOICE_UTTERANCE_MS
            or self.duration_ms > MAX_VOICE_UTTERANCE_SECONDS * 1000
        ):
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)


def pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    """Wrap canonical PCM frames in an in-memory WAV container."""
    if len(pcm_bytes) > MAX_VOICE_PCM_BYTES:
        raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(VOICE_CHANNELS)
        wav_file.setsampwidth(VOICE_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(VOICE_SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm_bytes)
    wav_bytes = buffer.getvalue()
    if len(wav_bytes) > MAX_VOICE_AUDIO_BYTES:
        raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
    return wav_bytes


def duration_ms_from_pcm(pcm_bytes: bytes) -> int:
    """Return duration in milliseconds for canonical PCM bytes."""
    frame_bytes = VOICE_CHANNELS * VOICE_SAMPLE_WIDTH_BYTES
    if frame_bytes < 1 or len(pcm_bytes) % frame_bytes != 0:
        raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
    frames = len(pcm_bytes) // frame_bytes
    return (frames * 1000) // VOICE_SAMPLE_RATE_HZ


def poll_windows_console_enter_stop() -> bool:
    """Return True when Enter is waiting on the Windows console input buffer.

    Polls with ``msvcrt`` on the calling thread. Does not block on ``input()``,
    does not spawn a waiter thread, and cannot outlive ``capture()``.
    """
    if sys.platform != "win32":
        return False
    import msvcrt

    stopped = False
    while msvcrt.kbhit():
        char = msvcrt.getwch()
        if char in {"\r", "\n"}:
            stopped = True
        elif char in {"\x00", "\xe0"}:
            # Consume the second half of a special-key sequence.
            if msvcrt.kbhit():
                msvcrt.getwch()
    return stopped


class MicrophoneCaptureAdapter:
    """Capture one bounded utterance from the default microphone.

    ``__init__`` stores configuration only and never queries or opens a device.
    """

    def __init__(
        self,
        *,
        sample_rate: int = VOICE_SAMPLE_RATE_HZ,
        channels: int = VOICE_CHANNELS,
        sample_width_bytes: int = VOICE_SAMPLE_WIDTH_BYTES,
        max_utterance_seconds: int = MAX_VOICE_UTTERANCE_SECONDS,
        blocksize: int = VOICE_CAPTURE_BLOCKSIZE,
    ) -> None:
        if sample_rate != VOICE_SAMPLE_RATE_HZ:
            raise VoiceInputValidationError(VOICE_CAPTURE_FAILED)
        if channels != VOICE_CHANNELS:
            raise VoiceInputValidationError(VOICE_CAPTURE_FAILED)
        if sample_width_bytes != VOICE_SAMPLE_WIDTH_BYTES:
            raise VoiceInputValidationError(VOICE_CAPTURE_FAILED)
        if max_utterance_seconds != MAX_VOICE_UTTERANCE_SECONDS:
            raise VoiceInputValidationError(VOICE_CAPTURE_FAILED)
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width_bytes = sample_width_bytes
        self._max_utterance_seconds = max_utterance_seconds
        self._blocksize = blocksize
        self._max_pcm_bytes = (
            sample_rate * channels * sample_width_bytes * max_utterance_seconds
        )

    def capture(self, *, stop_signal: Callable[[], bool]) -> NormalizedAudioInput:
        """Open the default mic, capture until stop or max duration, then close.

        ``stop_signal`` is polled on the calling thread and must return True to
        finish recording. It may raise ``EOFError`` or ``KeyboardInterrupt`` to
        cancel. No background stdin waiter thread is used, so nothing can remain
        blocked on ``input()`` after this method returns.
        """
        if not VOICE_INTERACTION_ENABLED:
            raise VoiceInputDisabledError(VOICE_DISABLED)
        if sys.platform != "win32":
            raise VoiceInputValidationError(VOICE_UNSUPPORTED_PLATFORM)

        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except Exception as error:
            logger.error(
                "Voice sounddevice import failed error_type=%s",
                type(error).__name__,
            )
            raise VoiceInputError(VOICE_MICROPHONE_UNAVAILABLE) from error

        chunks: list[bytes] = []
        chunks_lock = threading.Lock()
        recording_active = threading.Event()
        recording_active.set()
        capture_error: list[BaseException] = []
        cancel_error: BaseException | None = None
        max_pcm_bytes = self._max_pcm_bytes
        accepted_pcm_bytes = 0

        def callback(
            indata: object,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            nonlocal accepted_pcm_bytes
            del frames, time_info, status
            if not recording_active.is_set():
                return
            try:
                frame = memoryview(indata).tobytes()  # type: ignore[arg-type]
            except Exception as error:
                capture_error.append(error)
                recording_active.clear()
                return
            with chunks_lock:
                if not recording_active.is_set():
                    return
                remaining = max_pcm_bytes - accepted_pcm_bytes
                if remaining <= 0:
                    recording_active.clear()
                    return
                if len(frame) > remaining:
                    chunks.append(frame[:remaining])
                    accepted_pcm_bytes += remaining
                    recording_active.clear()
                    return
                chunks.append(frame)
                accepted_pcm_bytes += len(frame)
                if accepted_pcm_bytes >= max_pcm_bytes:
                    recording_active.clear()

        stream = None
        try:
            stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._blocksize,
                channels=self._channels,
                dtype="int16",
                callback=callback,
            )
            stream.start()
            deadline = time.monotonic() + float(self._max_utterance_seconds)
            try:
                while recording_active.is_set() and time.monotonic() < deadline:
                    try:
                        if stop_signal():
                            break
                    except (EOFError, KeyboardInterrupt) as error:
                        cancel_error = error
                        break
                    except Exception as error:
                        cancel_error = error
                        break
                    time.sleep(VOICE_STOP_POLL_INTERVAL_SECONDS)
            except KeyboardInterrupt as error:
                cancel_error = error
        except VoiceInputError:
            raise
        except Exception as error:
            logger.error(
                "Voice microphone open/capture failed error_type=%s",
                type(error).__name__,
            )
            raise VoiceInputError(VOICE_MICROPHONE_UNAVAILABLE) from error
        finally:
            recording_active.clear()
            if stream is not None:
                try:
                    stream.stop()
                except Exception as error:
                    logger.error(
                        "Voice stream stop failed error_type=%s",
                        type(error).__name__,
                    )
                try:
                    stream.close()
                except Exception as error:
                    logger.error(
                        "Voice stream close failed error_type=%s",
                        type(error).__name__,
                    )

        if cancel_error is not None:
            if isinstance(cancel_error, (EOFError, KeyboardInterrupt)):
                raise VoiceCaptureCancelledError(VOICE_CANCELLED) from cancel_error
            logger.error(
                "Voice stop signal failed error_type=%s",
                type(cancel_error).__name__,
            )
            raise VoiceInputError(VOICE_CAPTURE_FAILED) from cancel_error

        if capture_error:
            logger.error(
                "Voice capture callback failed error_type=%s",
                type(capture_error[0]).__name__,
            )
            raise VoiceInputError(VOICE_CAPTURE_FAILED) from capture_error[0]

        with chunks_lock:
            pcm_bytes = b"".join(chunks)

        if not pcm_bytes:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)
        if len(pcm_bytes) > max_pcm_bytes:
            pcm_bytes = pcm_bytes[:max_pcm_bytes]

        duration_ms = duration_ms_from_pcm(pcm_bytes)
        if duration_ms < MIN_VOICE_UTTERANCE_MS:
            raise VoiceInputValidationError(VOICE_EMPTY_AUDIO)

        wav_bytes = pcm_to_wav_bytes(pcm_bytes)
        logger.info(
            "Voice capture completed duration_ms=%s pcm_bytes=%s wav_bytes=%s",
            duration_ms,
            len(pcm_bytes),
            len(wav_bytes),
        )
        return NormalizedAudioInput(
            audio_bytes=wav_bytes,
            format="wav",
            sample_rate=self._sample_rate,
            channels=self._channels,
            sample_width_bytes=self._sample_width_bytes,
            duration_ms=duration_ms,
            source_kind="microphone",
        )
