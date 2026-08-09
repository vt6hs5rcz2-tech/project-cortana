"""Bounded realtime microphone streaming for Milestone 25.

Opens the default microphone only inside ``RealtimeMicrophoneStream.start``.
Construction and import never open a device. Frames are canonical 20 ms PCM at
24 kHz mono int16. NumPy is not required.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Full, Queue

from src.config import (
    REALTIME_VOICE_CAPTURE_BLOCKSIZE,
    REALTIME_VOICE_CHANNELS,
    REALTIME_VOICE_ENABLED,
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_INPUT_QUEUE_FRAMES,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
    REALTIME_VOICE_SAMPLE_WIDTH_BYTES,
    VOICE_INTERACTION_ENABLED,
)

logger = logging.getLogger("ProjectCortana")

REALTIME_VOICE_DISABLED = (
    "Cortana: Realtime voice interaction is currently disabled."
)
REALTIME_UNSUPPORTED_PLATFORM = (
    "Cortana: Realtime voice requires Windows in this milestone."
)
REALTIME_MICROPHONE_UNAVAILABLE = (
    "Cortana: No usable microphone is available for realtime voice."
)
REALTIME_MICROPHONE_FAILED = "Cortana: Realtime microphone capture failed."
REALTIME_INPUT_OVERFLOW = (
    "Cortana: Realtime microphone audio overflowed local bounds."
)
REALTIME_INVALID_FRAME = "Cortana: Realtime audio frame validation failed."


class RealtimeVoiceInputError(RuntimeError):
    """Raised when realtime microphone capture cannot continue safely."""

    def __init__(self, user_message: str, *, error_type: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.error_type = error_type


class RealtimeVoiceInputDisabledError(RealtimeVoiceInputError):
    """Raised when realtime or parent voice interaction is disabled."""


class RealtimeVoiceInputValidationError(RealtimeVoiceInputError):
    """Raised when a realtime audio frame fails validation."""


@dataclass(frozen=True)
class RealtimeAudioFrame:
    """One canonical realtime PCM frame. Contains no device or path metadata."""

    pcm_bytes: bytes
    sample_rate: int
    channels: int
    sample_width_bytes: int
    sequence: int

    def __post_init__(self) -> None:
        if self.sample_rate != REALTIME_VOICE_SAMPLE_RATE_HZ:
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )
        if self.channels != REALTIME_VOICE_CHANNELS:
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )
        if self.sample_width_bytes != REALTIME_VOICE_SAMPLE_WIDTH_BYTES:
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )
        if not isinstance(self.pcm_bytes, bytes):
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )
        if len(self.pcm_bytes) != REALTIME_VOICE_FRAME_BYTES:
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )
        if self.sequence < 0:
            raise RealtimeVoiceInputValidationError(
                REALTIME_INVALID_FRAME,
                error_type="microphone_failed",
            )


def realtime_voice_features_enabled() -> bool:
    """Return True only when both parent and realtime voice gates are enabled."""
    return bool(VOICE_INTERACTION_ENABLED and REALTIME_VOICE_ENABLED)


class RealtimeMicrophoneStream:
    """Stream fixed-size PCM frames from the default microphone into a queue.

    ``__init__`` stores configuration only and never queries or opens a device.
    """

    def __init__(
        self,
        *,
        frame_queue: Queue[RealtimeAudioFrame],
        on_overflow: Callable[[], None] | None = None,
        on_capture_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._frame_queue = frame_queue
        self._on_overflow = on_overflow
        self._on_capture_error = on_capture_error
        self._stream: object | None = None
        self._active = threading.Event()
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        """Return True while the microphone stream is open and accepting frames."""
        return self._active.is_set()

    def start(self) -> None:
        """Open the default microphone and begin enqueueing frames."""
        if not realtime_voice_features_enabled():
            raise RealtimeVoiceInputDisabledError(
                REALTIME_VOICE_DISABLED,
                error_type="realtime_unavailable",
            )
        if sys.platform != "win32":
            raise RealtimeVoiceInputValidationError(
                REALTIME_UNSUPPORTED_PLATFORM,
                error_type="unsupported_platform",
            )

        with self._lock:
            if self._stream is not None:
                raise RealtimeVoiceInputError(
                    REALTIME_MICROPHONE_FAILED,
                    error_type="microphone_failed",
                )

            try:
                import sounddevice as sd  # type: ignore[import-untyped]
            except Exception as error:
                logger.error(
                    "Realtime sounddevice import failed error_type=%s",
                    type(error).__name__,
                )
                raise RealtimeVoiceInputError(
                    REALTIME_MICROPHONE_UNAVAILABLE,
                    error_type="microphone_failed",
                ) from error

            self._sequence = 0
            self._active.set()

            def callback(
                indata: object,
                frames: int,
                time_info: object,
                status: object,
            ) -> None:
                del frames, time_info, status
                if not self._active.is_set():
                    return
                try:
                    pcm = memoryview(indata).tobytes()  # type: ignore[arg-type]
                    if len(pcm) != REALTIME_VOICE_FRAME_BYTES:
                        raise RealtimeVoiceInputValidationError(
                            REALTIME_INVALID_FRAME,
                            error_type="microphone_failed",
                        )
                    frame = RealtimeAudioFrame(
                        pcm_bytes=pcm,
                        sample_rate=REALTIME_VOICE_SAMPLE_RATE_HZ,
                        channels=REALTIME_VOICE_CHANNELS,
                        sample_width_bytes=REALTIME_VOICE_SAMPLE_WIDTH_BYTES,
                        sequence=self._sequence,
                    )
                    self._sequence += 1
                    self._frame_queue.put_nowait(frame)
                except Full:
                    self._active.clear()
                    logger.error(
                        "Realtime microphone input overflow max_frames=%s",
                        REALTIME_VOICE_INPUT_QUEUE_FRAMES,
                    )
                    if self._on_overflow is not None:
                        self._on_overflow()
                except Exception as error:
                    self._active.clear()
                    logger.error(
                        "Realtime microphone callback failed error_type=%s",
                        type(error).__name__,
                    )
                    if self._on_capture_error is not None:
                        self._on_capture_error(error)

            try:
                stream = sd.RawInputStream(
                    samplerate=REALTIME_VOICE_SAMPLE_RATE_HZ,
                    blocksize=REALTIME_VOICE_CAPTURE_BLOCKSIZE,
                    channels=REALTIME_VOICE_CHANNELS,
                    dtype="int16",
                    callback=callback,
                )
                stream.start()
            except Exception as error:
                self._active.clear()
                logger.error(
                    "Realtime microphone open failed error_type=%s",
                    type(error).__name__,
                )
                raise RealtimeVoiceInputError(
                    REALTIME_MICROPHONE_UNAVAILABLE,
                    error_type="microphone_failed",
                ) from error
            self._stream = stream
            logger.info(
                "Realtime microphone started sample_rate=%s frame_bytes=%s",
                REALTIME_VOICE_SAMPLE_RATE_HZ,
                REALTIME_VOICE_FRAME_BYTES,
            )

    def stop(self) -> None:
        """Stop accepting frames and close the microphone stream."""
        with self._lock:
            self._active.clear()
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        try:
            stop = getattr(stream, "stop", None)
            if callable(stop):
                stop()
        except Exception as error:
            logger.error(
                "Realtime microphone stop failed error_type=%s",
                type(error).__name__,
            )
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        except Exception as error:
            logger.error(
                "Realtime microphone close failed error_type=%s",
                type(error).__name__,
            )
        logger.info("Realtime microphone stopped")
