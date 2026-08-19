from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from src.realtime_vad import RealtimeVadObservation
from src.realtime_voice_input import RealtimeAudioFrame


SILERO_SAMPLE_RATE_HZ = 16_000
SILERO_WINDOW_SAMPLES = 512
SILERO_CONTEXT_SAMPLES = 64
DEFAULT_SPEECH_THRESHOLD = 0.50

FloatArray = NDArray[np.float32]


class SileroVadObserver:
    """Read-only Silero ONNX observer for realtime microphone audio."""

    def __init__(
        self,
        *,
        model_path: Path,
        speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0.0 <= speech_threshold <= 1.0:
            raise ValueError("speech_threshold must be between 0.0 and 1.0.")

        self._speech_threshold = speech_threshold

        if session_factory is None:
            try:
                import onnxruntime as ort  # type: ignore[import-untyped]
            except ImportError as error:
                raise RuntimeError(
                    "Optional local VAD dependency onnxruntime is unavailable."
                ) from error

            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            self._session = session_factory(str(model_path))

        self._state: FloatArray = np.zeros(
            (2, 1, 128), dtype=np.float32
        )
        self._context: FloatArray = np.zeros(
            (1, SILERO_CONTEXT_SAMPLES),
            dtype=np.float32,
        )
        self._buffer: FloatArray = np.empty(0, dtype=np.float32)

    def observe(
        self,
        frame: RealtimeAudioFrame,
    ) -> RealtimeVadObservation | None:
        samples = np.frombuffer(
            frame.pcm_bytes,
            dtype=np.int16,
        ).astype(np.float32)

        samples /= 32768.0

        converted = self._resample_24k_to_16k(samples)
        self._buffer = np.concatenate((self._buffer, converted))

        if len(self._buffer) < SILERO_WINDOW_SAMPLES:
            return None

        window = self._buffer[:SILERO_WINDOW_SAMPLES]
        self._buffer = self._buffer[SILERO_WINDOW_SAMPLES:]

        model_input = np.concatenate(
            (
                self._context,
                window.reshape(1, -1),
            ),
            axis=1,
        )

        outputs: list[Any] = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(SILERO_SAMPLE_RATE_HZ, dtype=np.int64),
            },
        )

        output = cast(FloatArray, outputs[0])
        state = cast(FloatArray, outputs[1])

        self._state = state
        self._context = model_input[:, -SILERO_CONTEXT_SAMPLES:]

        probability = float(output[0][0])

        return RealtimeVadObservation(
            speech_probability=probability,
            is_speech=probability >= self._speech_threshold,
        )

    @staticmethod
    def _resample_24k_to_16k(samples: FloatArray) -> FloatArray:
        target_length = int(
            round(len(samples) * SILERO_SAMPLE_RATE_HZ / 24_000)
        )

        source_positions = np.arange(
            len(samples),
            dtype=np.float32,
        )
        target_positions = np.linspace(
            0,
            len(samples) - 1,
            target_length,
            dtype=np.float32,
        )

        return cast(
            FloatArray,
            np.interp(
                target_positions,
                source_positions,
                samples,
            ).astype(np.float32),
        )
