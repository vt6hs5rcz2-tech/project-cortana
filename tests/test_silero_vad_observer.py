from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.realtime_voice_input import RealtimeAudioFrame
from src.silero_vad_observer import SileroVadObserver


class FakeSession:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        output_names: object,
        inputs: dict[str, Any],
    ) -> list[Any]:
        del output_names
        self.calls.append(inputs)

        output = np.array([[self.probability]], dtype=np.float32)
        state = np.ones((2, 1, 128), dtype=np.float32)

        return [output, state]


def _frame(sequence: int = 0) -> RealtimeAudioFrame:
    pcm = np.zeros(480, dtype=np.int16).tobytes()

    return RealtimeAudioFrame(
        pcm_bytes=pcm,
        sample_rate=24_000,
        channels=1,
        sample_width_bytes=2,
        sequence=sequence,
    )


def test_silero_observer_buffers_until_full_window() -> None:
    fake = FakeSession(0.90)

    observer = SileroVadObserver(
        model_path=Path("fake.onnx"),
        session_factory=lambda _path: fake,
    )

    assert observer.observe(_frame(0)) is None

    result = observer.observe(_frame(1))

    assert result is not None
    assert result.is_speech is True
    assert result.speech_probability > 0.89
    assert len(fake.calls) == 1


def test_silero_observer_threshold_marks_low_probability_as_non_speech() -> None:
    fake = FakeSession(0.25)

    observer = SileroVadObserver(
        model_path=Path("fake.onnx"),
        speech_threshold=0.50,
        session_factory=lambda _path: fake,
    )

    assert observer.observe(_frame(0)) is None
    result = observer.observe(_frame(1))

    assert result is not None
    assert result.is_speech is False
    assert result.speech_probability == np.float32(0.25)


def test_silero_observer_passes_context_state_and_sample_rate() -> None:
    fake = FakeSession(0.75)

    observer = SileroVadObserver(
        model_path=Path("fake.onnx"),
        session_factory=lambda _path: fake,
    )

    observer.observe(_frame(0))
    observer.observe(_frame(1))

    assert len(fake.calls) == 1

    inputs = fake.calls[0]

    assert inputs["input"].shape == (1, 576)
    assert inputs["state"].shape == (2, 1, 128)
    assert int(inputs["sr"]) == 16_000


def test_silero_observer_rejects_invalid_threshold() -> None:
    try:
        SileroVadObserver(
            model_path=Path("fake.onnx"),
            speech_threshold=1.5,
            session_factory=lambda _path: FakeSession(0.5),
        )
    except ValueError as error:
        assert "speech_threshold" in str(error)
    else:
        raise AssertionError("Expected invalid speech threshold to fail.")
