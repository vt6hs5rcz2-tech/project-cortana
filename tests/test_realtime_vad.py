from __future__ import annotations

from src.realtime_vad import RealtimeVadObservation


def test_realtime_vad_observation_is_read_only_data() -> None:
    observation = RealtimeVadObservation(
        speech_probability=0.91,
        is_speech=True,
    )

    assert observation.speech_probability == 0.91
    assert observation.is_speech is True
