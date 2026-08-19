from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.realtime_voice_input import RealtimeAudioFrame


@dataclass(frozen=True)
class RealtimeVadObservation:
    """One non-authoritative local VAD observation."""

    speech_probability: float
    is_speech: bool


class RealtimeVadObserver(Protocol):
    """Read-only observer for realtime microphone frames.

    Implementations may inspect audio and report speech likelihood, but they
    must not create turns, stop turns, interrupt responses, or modify audio.
    """

    def observe(self, frame: RealtimeAudioFrame) -> RealtimeVadObservation | None:
        """Inspect one frame and optionally return a VAD observation."""
        ...


def optional_vad_dependencies_available() -> bool:
    """Return True when optional local VAD dependencies can be imported."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True
