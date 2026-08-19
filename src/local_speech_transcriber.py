from __future__ import annotations

from typing import Protocol

from src.voice_input import NormalizedAudioInput


class LocalSpeechTranscriber(Protocol):
    """Optional local speech-to-text adapter.

    Implementations receive Cortana's normalized ephemeral WAV input and
    return plain transcript text. They must not modify the input audio,
    persist microphone audio, execute commands, or alter conversation state.
    """

    def transcribe(self, audio: NormalizedAudioInput) -> str:
        """Return one transcript for normalized microphone audio."""
        ...
