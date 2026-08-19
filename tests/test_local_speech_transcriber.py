from __future__ import annotations

from src.local_speech_transcriber import LocalSpeechTranscriber
from src.voice_input import NormalizedAudioInput


class FakeLocalTranscriber:
    def transcribe(self, audio: NormalizedAudioInput) -> str:
        del audio
        return "hello"


def test_local_speech_transcriber_protocol_accepts_basic_adapter() -> None:
    transcriber: LocalSpeechTranscriber = FakeLocalTranscriber()

    assert transcriber is not None
