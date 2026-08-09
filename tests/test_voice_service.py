"""Tests for Milestone 24 STT/TTS voice service."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest

from src.config import MAX_TTS_CHARS, MAX_VOICE_TRANSCRIPT_CHARS
from src.settings import Settings
from src.voice_input import NormalizedAudioInput, pcm_to_wav_bytes
from src.voice_service import (
    VoiceService,
    VoiceServiceError,
    VoiceServiceValidationError,
)


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        transcription_model="gpt-4o-mini-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="coral",
    )


def _audio() -> NormalizedAudioInput:
    pcm = b"\x00\x00" * 4800  # 300ms
    wav = pcm_to_wav_bytes(pcm)
    return NormalizedAudioInput(
        audio_bytes=wav,
        format="wav",
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        duration_ms=300,
        source_kind="microphone",
    )


class _Transcription:
    def __init__(self, text: str) -> None:
        self.text = text


class _Speech:
    def __init__(self, content: bytes) -> None:
        self.content = content


class _FakeAudio:
    def __init__(self) -> None:
        self.transcriptions = self
        self.speech = self
        self.last_file: tuple[str, BytesIO, str] | None = None
        self.last_speech_input: str | None = None
        self.transcribe_text = "hello there"
        self.speech_bytes = b"RIFF....WAVEfmt "
        self.transcribe_error: Exception | None = None
        self.speech_error: Exception | None = None

    def create(self, **kwargs: Any) -> Any:
        if "file" in kwargs:
            if self.transcribe_error is not None:
                raise self.transcribe_error
            self.last_file = kwargs["file"]
            assert kwargs["model"] == "gpt-4o-mini-transcribe"
            return _Transcription(self.transcribe_text)
        if self.speech_error is not None:
            raise self.speech_error
        self.last_speech_input = kwargs["input"]
        assert kwargs["model"] == "gpt-4o-mini-tts"
        assert kwargs["voice"] == "coral"
        assert kwargs["response_format"] == "wav"
        return _Speech(self.speech_bytes)


class _FakeClient:
    def __init__(self, audio: _FakeAudio) -> None:
        self.audio = audio


def test_voice_service_init_does_not_require_audio_client_methods() -> None:
    service = VoiceService(settings=_settings(), client=None)
    assert service is not None


def test_transcribe_returns_stripped_text() -> None:
    audio_api = _FakeAudio()
    audio_api.transcribe_text = "  hello world  "
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    text = service.transcribe(_audio())
    assert text == "hello world"
    assert audio_api.last_file is not None
    assert audio_api.last_file[0] == "speech.wav"
    assert audio_api.last_file[2] == "audio/wav"
    assert isinstance(audio_api.last_file[1], BytesIO)


def test_transcribe_rejects_blank_and_oversized() -> None:
    audio_api = _FakeAudio()
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))

    audio_api.transcribe_text = "   "
    with pytest.raises(VoiceServiceValidationError):
        service.transcribe(_audio())

    audio_api.transcribe_text = "x" * (MAX_VOICE_TRANSCRIPT_CHARS + 1)
    with pytest.raises(VoiceServiceValidationError):
        service.transcribe(_audio())


def test_transcribe_maps_provider_errors() -> None:
    audio_api = _FakeAudio()
    audio_api.transcribe_error = RuntimeError("provider boom")
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    with pytest.raises(VoiceServiceError, match="could not transcribe"):
        service.transcribe(_audio())


def test_synthesize_returns_wav_bytes() -> None:
    audio_api = _FakeAudio()
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    content = service.synthesize("Hello from Cortana.")
    assert content == audio_api.speech_bytes
    assert audio_api.last_speech_input == "Hello from Cortana."


def test_synthesize_rejects_oversized_without_truncation() -> None:
    audio_api = _FakeAudio()
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    huge = "a" * (MAX_TTS_CHARS + 1)
    with pytest.raises(VoiceServiceValidationError, match="could not generate speech"):
        service.synthesize(huge)
    assert audio_api.last_speech_input is None


def test_synthesize_maps_provider_errors() -> None:
    audio_api = _FakeAudio()
    audio_api.speech_error = RuntimeError("tts boom")
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    with pytest.raises(VoiceServiceError, match="could not generate speech"):
        service.synthesize("Hello")
