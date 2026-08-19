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
    canonicalize_wav_bytes,
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


def _streamed_wav(pcm: bytes = b"\x00\x00" * 24) -> bytes:
    valid = pcm_to_wav_bytes(pcm)
    header = bytearray(valid[:44])
    header[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    header[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
    return bytes(header) + valid[44:]


def test_canonicalize_wav_repairs_streamed_sizes() -> None:
    pcm = b"\x01\x00" * 48
    streamed = _streamed_wav(pcm)
    assert int.from_bytes(streamed[4:8], "little") == 0xFFFFFFFF
    assert int.from_bytes(streamed[40:44], "little") == 0xFFFFFFFF
    repaired = canonicalize_wav_bytes(streamed)
    assert repaired[:4] == b"RIFF"
    assert repaired[8:12] == b"WAVE"
    assert int.from_bytes(repaired[4:8], "little") == len(repaired) - 8
    assert int.from_bytes(repaired[40:44], "little") == len(pcm)
    assert repaired[44:] == pcm


def test_canonicalize_wav_leaves_valid_and_short_buffers() -> None:
    valid = pcm_to_wav_bytes(b"\x00\x00" * 16)
    assert canonicalize_wav_bytes(valid) == valid
    short = b"RIFF....WAVEfmt "
    assert canonicalize_wav_bytes(short) == short
    assert canonicalize_wav_bytes(b"not-a-wav-file-at-all!!!!") == (
        b"not-a-wav-file-at-all!!!!"
    )


def test_synthesize_returns_wav_bytes() -> None:
    audio_api = _FakeAudio()
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    content = service.synthesize("Hello from Cortana.")
    assert content == audio_api.speech_bytes
    assert audio_api.last_speech_input == "Hello from Cortana."


def test_synthesize_repairs_streamed_provider_wav() -> None:
    audio_api = _FakeAudio()
    audio_api.speech_bytes = _streamed_wav(b"\x02\x00" * 32)
    service = VoiceService(settings=_settings(), client=_FakeClient(audio_api))
    content = service.synthesize("Hello from Cortana.")
    assert int.from_bytes(content[4:8], "little") == len(content) - 8
    assert int.from_bytes(content[40:44], "little") == 64
    assert content[44:] == b"\x02\x00" * 32


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

class _FakeLocalTranscriber:
    def __init__(self) -> None:
        self.text = "local transcript"
        self.error: Exception | None = None
        self.received: NormalizedAudioInput | None = None

    def transcribe(self, audio: NormalizedAudioInput) -> str:
        self.received = audio
        if self.error is not None:
            raise self.error
        return self.text


def test_transcribe_uses_local_transcriber_when_supplied() -> None:
    local = _FakeLocalTranscriber()
    local.text = "  local hello  "

    service = VoiceService(
        settings=_settings(),
        client=None,
        local_transcriber=local,
    )

    audio = _audio()
    text = service.transcribe(audio)

    assert text == "local hello"
    assert local.received is audio


def test_transcribe_local_failure_maps_to_voice_service_error() -> None:
    local = _FakeLocalTranscriber()
    local.error = RuntimeError("local stt failed")

    service = VoiceService(
        settings=_settings(),
        client=None,
        local_transcriber=local,
    )

    with pytest.raises(VoiceServiceError, match="could not transcribe"):
        service.transcribe(_audio())


def test_transcribe_local_rejects_blank_and_oversized() -> None:
    local = _FakeLocalTranscriber()

    service = VoiceService(
        settings=_settings(),
        client=None,
        local_transcriber=local,
    )

    local.text = "   "
    with pytest.raises(VoiceServiceValidationError):
        service.transcribe(_audio())

    local.text = "x" * (MAX_VOICE_TRANSCRIPT_CHARS + 1)
    with pytest.raises(VoiceServiceValidationError):
        service.transcribe(_audio())


def test_transcribe_without_local_transcriber_still_uses_provider() -> None:
    audio_api = _FakeAudio()
    audio_api.transcribe_text = "provider transcript"

    service = VoiceService(
        settings=_settings(),
        client=_FakeClient(audio_api),
    )

    assert service.transcribe(_audio()) == "provider transcript"
    assert audio_api.last_file is not None

def test_transcribe_local_failure_does_not_fall_back_to_provider() -> None:
    local = _FakeLocalTranscriber()
    local.error = RuntimeError("local stt failed")

    audio_api = _FakeAudio()
    audio_api.transcribe_text = "provider transcript"

    service = VoiceService(
        settings=_settings(),
        client=_FakeClient(audio_api),
        local_transcriber=local,
    )

    with pytest.raises(VoiceServiceError, match="could not transcribe"):
        service.transcribe(_audio())

    assert audio_api.last_file is None
