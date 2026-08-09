"""Tests for Milestone 24 voice slash commands."""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.conversation import ConversationHistory
from src.settings import Settings
from src.voice_commands import (
    VoiceCommandContext,
    create_default_voice_services,
    handle_voice_command,
)
from src.voice_input import (
    MicrophoneCaptureAdapter,
    NormalizedAudioInput,
    VoiceCaptureCancelledError,
    pcm_to_wav_bytes,
)
from src.voice_service import VoiceService, VoiceServiceError


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        transcription_model="gpt-4o-mini-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="coral",
    )


def _audio() -> NormalizedAudioInput:
    pcm = b"\x00\x00" * 4800
    return NormalizedAudioInput(
        audio_bytes=pcm_to_wav_bytes(pcm),
        format="wav",
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        duration_ms=300,
        source_kind="microphone",
    )


def _context(
    *,
    capture: MicrophoneCaptureAdapter | None = None,
    voice_service: VoiceService | None = None,
    history: ConversationHistory | None = None,
    stop_signal: Any = None,
) -> VoiceCommandContext:
    return VoiceCommandContext(
        message="/voice-turn",
        settings=_settings(),
        client=cast(OpenAIClient, object()),
        conversation_history=history or ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        logger=logging.getLogger("ProjectCortanaTest"),
        stop_signal=stop_signal or (lambda: True),
        capture=capture,
        voice_service=voice_service,
    )


def test_create_default_voice_services_does_not_open_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = {"count": 0}

    class Boom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            opened["count"] += 1
            raise AssertionError("mic opened")

    monkeypatch.setitem(
        __import__("sys").modules,
        "sounddevice",
        type("SD", (), {"RawInputStream": Boom})(),
    )
    capture, service = create_default_voice_services(
        settings=_settings(),
        client=None,
    )
    assert isinstance(capture, MicrophoneCaptureAdapter)
    assert isinstance(service, VoiceService)
    assert opened["count"] == 0


def test_voice_status_does_not_open_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = {"count": 0}

    class BoomAdapter(MicrophoneCaptureAdapter):
        def capture(self, *, stop_signal: Any) -> NormalizedAudioInput:
            opened["count"] += 1
            raise AssertionError("capture must not run for status")

    result = handle_voice_command(
        "voice-status",
        _context(capture=BoomAdapter()),
    )
    assert result is not None
    assert "Voice status" in result.message
    assert "Transcription model: gpt-4o-mini-transcribe" in result.message
    assert "TTS voice: coral" in result.message
    assert opened["count"] == 0


def test_voice_turn_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "What is MFA?"
    service.synthesize.return_value = b"RIFFWAVEdata"

    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message("MFA is multi-factor authentication.")
        return "MFA is multi-factor authentication."

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    played: list[bytes] = []
    monkeypatch.setattr(
        "src.voice_commands._play_wav_synchronously",
        lambda data: played.append(data),
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(capture=capture, voice_service=service, history=history),
    )
    output = capsys.readouterr().out
    assert result is not None
    assert result.message == ""
    assert "Listening..." in output
    assert "(Heard) What is MFA?" in output
    assert "MFA is multi-factor authentication." in output
    assert played == [b"RIFFWAVEdata"]
    assert len(history.turns) == 2
    assert history.turns[0].content == "What is MFA?"
    capture.capture.assert_called_once()


def test_voice_turn_cancel_skips_stt_history_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.side_effect = VoiceCaptureCancelledError(
        "Cortana: Voice turn cancelled."
    )
    service = MagicMock(spec=VoiceService)
    process_calls = {"count": 0}

    def boom_process(**kwargs: object) -> str | None:
        process_calls["count"] += 1
        return "nope"

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        boom_process,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(capture=capture, voice_service=service, history=history),
    )
    assert result is not None
    assert result.message == "Cortana: Voice turn cancelled."
    assert history.turns == []
    assert process_calls["count"] == 0
    service.transcribe.assert_not_called()
    service.synthesize.assert_not_called()


def test_voice_turn_tts_failure_keeps_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "hello"
    service.synthesize.side_effect = VoiceServiceError(
        "Cortana: I could not generate speech for that response."
    )
    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message("Hello back.")
        return "Hello back."

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(capture=capture, voice_service=service, history=history),
    )
    output = capsys.readouterr().out
    assert "Hello back." in output
    assert result is not None
    assert "could not generate speech" in result.message
    assert len(history.turns) == 2


def test_voice_turn_chat_failure_skips_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "hello"
    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(capture=capture, voice_service=service, history=history),
    )
    assert result is not None
    assert "could not complete" in result.message
    assert history.turns == []
    service.synthesize.assert_not_called()
