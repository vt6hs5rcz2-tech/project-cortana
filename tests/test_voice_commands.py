"""Tests for Milestone 24 voice slash commands."""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.conversation import ConversationHistory
from src.conversation_state import ConversationState
from src.settings import Settings
from src.speech_delivery import SpeechDeliveryState
from src.voice_commands import (
    VOICE_EMPTY_TRANSCRIPT,
    VOICE_PLAYBACK_FAILED,
    VoiceCommandContext,
    _play_wav_synchronously,
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
    speech_delivery_state: SpeechDeliveryState | None = None,
    conversation_state: ConversationState | None = None,
) -> VoiceCommandContext:
    return VoiceCommandContext(
        message="/voice-turn",
        settings=_settings(),
        client=cast(OpenAIClient, object()),
        conversation_history=history or ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        logger=logging.getLogger("ProjectCortanaTest"),
        stop_signal=stop_signal or (lambda: False),
        capture=capture,
        voice_service=voice_service,
        speech_delivery_state=speech_delivery_state,
        conversation_state=conversation_state,
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


def test_voice_turn_uses_centralized_speech_delivery_without_changing_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = ConversationHistory()
    delivery_state = SpeechDeliveryState()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "What are the steps?"
    spoken_inputs: list[str] = []

    def fake_synthesize(text: str) -> bytes:
        spoken_inputs.append(text)
        return b"RIFFWAVEdata"

    service.synthesize.side_effect = fake_synthesize
    canonical = (
        "## Steps\n"
        "1. Back up files.\n"
        "2. Update software.\n"
        "3. Restart."
    )

    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message(canonical)
        return canonical

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
        _context(
            capture=capture,
            voice_service=service,
            history=history,
            stop_signal=lambda: False,
            speech_delivery_state=delivery_state,
        ),
    )
    output = capsys.readouterr().out
    assert result is not None
    assert result.message == ""
    assert canonical in output
    assert history.turns[-1].content == canonical
    assert spoken_inputs
    assert spoken_inputs != [canonical]
    assert any("First," in item for item in spoken_inputs)
    assert "##" not in "".join(spoken_inputs)
    assert played


def test_voice_turn_stops_remaining_chunks_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = ConversationHistory()
    delivery_state = SpeechDeliveryState()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "Explain the options."
    calls = {"count": 0}

    def fake_synthesize(text: str) -> bytes:
        calls["count"] += 1
        return b"RIFFWAVEdata"

    service.synthesize.side_effect = fake_synthesize
    long_answer = (
        "Alpha is the first idea and it continues with extra spoken words. "
        "Beta is the second idea and it also needs extra spoken words. "
        "Gamma is the third idea with still more spoken detail included. "
        "Delta is the fourth idea that keeps the utterance moving forward. "
        "Epsilon is the fifth idea for additional spoken pacing coverage. "
        "Zeta is the sixth idea so remaining chunks can be cancelled."
    )
    played = {"count": 0}

    def fake_play(_data: bytes) -> None:
        played["count"] += 1

    started = {"play_seen": False}

    def stop_signal() -> bool:
        return started["play_seen"]

    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message(long_answer)
        return long_answer

    def recording_play(data: bytes) -> None:
        fake_play(data)
        started["play_seen"] = True

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    monkeypatch.setattr(
        "src.voice_commands._play_wav_synchronously",
        recording_play,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(
            capture=capture,
            voice_service=service,
            history=history,
            stop_signal=stop_signal,
            speech_delivery_state=delivery_state,
        ),
    )
    assert result is not None
    assert history.turns[-1].content == long_answer
    assert calls["count"] == 1
    assert played["count"] == 1
    assert delivery_state.pop_pending_chunk() is None


def test_voice_turn_cancel_before_first_chunk_plays_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = ConversationHistory()
    delivery_state = SpeechDeliveryState()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "Explain the options."
    played: list[bytes] = []

    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message("The first option is daily backups.")
        return "The first option is daily backups."

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    monkeypatch.setattr(
        "src.voice_commands._play_wav_synchronously",
        lambda data: played.append(data),
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(
            capture=capture,
            voice_service=service,
            history=history,
            stop_signal=lambda: True,
            speech_delivery_state=delivery_state,
        ),
    )
    assert result is not None
    assert result.message == ""
    assert history.turns[-1].content == "The first option is daily backups."
    service.synthesize.assert_not_called()
    assert played == []
    assert delivery_state.pop_pending_chunk() is None


@pytest.mark.parametrize("transcript", ["", "   ", "\n\t  \n"])
def test_voice_turn_blank_transcript_skips_chat_history_and_tts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    transcript: str,
) -> None:
    history = ConversationHistory()
    state = ConversationState()
    state.set_topic("keep this topic")
    prior = state.snapshot()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = transcript
    process_calls = {"count": 0}

    def boom_process(**kwargs: object) -> str | None:
        process_calls["count"] += 1
        return "should not chat"

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        boom_process,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(
            capture=capture,
            voice_service=service,
            history=history,
            conversation_state=state,
        ),
    )
    assert result is not None
    assert result.message == VOICE_EMPTY_TRANSCRIPT
    assert history.turns == []
    assert state.snapshot() == prior
    assert process_calls["count"] == 0
    service.synthesize.assert_not_called()
    service.transcribe.assert_called_once()
    assert "(Heard)" not in capsys.readouterr().out


def test_voice_turn_playback_failure_keeps_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "What is two plus two?"
    service.synthesize.return_value = b"RIFFWAVEdata"

    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message("4")
        return "4"

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    monkeypatch.setattr(
        "src.voice_commands._play_wav_synchronously",
        lambda _data: (_ for _ in ()).throw(VoiceServiceError(VOICE_PLAYBACK_FAILED)),
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    result = handle_voice_command(
        "voice-turn",
        _context(capture=capture, voice_service=service, history=history),
    )
    output = capsys.readouterr().out
    assert result is not None
    assert result.message == VOICE_PLAYBACK_FAILED
    assert "(Heard) What is two plus two?" in output
    assert "Cortana: 4" in output
    assert history.turns[-1].content == "4"
    service.transcribe.assert_called_once()
    service.synthesize.assert_called_once()


def test_play_wav_canonicalizes_streamed_header_before_winsound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = pcm_to_wav_bytes(b"\x00\x00" * 24)
    streamed = bytearray(valid)
    streamed[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    streamed[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
    played: list[bytes] = []

    class FakeWinsound:
        SND_MEMORY = 4

        @staticmethod
        def PlaySound(sound: bytes, flags: int) -> None:
            played.append(sound)
            assert flags == 4

    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")
    monkeypatch.setitem(__import__("sys").modules, "winsound", FakeWinsound)

    _play_wav_synchronously(bytes(streamed))
    assert len(played) == 1
    assert int.from_bytes(played[0][4:8], "little") == len(played[0]) - 8
    assert int.from_bytes(played[0][40:44], "little") == 48


def test_play_wav_maps_backend_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWinsound:
        SND_MEMORY = 4

        @staticmethod
        def PlaySound(sound: bytes, flags: int) -> None:
            raise RuntimeError("Failed to play sound")

    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")
    monkeypatch.setitem(__import__("sys").modules, "winsound", FakeWinsound)

    with pytest.raises(VoiceServiceError, match="could not play the spoken response"):
        _play_wav_synchronously(pcm_to_wav_bytes(b"\x00\x00" * 8))

def test_optional_local_stt_factory_disabled_returns_none(monkeypatch) -> None:
    from src.config import LOCAL_STT_ENV
    from src.voice_commands import _build_optional_local_transcriber

    monkeypatch.delenv(LOCAL_STT_ENV, raising=False)

    assert _build_optional_local_transcriber() is None


def test_optional_local_stt_factory_missing_paths_returns_none(monkeypatch) -> None:
    from src.config import (
        LOCAL_STT_ENV,
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        LOCAL_STT_MODEL_PATH_ENV,
    )
    from src.voice_commands import _build_optional_local_transcriber

    monkeypatch.setenv(LOCAL_STT_ENV, "true")
    monkeypatch.delenv(LOCAL_STT_EXECUTABLE_PATH_ENV, raising=False)
    monkeypatch.delenv(LOCAL_STT_MODEL_PATH_ENV, raising=False)

    assert _build_optional_local_transcriber() is None


def test_optional_local_stt_factory_missing_executable_returns_none(
    monkeypatch,
    tmp_path,
) -> None:
    from src.config import (
        LOCAL_STT_ENV,
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        LOCAL_STT_MODEL_PATH_ENV,
    )
    from src.voice_commands import _build_optional_local_transcriber

    missing_executable = tmp_path / "missing-whisper-cli.exe"
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")

    monkeypatch.setenv(LOCAL_STT_ENV, "true")
    monkeypatch.setenv(
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        str(missing_executable),
    )
    monkeypatch.setenv(
        LOCAL_STT_MODEL_PATH_ENV,
        str(model_path),
    )

    assert _build_optional_local_transcriber() is None


def test_optional_local_stt_factory_missing_model_returns_none(
    monkeypatch,
    tmp_path,
) -> None:
    from src.config import (
        LOCAL_STT_ENV,
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        LOCAL_STT_MODEL_PATH_ENV,
    )
    from src.voice_commands import _build_optional_local_transcriber

    executable_path = tmp_path / "whisper-cli.exe"
    executable_path.write_bytes(b"exe")
    missing_model = tmp_path / "missing-model.bin"

    monkeypatch.setenv(LOCAL_STT_ENV, "true")
    monkeypatch.setenv(
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        str(executable_path),
    )
    monkeypatch.setenv(
        LOCAL_STT_MODEL_PATH_ENV,
        str(missing_model),
    )

    assert _build_optional_local_transcriber() is None

def test_optional_local_stt_factory_valid_paths_returns_transcriber(
    monkeypatch,
    tmp_path,
) -> None:
    from src.config import (
        LOCAL_STT_ENV,
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        LOCAL_STT_MODEL_PATH_ENV,
    )
    from src.voice_commands import _build_optional_local_transcriber
    from src.whisper_cpp_transcriber import WhisperCppTranscriber

    executable_path = tmp_path / "whisper-cli.exe"
    model_path = tmp_path / "ggml-small.en.bin"

    executable_path.write_bytes(b"exe")
    model_path.write_bytes(b"model")

    monkeypatch.setenv(LOCAL_STT_ENV, "true")
    monkeypatch.setenv(
        LOCAL_STT_EXECUTABLE_PATH_ENV,
        str(executable_path),
    )
    monkeypatch.setenv(
        LOCAL_STT_MODEL_PATH_ENV,
        str(model_path),
    )

    result = _build_optional_local_transcriber()

    assert isinstance(result, WhisperCppTranscriber)
