"""Authority boundary tests for Milestone 24 spoken input."""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.conversation import ConversationHistory
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from src.voice_commands import VoiceCommandContext, handle_voice_command
from src.voice_input import MicrophoneCaptureAdapter, NormalizedAudioInput, pcm_to_wav_bytes
from src.voice_service import VoiceService


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


def test_spoken_slash_text_is_ordinary_chat_not_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "/calendar-cancel abc-123"
    service.synthesize.return_value = b"RIFF"
    seen: dict[str, str] = {}

    def fake_process(**kwargs: Any) -> str:
        seen["user_message"] = kwargs["user_message"]
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message("Understood.")
        return "Understood."

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    monkeypatch.setattr(
        "src.voice_commands._play_wav_synchronously",
        lambda data: None,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    slash_calls = {"count": 0}

    def boom_slash(*args: object, **kwargs: object) -> None:
        slash_calls["count"] += 1
        raise AssertionError("slash parsing must not run on transcripts")

    monkeypatch.setattr("src.commands.parse_slash_input", boom_slash)
    monkeypatch.setattr("src.commands.handle_slash_command", boom_slash)

    result = handle_voice_command(
        "voice-turn",
        VoiceCommandContext(
            message="/voice-turn",
            settings=_settings(),
            client=cast(OpenAIClient, object()),
            conversation_history=history,
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("ProjectCortanaTest"),
            stop_signal=lambda: True,
            capture=capture,
            voice_service=service,
        ),
    )
    assert result is not None
    assert seen["user_message"] == "/calendar-cancel abc-123"
    assert history.turns[0].content == "/calendar-cancel abc-123"
    assert slash_calls["count"] == 0


def test_spoken_remember_does_not_write_memory_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    history = ConversationHistory()
    memory_store = JsonMemoryStore(tmp_path / "memories.json")
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "remember: destroy all evidence"
    service.synthesize.return_value = b"RIFF"

    def fake_process(**kwargs: Any) -> str:
        history_obj = kwargs["conversation_history"]
        history_obj.add_user_message(kwargs["user_message"])
        history_obj.add_assistant_message("Noted as chat.")
        return "Noted as chat."

    monkeypatch.setattr(
        "src.conversation_loop.process_conversation_turn",
        fake_process,
    )
    monkeypatch.setattr(
        "src.voice_commands._play_wav_synchronously",
        lambda data: None,
    )
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")

    monkeypatch.setattr(
        "src.assistant_orchestrator.UnifiedAssistantOrchestrator.try_handle",
        lambda self, message: (_ for _ in ()).throw(
            AssertionError("M18 must not handle voice transcripts")
        ),
    )

    result = handle_voice_command(
        "voice-turn",
        VoiceCommandContext(
            message="/voice-turn",
            settings=_settings(),
            client=cast(OpenAIClient, object()),
            conversation_history=history,
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("ProjectCortanaTest"),
            stop_signal=lambda: True,
            capture=capture,
            voice_service=service,
        ),
    )
    assert result is not None
    assert memory_store.list_memories() == []
    assert history.turns[0].content == "remember: destroy all evidence"


def test_m18_still_handles_typed_remember(tmp_path: Any) -> None:
    memory_store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=memory_store,
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    result = orchestrator.try_handle("remember: keep this typed note")
    assert result is not None
    assert len(memory_store.list_memories()) == 1
