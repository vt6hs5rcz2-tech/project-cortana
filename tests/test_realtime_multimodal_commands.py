"""Tests for Milestone 26 multimodal command gating and status."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.camera_capture import MULTIMODAL_DISABLED
from src.commands import handle_slash_command
from src.conversation import ConversationHistory
from src.document_extractor import TextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.realtime_multimodal_commands import (
    RealtimeMultimodalCommandContext,
    handle_realtime_multimodal_command,
)
from src.settings import Settings


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        transcription_model="gpt-4o-mini-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="coral",
        realtime_model="gpt-realtime-mini",
        realtime_voice="coral",
    )


def test_multimodal_gate_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.realtime_multimodal_commands.REALTIME_MULTIMODAL_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "src.camera_capture.REALTIME_MULTIMODAL_ENABLED",
        False,
    )
    result = handle_realtime_multimodal_command(
        "multimodal-realtime",
        RealtimeMultimodalCommandContext(
            message="/multimodal-realtime",
            settings=_settings(),
            client=MagicMock(),
            conversation_history=ConversationHistory(),
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("test"),
        ),
    )
    assert result is not None
    assert result.message == MULTIMODAL_DISABLED


@pytest.mark.parametrize(
    "flag_name",
    [
        "VOICE_INTERACTION_ENABLED",
        "REALTIME_VOICE_ENABLED",
        "VISION_ANALYSIS_ENABLED",
        "REALTIME_MULTIMODAL_ENABLED",
    ],
)
def test_each_parent_gate_disables_multimodal(
    monkeypatch: pytest.MonkeyPatch,
    flag_name: str,
) -> None:
    monkeypatch.setattr(
        f"src.realtime_multimodal_commands.{flag_name}",
        False,
    )
    monkeypatch.setattr(
        f"src.camera_capture.{flag_name}",
        False,
    )
    result = handle_realtime_multimodal_command(
        "multimodal-realtime",
        RealtimeMultimodalCommandContext(
            message="/multimodal-realtime",
            settings=_settings(),
            client=MagicMock(),
            conversation_history=ConversationHistory(),
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("test"),
        ),
    )
    assert result is not None
    assert result.message == MULTIMODAL_DISABLED


def test_help_and_voice_status_include_multimodal_without_devices(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom_camera(*_a: object, **_k: object) -> object:
        raise AssertionError("camera must not open")

    monkeypatch.setattr(
        "src.camera_capture._default_capture_factory",
        boom_camera,
    )

    help_result = handle_slash_command(
        "/help",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_extractor=MagicMock(spec=TextExtractor),
        client=None,
    )
    assert "/multimodal-realtime" in help_result.message

    status = handle_slash_command(
        "/voice-status",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_extractor=MagicMock(spec=TextExtractor),
        client=None,
    )
    message = status.message
    assert "Multimodal" in message
    assert "2 FPS" in message
    assert "1280x720" in message
