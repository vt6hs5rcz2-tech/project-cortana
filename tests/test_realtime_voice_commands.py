"""Tests for Milestone 25 realtime voice slash commands."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.commands import handle_slash_command
from src.conversation import ConversationHistory
from src.document_extractor import TextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.realtime_voice_commands import (
    RealtimeVoiceCommandContext,
    handle_realtime_voice_command,
)
from src.realtime_voice_input import REALTIME_VOICE_DISABLED
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


def test_voice_realtime_disabled_by_child_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.realtime_voice_commands.REALTIME_VOICE_ENABLED",
        False,
    )
    result = handle_realtime_voice_command(
        "voice-realtime",
        RealtimeVoiceCommandContext(
            message="/voice-realtime",
            settings=_settings(),
            client=MagicMock(),
            conversation_history=ConversationHistory(),
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("test"),
        ),
    )
    assert result is not None
    assert result.message == REALTIME_VOICE_DISABLED


def test_voice_realtime_disabled_when_parent_flag_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.realtime_voice_commands.VOICE_INTERACTION_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "src.realtime_voice_commands.REALTIME_VOICE_ENABLED",
        True,
    )
    result = handle_realtime_voice_command(
        "voice-realtime",
        RealtimeVoiceCommandContext(
            message="/voice-realtime",
            settings=_settings(),
            client=MagicMock(),
            conversation_history=ConversationHistory(),
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("test"),
        ),
    )
    assert result is not None
    assert result.message == REALTIME_VOICE_DISABLED


def test_voice_status_includes_realtime_lines(
    tmp_path: Any,
) -> None:
    result = handle_slash_command(
        "/voice-status",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_extractor=MagicMock(spec=TextExtractor),
        client=None,
    )
    assert "Realtime model: gpt-realtime-mini" in result.message
    assert "Realtime voice: coral" in result.message
    assert "Realtime available:" in result.message


def test_help_lists_voice_realtime(tmp_path: Any) -> None:
    result = handle_slash_command(
        "/help",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_extractor=MagicMock(spec=TextExtractor),
        client=None,
    )
    assert "/voice-realtime" in result.message
