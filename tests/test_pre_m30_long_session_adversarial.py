"""Pre-M30 extended stress tests: long-session bounds and resource growth."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from src.active_memory import ActiveMemoryContext
from src.commands import handle_slash_command
from src.config import (
    MAX_ACTIVE_MEMORIES,
    MAX_CONVERSATIONAL_REFERENTS,
    MAX_RECENT_SPOKEN_FINGERPRINTS,
)
from src.conversation import ConversationHistory, DEFAULT_MAX_COMPLETED_TURNS as HISTORY_CAP
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_loop import process_conversation_turn
from src.conversation_state import ConversationState
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.realtime_voice import _TurnAssembler
from src.settings import Settings
from src.speech_delivery import SpeechDeliveryState, safe_prepare_spoken_delivery


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_model="test-model")


def _slash(message: str, tmp_path: Path, **kwargs: Any) -> Any:
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=kwargs.get("history") or ConversationHistory(),
        memory_store=kwargs.get("memory") or JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=kwargs.get("active") or ActiveMemoryContext(),
        document_vault=kwargs.get("vault") or JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        conversation_state=kwargs.get("state"),
        speech_delivery_state=kwargs.get("delivery"),
    )


def test_five_hundred_mixed_local_turns_stay_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history = ConversationHistory()
    state = ConversationState()
    intel = ConversationIntelligence()
    delivery = SpeechDeliveryState()
    active = ActiveMemoryContext()
    memory = JsonMemoryStore(tmp_path / "memories.json")
    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **kwargs: "Acknowledged.",
    )
    before_temps = {p.name for p in Path(tempfile.gettempdir()).glob("cortana-*") if p.is_dir()}
    for index in range(500):
        kind = index % 10
        if kind == 0:
            process_conversation_turn(
                client=cast(Any, object()),
                settings=_settings(),
                user_message=f"Question {index} about MFA?",
                logger=logging.getLogger("hunt3"),
                conversation_history=history,
                conversation_state=state,
                conversation_intelligence=intel,
                active_memory_context=active,
            )
        elif kind == 1:
            intel.interpret("the other one", state)
        elif kind == 2:
            intel.interpret("actually, Tuesday", state)
        elif kind == 3:
            intel.interpret("go back", state)
        elif kind == 4:
            _slash("/memories", tmp_path, memory=memory, history=history, active=active)
        elif kind == 5:
            _slash("/documents", tmp_path, memory=memory, history=history)
        elif kind == 6:
            safe_prepare_spoken_delivery(f"Spoken answer {index}.", None, delivery)
        elif kind == 7:
            assembler = _TurnAssembler(history)
            assembler.set_current_user_item(f"item_{index}")
            assembler.bind_response(f"resp_{index}")
            assembler.store_user_transcript(f"item_{index}", f"user {index}")
            assembler.store_assistant_transcript(f"resp_{index}", f"assistant {index}")
            assembler.on_response_done(response_id=f"resp_{index}", status="completed")
        else:
            intel.interpret(f"switch topic to item {index}", state)
    after_temps = {p.name for p in Path(tempfile.gettempdir()).glob("cortana-*") if p.is_dir()}
    assert history.completed_turn_count <= HISTORY_CAP
    assert len(history.turns) <= HISTORY_CAP * 2
    assert len(state.recent_referents) <= MAX_CONVERSATIONAL_REFERENTS
    assert len(delivery.recently_spoken_fingerprints) <= MAX_RECENT_SPOKEN_FINGERPRINTS
    assert active.active_count <= MAX_ACTIVE_MEMORIES
    assert after_temps - before_temps == set()


def test_one_hundred_clear_cycles_reset_session_only(tmp_path: Path) -> None:
    history = ConversationHistory()
    state = ConversationState()
    delivery = SpeechDeliveryState()
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("persists across clear")
    for index in range(100):
        history.add_user_message(f"u{index}")
        history.add_assistant_message(f"a{index}")
        state.set_topic(f"topic {index}")
        delivery.last_spoken_opening = f"opening {index}"
        result = _slash(
            "/clear",
            tmp_path,
            history=history,
            memory=memory,
            state=state,
            delivery=delivery,
        )
        assert result.message is not None
        assert history.turns == []
    assert [item.text for item in memory.list_memories()] == ["persists across clear"]


def test_one_hundred_store_reconstructions_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    JsonMemoryStore(path).add_memory("anchor")
    for _index in range(100):
        store = JsonMemoryStore(path)
        assert len(store.list_memories()) == 1
        assert store.list_memories()[0].text == "anchor"


def test_one_hundred_speech_and_assembler_cycles_do_not_grow() -> None:
    delivery = SpeechDeliveryState()
    history = ConversationHistory()
    for index in range(100):
        safe_prepare_spoken_delivery(f"Cycle {index} spoken text.", None, delivery)
        assembler = _TurnAssembler(ConversationHistory())
        assembler.set_current_user_item(f"u{index}")
        assembler.bind_response(f"r{index}")
        assembler.store_user_transcript(f"u{index}", f"hello {index}")
        assembler.store_assistant_transcript(f"r{index}", f"reply {index}")
        assembler.on_response_done(response_id=f"r{index}", status="completed")
        history.add_user_message(f"u{index}")
        history.add_assistant_message(f"a{index}")
    assert len(delivery.recently_spoken_fingerprints) <= MAX_RECENT_SPOKEN_FINGERPRINTS
    assert history.completed_turn_count == HISTORY_CAP
    assert delivery.pending_chunks == [] or len(delivery.pending_chunks) <= 8


def test_one_hundred_help_commands_do_not_leak_temp_dirs(tmp_path: Path) -> None:
    before = {p.name for p in Path(tempfile.gettempdir()).glob("cortana-*") if p.is_dir()}
    for _index in range(100):
        result = _slash("/help", tmp_path)
        assert result.message is not None
        assert result.message.startswith("Cortana:")
    after = {p.name for p in Path(tempfile.gettempdir()).glob("cortana-*") if p.is_dir()}
    assert after - before == set()


def test_one_hundred_realtime_session_start_stop_cycles() -> None:
    from tests.test_realtime_voice import FakeConnection, _run_session
    from src.realtime_voice import REALTIME_STOPPED_MESSAGE

    for _index in range(100):
        connection = FakeConnection()
        history = ConversationHistory()
        thread, session, result_box, _printed = _run_session(connection, history)
        session.request_stop(error_type="cancelled")
        thread.join(timeout=5)
        assert thread.is_alive() is False
        assert result_box["message"] == REALTIME_STOPPED_MESSAGE
        assert connection.closed is True


def test_one_hundred_multimodal_session_start_stop_cycles() -> None:
    from tests.test_realtime_multimodal import FakeConnection, _run_session
    from src.realtime_multimodal import MULTIMODAL_STOPPED_MESSAGE

    for _index in range(100):
        connection = FakeConnection()
        history = ConversationHistory()
        thread, session, result_box, _printed = _run_session(connection, history)
        session.request_stop(error_type="cancelled")
        thread.join(timeout=5)
        assert thread.is_alive() is False
        assert result_box["message"] == MULTIMODAL_STOPPED_MESSAGE
        assert connection.closed is True
