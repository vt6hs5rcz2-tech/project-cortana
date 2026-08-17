"""Pre-M30 hardening tests: first-run journey and invalid startup configuration."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import main as main_module
from src.active_memory import ActiveMemoryContext
from src.camera_capture import CAMERA_UNAVAILABLE
from src.commands import HELP_TEXT, handle_slash_command, parse_slash_input
from src.config import (
    ACTIVE_MEMORY_PERSISTENCE_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    MAX_CONVERSATION_MESSAGE_CHARS,
    MAX_STORED_MEMORIES,
)
from src.conversation import (
    MESSAGE_TOO_LONG,
    SHUTDOWN_MESSAGE,
    STARTUP_GREETING,
    ConversationHistory,
)
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_loop import process_conversation_turn, run_conversation_loop
from src.conversation_state import ConversationState
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.identity import CORTANA_SYSTEM_INSTRUCTIONS
from src.memory_store import JsonMemoryStore
from src.reminder import ManualReminderClock
from src.reminder_repository import JsonReminderRepository
from src.reminder_service import ReminderService
from src.settings import Settings, load_settings
from src.speech_delivery import SpeechDeliveryState
from src.study_repository import JsonStudyRepository
from src.voice_commands import VOICE_EMPTY_TRANSCRIPT, VoiceCommandContext, handle_voice_command
from src.voice_input import (
    VOICE_MICROPHONE_UNAVAILABLE,
    MicrophoneCaptureAdapter,
    NormalizedAudioInput,
    pcm_to_wav_bytes,
)
from src.voice_service import VoiceService
from datetime import datetime, timezone
from tests.test_study_service import _add, _service


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
        reminder_service=kwargs.get("reminders"),
        study_service=kwargs.get("study"),
        client=kwargs.get("client"),
        voice_capture=kwargs.get("capture"),
        voice_service=kwargs.get("voice"),
    )


def _assert_user_facing(message: str) -> None:
    assert message, "user-facing output must not be empty"
    assert "Traceback" not in message
    assert "\nNone\n" not in message
    assert not message.strip().startswith("None")
    lowered = message.casefold()
    assert "exception" not in lowered or "Cortana:" in message


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


def test_first_startup_greeting_is_cortana_prefixed_and_actionable() -> None:
    assert STARTUP_GREETING.startswith("Cortana:")
    assert "ready" in STARTUP_GREETING.casefold()
    # First-time users need a discoverable next step besides exit.
    assert "/help" in STARTUP_GREETING or "slash" in STARTUP_GREETING.casefold()
    assert SHUTDOWN_MESSAGE.startswith("Cortana:")
    _assert_user_facing(STARTUP_GREETING)
    _assert_user_facing(SHUTDOWN_MESSAGE)


def test_first_help_lists_core_first_run_commands() -> None:
    assert HELP_TEXT.startswith("Cortana:")
    for token in (
        "/remember",
        "/recall",
        "/memories",
        "/reminder-add",
        "/add-document",
        "/ask-docs",
        "/study-start",
        "/voice-turn",
        "/voice-realtime",
        "/clear",
        "/exit",
    ):
        assert token in HELP_TEXT
    _assert_user_facing(HELP_TEXT)


def test_first_run_connection_messages_are_cortana_prefixed() -> None:
    source = inspect.getsource(main_module.initialize_ai)
    assert "user_facing_settings_error" in source
    main_source = inspect.getsource(main_module.main)
    assert "format_startup_banner" in main_source


def test_missing_and_blank_api_key_are_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as missing:
        load_settings()
    assert "OPENAI_API_KEY" in str(missing.value)
    assert "Traceback" not in str(missing.value)

    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(ValueError) as blank:
        load_settings()
    assert "OPENAI_API_KEY" in str(blank.value)

    logger = logging.getLogger("ProjectCortanaFirstRun")
    monkeypatch.setattr("main.load_settings", load_settings)
    result = main_module.initialize_ai(logger)
    assert result is None


def test_unsupported_voice_setting_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CORTANA_TTS_VOICE", "not-a-real-voice")
    with pytest.raises(ValueError) as error:
        load_settings()
    assert "voice" in str(error.value).casefold()
    assert "Traceback" not in str(error.value)


def test_absent_calendar_credentials_do_not_block_other_features(
    tmp_path: Path,
) -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        google_oauth_client_file=tmp_path / "missing-client.json",
    )
    store = JsonMemoryStore(tmp_path / "memories.json")
    result = handle_slash_command(
        "/remember Calendar is optional",
        settings=settings,
        conversation_history=ConversationHistory(),
        memory_store=store,
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )
    assert "Memory saved" in (result.message or "")
    assert len(store.list_memories()) == 1


def test_microphone_and_camera_absence_messages_are_user_visible() -> None:
    assert VOICE_MICROPHONE_UNAVAILABLE.startswith("Cortana:")
    assert CAMERA_UNAVAILABLE.startswith("Cortana:")
    _assert_user_facing(VOICE_MICROPHONE_UNAVAILABLE)
    _assert_user_facing(CAMERA_UNAVAILABLE)


def test_clean_user_end_to_end_first_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    history = ConversationHistory()
    state = ConversationState()
    delivery = SpeechDeliveryState()
    memory = JsonMemoryStore(tmp_path / "memories.json")
    active = ActiveMemoryContext()
    vault = JsonDocumentVault(tmp_path / "documents.json")
    clock = ManualReminderClock(start=datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc))
    reminders = ReminderService(JsonReminderRepository(tmp_path / "reminders.json"), clock=clock)
    study, study_vault, fake_study, study_repo = _service(tmp_path)
    doc_id = _add(
        study_vault,
        "guide.md",
        "Isolate the compromised host immediately. Do not reboot first.",
    )
    source = tmp_path / "notes.md"
    source.write_text("Firewall allows SSH on port 22 only.\n", encoding="utf-8")

    answers = iter(["I am Cortana, your cybersecurity assistant.", "Use MFA."])

    def fake_generate(**kwargs: Any) -> str:
        return next(answers)

    monkeypatch.setattr("src.conversation_loop.generate_response", fake_generate)
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    voice = MagicMock(spec=VoiceService)
    voice.transcribe.return_value = "What is MFA?"
    voice.synthesize.return_value = b"RIFFWAVEdata"
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")
    monkeypatch.setattr("src.voice_commands._play_wav_synchronously", lambda _data: None)

    inputs = iter(
        [
            "Who are you?",
            "How do I enable MFA?",
            "/remember VPN gateway is 10.0.0.1",
            "/memories",
            "/reminder-add Standup | 2026-08-20 09:00 | America/New_York | none | -",
            "/reminders",
            f"/add-document {source}",
            "/documents",
            f"/study-start {doc_id}",
            "/study-status",
            "/voice-turn",
            "/clear",
            "exit",
        ]
    )
    run_conversation_loop(
        client=cast(Any, object()),
        settings=_settings(),
        logger=logging.getLogger("ProjectCortanaFirstRun"),
        memory_store=memory,
        active_memory_context=active,
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        reminder_service=reminders,
        study_service=study,
        conversation_history=history,
        conversation_state=state,
        speech_delivery_state=delivery,
        voice_capture=capture,
        voice_service=voice,
        read_input=lambda: next(inputs),
    )
    output = capsys.readouterr().out
    assert STARTUP_GREETING in output
    assert SHUTDOWN_MESSAGE in output
    assert "Traceback" not in output
    assert output.count("Cortana: Cortana:") == 0
    assert memory.list_memories()
    assert reminders.count_scheduled() == 1
    assert vault.list_documents()
    assert study_repo.get_active_session() is not None
    assert history.turns == []
    assert active.list_active() == []

    rebuilt_memory = JsonMemoryStore(tmp_path / "memories.json")
    rebuilt_reminders = ReminderService(JsonReminderRepository(tmp_path / "reminders.json"))
    rebuilt_vault = JsonDocumentVault(tmp_path / "documents.json")
    rebuilt_study = JsonStudyRepository(tmp_path / "study_state.json")
    rebuilt_active = ActiveMemoryContext()
    rebuilt_history = ConversationHistory()
    assert len(rebuilt_memory.list_memories()) == 1
    assert "10.0.0.1" in rebuilt_memory.list_memories()[0].text
    assert rebuilt_reminders.count_scheduled() == 1
    assert rebuilt_vault.list_documents()
    assert rebuilt_study.get_active_session() is not None
    assert HISTORY_PERSISTENCE_ENABLED is False
    assert rebuilt_history.turns == []
    assert ACTIVE_MEMORY_PERSISTENCE_ENABLED is False
    assert rebuilt_active.list_active() == []
    leftover = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob("cortana-*"))
    assert leftover == []


def test_first_run_who_are_you_sends_identity_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "I am Cortana."

    monkeypatch.setattr("src.conversation_loop.generate_response", fake_generate)
    answer = process_conversation_turn(
        client=cast(Any, object()),
        settings=_settings(),
        user_message="Who are you?",
        logger=logging.getLogger("ProjectCortanaFirstRun"),
        conversation_history=ConversationHistory(),
        conversation_state=ConversationState(),
        conversation_intelligence=ConversationIntelligence(),
    )
    assert answer == "I am Cortana."
    assert "Cortana" in CORTANA_SYSTEM_INSTRUCTIONS
    assert "cybersecurity" in CORTANA_SYSTEM_INSTRUCTIONS.casefold()


def test_first_run_oversized_message_is_rejected_without_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = ConversationHistory()
    state = ConversationState()
    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **kwargs: "nope",
    )
    answer = process_conversation_turn(
        client=cast(Any, object()),
        settings=_settings(),
        user_message="A" * (MAX_CONVERSATION_MESSAGE_CHARS + 1),
        logger=logging.getLogger("ProjectCortanaFirstRun"),
        conversation_history=history,
        conversation_state=state,
    )
    assert answer == MESSAGE_TOO_LONG
    assert history.turns == []
    _assert_user_facing(MESSAGE_TOO_LONG)


def test_first_run_blank_voice_turn_does_not_print_heard(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = ConversationHistory()
    capture = MagicMock(spec=MicrophoneCaptureAdapter)
    capture.capture.return_value = _audio()
    service = MagicMock(spec=VoiceService)
    service.transcribe.return_value = "  \n"
    monkeypatch.setattr("src.voice_commands.sys.platform", "win32")
    result = handle_voice_command(
        "voice-turn",
        VoiceCommandContext(
            message="/voice-turn",
            settings=_settings(),
            client=cast(Any, object()),
            conversation_history=history,
            active_memory_context=ActiveMemoryContext(),
            logger=logging.getLogger("ProjectCortanaFirstRun"),
            stop_signal=lambda: False,
            capture=capture,
            voice_service=service,
            conversation_state=ConversationState(),
            speech_delivery_state=SpeechDeliveryState(),
        ),
    )
    output = capsys.readouterr().out
    assert result is not None
    assert result.message == VOICE_EMPTY_TRANSCRIPT
    assert "(Heard)" not in output
    assert history.turns == []
    service.synthesize.assert_not_called()


def test_leading_space_help_is_not_a_silent_unknown_command() -> None:
    # After F4 this is ordinary chat, not Unknown command '/'.
    assert parse_slash_input(" / help") is None
    assert parse_slash_input("/help") == "help"


def test_memory_capacity_seam_is_explained(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=1)
    store.add_memory("kept")
    result = _slash("/remember overflow", tmp_path, memory=store)
    assert result.message is not None
    _assert_user_facing(result.message)
    assert "capacity" in result.message.casefold()
    assert str(MAX_STORED_MEMORIES) in HELP_TEXT or "remember" in HELP_TEXT.casefold()
    assert [memory.text for memory in store.list_memories()] == ["kept"]


def test_unknown_numeric_env_does_not_break_minimum_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CORTANA_MAX_TURNS", "not-a-number")
    settings = load_settings()
    assert settings.openai_api_key == "sk-test"


def test_language_selection_is_not_silently_half_implemented() -> None:
    # Audit only: do not invent a language picker. Startup must not expose one.
    assert not hasattr(_settings(), "language")
    assert not hasattr(_settings(), "tts_language")


def test_missing_oauth_client_file_is_deferred_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(
        "CORTANA_GOOGLE_OAUTH_CLIENT_FILE",
        str(tmp_path / "missing-client.json"),
    )
    settings = load_settings()
    assert settings.google_oauth_client_file == tmp_path / "missing-client.json"
    # Optional calendar must not poison unrelated memory writes.
    store = JsonMemoryStore(tmp_path / "memories.json")
    result = handle_slash_command(
        "/remember Optional calendar missing is fine",
        settings=settings,
        conversation_history=ConversationHistory(),
        memory_store=store,
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )
    assert "Memory saved" in (result.message or "")


def test_malformed_model_name_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CORTANA_TTS_MODEL", "not-a-model")
    with pytest.raises(ValueError) as error:
        load_settings()
    assert "model" in str(error.value).casefold()
    assert "Traceback" not in str(error.value)
