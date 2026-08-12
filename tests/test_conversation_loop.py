"""Tests for Project Cortana conversation loop."""

import logging
from pathlib import Path
from typing import Any, cast

import pytest

import main as main_module
from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.conversation import ConversationHistory, SHUTDOWN_MESSAGE, STARTUP_GREETING
from src.conversation_loop import handle_message, run_conversation_loop
from src.document_chunker import DocumentChunker
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import DocumentVault, JsonDocumentVault
from src.evidence_store import EvidenceStore, LocalEvidenceStore
from src.incident_analysis_audit import InMemoryIncidentAnalysisAuditLog
from src.incident_analysis_repository import InMemoryIncidentAnalysisRepository
from src.incident_repository import IncidentRepository, JsonIncidentRepository
from src.memory_store import JsonMemoryStore, MemoryStore
from src.calendar_service import CalendarService
from src.reminder_service import ReminderService
from src.retrieval_session import RetrievalSession
from src.settings import Settings
from src.study_service import StudyPartnerService
from src.vision_input import VisualInputLoader
from src.vision_service import VisualAnalysisService
from src.voice_input import MicrophoneCaptureAdapter
from src.voice_service import VoiceService
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry
from src.tool_repository import JsonToolControlRepository, ToolControlRepository
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import (
    JsonWorkflowRunRepository,
    WorkflowRunRepository,
)

FAKE_CLIENT = cast(OpenAIClient, object())



def _document_vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _document_extractor() -> DefaultTextExtractor:
    return DefaultTextExtractor()


def _memory_store(tmp_path: Path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "memories.json")


def _active_memory_context() -> ActiveMemoryContext:
    return ActiveMemoryContext()


class FakeLogger(logging.Logger):
    """Logger substitute used during conversation loop tests."""

    def __init__(self) -> None:
        super().__init__("ProjectCortanaTest")
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(
        self,
        msg: object,
        *args: object,
        **kwargs: Any,
    ) -> None:
        """Record informational log messages."""
        message = str(msg)
        self.info_messages.append(message % args if args else message)

    def error(
        self,
        msg: object,
        *args: object,
        **kwargs: Any,
    ) -> None:
        """Record error log messages."""
        message = str(msg)
        self.error_messages.append(message % args if args else message)


def test_handle_message_prints_ai_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful AI response should be displayed to the user."""
    logger = FakeLogger()
    fake_client = FAKE_CLIENT
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    captured_message: str | None = None

    def fake_generate_response(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        conversation_history: ConversationHistory | None = None,
        active_memories: object = None,
        memory_boundary_token: object = None,
        **_kwargs: object,
    ) -> str:
        nonlocal captured_message
        captured_message = user_message

        assert client is FAKE_CLIENT
        assert settings is fake_settings

        return "Analysis complete."

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    handle_message(
        client=fake_client,
        settings=fake_settings,
        user_message="Analyze this log",
        logger=logger,
    )

    output = capsys.readouterr().out

    assert captured_message == "Analyze this log"
    assert "Cortana: Analysis complete." in output
    assert logger.info_messages == ["Response completed."]


def test_handle_message_updates_conversation_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful responses should be stored in session history."""
    logger = FakeLogger()
    history = ConversationHistory()

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **kwargs: "Here is the answer.",
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        user_message="What is MFA?",
        logger=logger,
        conversation_history=history,
    )

    capsys.readouterr()
    turns = history.turns

    assert len(turns) == 2
    assert turns[0].content == "What is MFA?"
    assert turns[1].content == "Here is the answer."


def test_process_conversation_turn_returns_answer_without_printing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Voice/chat reuse path should return text and not print it."""
    from src.conversation_loop import process_conversation_turn

    logger = FakeLogger()
    history = ConversationHistory()
    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        lambda **kwargs: "Silent answer.",
    )

    answer = process_conversation_turn(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        user_message="Hello",
        logger=logger,
        conversation_history=history,
    )

    output = capsys.readouterr().out
    assert answer == "Silent answer."
    assert output == ""
    assert len(history.turns) == 2


def test_handle_message_does_not_update_history_on_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failed AI requests should not alter conversation history."""
    logger = FakeLogger()
    history = ConversationHistory()

    def fake_generate_response(**kwargs: object) -> str:
        raise RuntimeError("Temporary outage")

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        user_message="Hello",
        logger=logger,
        conversation_history=history,
    )

    capsys.readouterr()

    assert history.turns == []


def test_handle_message_logs_only_safe_error_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AI failures should not expose the exception message."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )

    def fake_generate_response(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        conversation_history: ConversationHistory | None = None,
        active_memories: object = None,
        memory_boundary_token: object = None,
        **_kwargs: object,
    ) -> str:
        raise RuntimeError("Sensitive simulated error details")

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    handle_message(
        client=FAKE_CLIENT,
        settings=fake_settings,
        user_message="Hello",
        logger=logger,
    )

    output = capsys.readouterr().out

    assert "Cortana: I could not complete that request." in output
    assert "Sensitive simulated error details" not in output
    assert logger.error_messages == [
        "The OpenAI request failed with error type: RuntimeError"
    ]


def test_run_conversation_loop_displays_startup_greeting(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The loop should greet the user before accepting input."""
    logger = FakeLogger()
    inputs = iter(["exit"])

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert STARTUP_GREETING in output


def test_run_conversation_loop_exits_on_exit_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Exit commands should end the session with a shutdown message."""
    logger = FakeLogger()
    inputs = iter(["  QUIT  "])

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_run_conversation_loop_rejects_blank_input_without_ai_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Blank input should be rejected and the loop should continue."""
    logger = FakeLogger()
    ai_calls = 0
    inputs = iter(["", "exit"])

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal ai_calls
        ai_calls += 1

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert "Cortana: Please enter a message." in output
    assert ai_calls == 0


def test_run_conversation_loop_continues_after_ai_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Temporary AI failures should not terminate the session."""
    logger = FakeLogger()
    inputs = iter(["Hello", "exit"])
    call_count = 0

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            print("Cortana: I could not complete that request.")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert call_count == 1
    assert "Cortana: I could not complete that request." in output
    assert SHUTDOWN_MESSAGE in output


def test_run_conversation_loop_processes_multiple_messages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The loop should handle several messages before exiting."""
    logger = FakeLogger()
    inputs = iter(["First question", "Second question", "goodbye"])
    handled_messages: list[str] = []

    def fake_handle_message(
        *,
        client: OpenAIClient,
        settings: Settings,
        user_message: str,
        logger: logging.Logger,
        conversation_history: ConversationHistory | None = None,
        active_memory_context: ActiveMemoryContext | None = None,
        **_kwargs: object,
    ) -> None:
        handled_messages.append(user_message)
        print(f"Cortana: Response to {user_message}")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert handled_messages == ["First question", "Second question"]
    assert "Response to First question" in output
    assert "Response to Second question" in output
    assert SHUTDOWN_MESSAGE in output


def test_run_conversation_loop_rejects_whitespace_only_input_without_ai_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Whitespace-only input should be rejected and the loop should continue."""
    logger = FakeLogger()
    ai_calls = 0
    inputs = iter(["   ", "exit"])

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal ai_calls
        ai_calls += 1

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out

    assert "Cortana: Please enter a message." in output
    assert ai_calls == 0


def test_run_conversation_loop_exits_cleanly_on_eof(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """EOF during input should print the shutdown message without a traceback."""
    logger = FakeLogger()

    def raise_eof() -> str:
        raise EOFError

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=raise_eof,
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_run_conversation_loop_exits_cleanly_on_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """KeyboardInterrupt during input should exit cleanly without a traceback."""
    logger = FakeLogger()

    def raise_keyboard_interrupt() -> str:
        raise KeyboardInterrupt

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=Settings(
            openai_api_key="test-api-key",
            openai_model="test-model",
        ),
        logger=logger,
        active_memory_context=_active_memory_context(),
        document_vault=_document_vault(tmp_path),
        document_extractor=_document_extractor(),
        memory_store=_memory_store(tmp_path),
        read_input=raise_keyboard_interrupt,
    )

    output = capsys.readouterr().out

    assert SHUTDOWN_MESSAGE in output
    assert logger.info_messages == ["Conversation session ended by user."]


def test_main_orchestrates_conversation_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Main should initialize Cortana, inject storage dependencies, and start the loop."""
    logger = FakeLogger()
    fake_settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )
    fake_client = cast(OpenAIClient, object())
    memory_path = tmp_path / "memories.json"
    vault_path = tmp_path / "documents.json"
    incident_path = tmp_path / "incidents.json"
    evidence_dir = tmp_path / "evidence"
    tool_path = tmp_path / "tool_control.json"
    workflow_path = tmp_path / "workflow_runs.json"
    reminder_path = tmp_path / "reminders.json"
    calendar_path = tmp_path / "calendar_control.json"
    study_path = tmp_path / "study_state.json"
    received_client: OpenAIClient | None = None
    received_settings: Settings | None = None
    received_logger: logging.Logger | None = None
    received_memory_store: MemoryStore | None = None
    received_active_memory_context: ActiveMemoryContext | None = None
    received_document_vault: DocumentVault | None = None
    received_document_extractor: DefaultTextExtractor | None = None
    received_document_chunker: DocumentChunker | None = None
    received_document_retriever: LexicalDocumentRetriever | None = None
    received_retrieval_session: RetrievalSession | None = None
    received_incident_repository: IncidentRepository | None = None
    received_evidence_store: EvidenceStore | None = None
    received_tool_registry: ToolRegistry | None = None
    received_tool_repository: ToolControlRepository | None = None
    received_tool_executor: DefensiveToolExecutor | None = None
    received_workflow_registry: WorkflowRegistry | None = None
    received_workflow_run_repository: WorkflowRunRepository | None = None
    received_workflow_executor: WorkflowExecutor | None = None
    received_analysis_repository: InMemoryIncidentAnalysisRepository | None = None
    received_analysis_audit_log: InMemoryIncidentAnalysisAuditLog | None = None
    received_reminder_service: ReminderService | None = None
    received_calendar_service: CalendarService | None = None
    received_study_service: StudyPartnerService | None = None
    received_vision_loader: VisualInputLoader | None = None
    received_vision_service: VisualAnalysisService | None = None
    received_voice_capture: MicrophoneCaptureAdapter | None = None
    received_voice_service: VoiceService | None = None

    monkeypatch.setattr(main_module, "setup_logging", lambda: logger)
    monkeypatch.setattr(
        main_module,
        "initialize_ai",
        lambda supplied_logger: (fake_settings, fake_client),
    )
    monkeypatch.setattr(
        main_module,
        "get_default_memory_file_path",
        lambda: memory_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_document_vault_file_path",
        lambda: vault_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_incident_repository_file_path",
        lambda: incident_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_evidence_store_dir_path",
        lambda: evidence_dir,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_tool_control_repository_file_path",
        lambda: tool_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_workflow_repository_file_path",
        lambda: workflow_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_reminder_repository_file_path",
        lambda: reminder_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_calendar_repository_file_path",
        lambda: calendar_path,
    )
    monkeypatch.setattr(
        main_module,
        "get_default_study_repository_file_path",
        lambda: study_path,
    )

    def fake_run_conversation_loop(
        *,
        client: OpenAIClient,
        settings: Settings,
        logger: logging.Logger,
        memory_store: MemoryStore,
        active_memory_context: ActiveMemoryContext,
        document_vault: DocumentVault,
        document_extractor: DefaultTextExtractor,
        document_chunker: DocumentChunker,
        document_retriever: LexicalDocumentRetriever,
        retrieval_session: RetrievalSession,
        incident_repository: IncidentRepository,
        evidence_store: EvidenceStore,
        tool_registry: ToolRegistry,
        tool_repository: ToolControlRepository,
        tool_executor: DefensiveToolExecutor,
        workflow_registry: WorkflowRegistry,
        workflow_run_repository: WorkflowRunRepository,
        workflow_executor: WorkflowExecutor,
        analysis_repository: InMemoryIncidentAnalysisRepository,
        analysis_audit_log: InMemoryIncidentAnalysisAuditLog,
        reminder_service: ReminderService,
        calendar_service: CalendarService,
        study_service: StudyPartnerService | None = None,
        vision_loader: VisualInputLoader | None = None,
        vision_service: VisualAnalysisService | None = None,
        voice_capture: MicrophoneCaptureAdapter | None = None,
        voice_service: VoiceService | None = None,
    ) -> None:
        nonlocal received_client, received_settings, received_logger, received_memory_store
        nonlocal received_active_memory_context
        nonlocal received_document_vault, received_document_extractor
        nonlocal received_document_chunker, received_document_retriever
        nonlocal received_retrieval_session
        nonlocal received_incident_repository, received_evidence_store
        nonlocal received_tool_registry, received_tool_repository, received_tool_executor
        nonlocal received_workflow_registry, received_workflow_run_repository
        nonlocal received_workflow_executor
        nonlocal received_analysis_repository, received_analysis_audit_log
        nonlocal received_reminder_service, received_calendar_service
        nonlocal received_study_service
        nonlocal received_vision_loader, received_vision_service
        nonlocal received_voice_capture, received_voice_service
        received_client = client
        received_settings = settings
        received_logger = logger
        received_memory_store = memory_store
        received_active_memory_context = active_memory_context
        received_document_vault = document_vault
        received_document_extractor = document_extractor
        received_document_chunker = document_chunker
        received_document_retriever = document_retriever
        received_retrieval_session = retrieval_session
        received_incident_repository = incident_repository
        received_evidence_store = evidence_store
        received_tool_registry = tool_registry
        received_tool_repository = tool_repository
        received_tool_executor = tool_executor
        received_workflow_registry = workflow_registry
        received_workflow_run_repository = workflow_run_repository
        received_workflow_executor = workflow_executor
        received_analysis_repository = analysis_repository
        received_analysis_audit_log = analysis_audit_log
        received_reminder_service = reminder_service
        received_calendar_service = calendar_service
        received_study_service = study_service
        received_vision_loader = vision_loader
        received_vision_service = vision_service
        received_voice_capture = voice_capture
        received_voice_service = voice_service

    monkeypatch.setattr(
        main_module,
        "run_conversation_loop",
        fake_run_conversation_loop,
    )

    main_module.main()

    assert received_client is fake_client
    assert received_settings is fake_settings
    assert received_logger is logger
    assert isinstance(received_memory_store, JsonMemoryStore)
    assert received_memory_store.file_path == memory_path
    assert isinstance(received_active_memory_context, ActiveMemoryContext)
    assert received_active_memory_context.active_count == 0
    assert isinstance(received_document_vault, JsonDocumentVault)
    assert received_document_vault.file_path == vault_path
    assert isinstance(received_document_extractor, DefaultTextExtractor)
    assert isinstance(received_document_chunker, DocumentChunker)
    assert isinstance(received_document_retriever, LexicalDocumentRetriever)
    assert received_document_retriever.chunker is received_document_chunker
    assert isinstance(received_retrieval_session, RetrievalSession)
    assert received_retrieval_session.has_source_manifest is False
    assert isinstance(received_incident_repository, JsonIncidentRepository)
    assert received_incident_repository.file_path == incident_path
    assert isinstance(received_evidence_store, LocalEvidenceStore)
    assert received_evidence_store.directory_path == evidence_dir
    assert isinstance(received_tool_registry, ToolRegistry)
    assert isinstance(received_tool_repository, JsonToolControlRepository)
    assert received_tool_repository.file_path == tool_path
    assert isinstance(received_tool_executor, DefensiveToolExecutor)
    assert isinstance(received_workflow_registry, WorkflowRegistry)
    assert isinstance(received_workflow_run_repository, JsonWorkflowRunRepository)
    assert received_workflow_run_repository.file_path == workflow_path
    assert isinstance(received_workflow_executor, WorkflowExecutor)
    assert isinstance(received_analysis_repository, InMemoryIncidentAnalysisRepository)
    assert received_analysis_repository.analysis_count() == 0
    assert isinstance(received_analysis_audit_log, InMemoryIncidentAnalysisAuditLog)
    assert isinstance(received_reminder_service, ReminderService)
    assert received_reminder_service.count_all() == 0
    assert isinstance(received_calendar_service, CalendarService)
    assert received_calendar_service.status_view().connection_state == "not_connected"
    assert isinstance(received_study_service, StudyPartnerService)
    assert isinstance(received_vision_loader, VisualInputLoader)
    assert isinstance(received_vision_service, VisualAnalysisService)
    assert isinstance(received_voice_capture, MicrophoneCaptureAdapter)
    assert isinstance(received_voice_service, VoiceService)
    assert reminder_path.parent.exists()
    assert calendar_path.parent.exists()
    assert study_path.parent.exists()
    assert received_workflow_registry.count() >= 2
