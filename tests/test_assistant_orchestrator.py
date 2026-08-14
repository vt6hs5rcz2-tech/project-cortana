"""Tests for Milestone 18 unified assistant orchestration."""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.assistant_orchestrator import (
    OrchestrationResult,
    UnifiedAssistantOrchestrator,
)
from src.commands import CommandOutcome, CommandResult, handle_slash_command
from src.config import MAX_MEMORY_TEXT_LENGTH, PROJECT_ROOT
from src.conversation import ConversationHistory
from src.conversation_loop import run_conversation_loop
from src.document import create_document
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore, MemoryCountLimitError
from src.security_incident import create_security_incident
from src.settings import Settings

FAKE_CLIENT = cast(OpenAIClient, object())

ORCHESTRATOR_SOURCE = (
    PROJECT_ROOT / "src" / "assistant_orchestrator.py"
).read_text(encoding="utf-8")

FORBIDDEN_ORCHESTRATOR_TOKENS = (
    "DefensiveToolExecutor",
    "WorkflowExecutor",
    "IncidentAnalysisService",
    "ReminderService",
    "ReminderRepository",
    "JsonReminderRepository",
    "CalendarService",
    "CalendarRepository",
    "JsonCalendarRepository",
    "ai_service",
    "openai_client",
    "OpenAIClient",
    "generate_response",
    "tool_process_runner",
    "safe_open_for_read",
    "stream_text_search_from_safe_handle",
    "handle_slash_command",
    "parse_slash_input",
)


class FakeLogger(logging.Logger):
    """Logger substitute used during orchestration tests."""

    def __init__(self) -> None:
        super().__init__("ProjectCortanaOrchestratorTest")
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(
        self,
        msg: object,
        *args: object,
        **kwargs: Any,
    ) -> None:
        message = str(msg)
        self.info_messages.append(message % args if args else message)

    def error(
        self,
        msg: object,
        *args: object,
        **kwargs: Any,
    ) -> None:
        message = str(msg)
        self.error_messages.append(message % args if args else message)


def _memory_store(tmp_path: Path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "memories.json")


def _document_vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _incident_repository(tmp_path: Path) -> JsonIncidentRepository:
    return JsonIncidentRepository(tmp_path / "incidents.json")


def _orchestrator(
    tmp_path: Path,
    *,
    incident_repository: JsonIncidentRepository | None = None,
) -> UnifiedAssistantOrchestrator:
    return UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=(
            incident_repository
            if incident_repository is not None
            else _incident_repository(tmp_path)
        ),
    )


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _add_document(vault: JsonDocumentVault, text: str, filename: str = "note.txt") -> None:
    vault.add_document(
        create_document(
            filename=filename,
            extension=".txt",
            source_size_bytes=len(text.encode("utf-8")),
            content_hash="a" * 64,
            extracted_text=text,
        )
    )


# --- A. Slash precedence -------------------------------------------------


def test_slash_command_bypasses_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Slash input must never be offered to the orchestrator."""
    orchestrator_calls: list[str] = []
    slash_calls: list[str] = []

    original_try_handle = UnifiedAssistantOrchestrator.try_handle

    def tracking_try_handle(
        self: UnifiedAssistantOrchestrator,
        user_message: str,
    ) -> OrchestrationResult | None:
        orchestrator_calls.append(user_message)
        return original_try_handle(self, user_message)

    def fake_handle_slash_command(message: str, **kwargs: object) -> CommandResult:
        slash_calls.append(message)
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message="Cortana: Slash handled.",
        )

    monkeypatch.setattr(
        UnifiedAssistantOrchestrator,
        "try_handle",
        tracking_try_handle,
    )
    monkeypatch.setattr(
        "src.conversation_loop.handle_slash_command",
        fake_handle_slash_command,
    )
    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        lambda **kwargs: None,
    )

    inputs = iter(["/help", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert slash_calls == ["/help"]
    assert orchestrator_calls == []
    assert "Cortana: Slash handled." in output


def test_slash_remember_still_uses_existing_slash_dispatch(
    tmp_path: Path,
) -> None:
    """Existing /remember slash behavior remains authoritative and unchanged."""
    store = _memory_store(tmp_path)
    history = ConversationHistory()
    result = handle_slash_command(
        "/remember slash memory payload",
        settings=_settings(),
        conversation_history=history,
        memory_store=store,
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
    )

    memories = store.list_memories()
    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "Memory saved" in result.message
    assert len(memories) == 1
    assert memories[0].text == "slash memory payload"
    assert history.turns == []


# --- B. Conversation fallback --------------------------------------------


def test_unmatched_message_falls_through_to_handle_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Unmatched ordinary language continues through handle_message."""
    handled: list[str] = []
    history = ConversationHistory()

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
        handled.append(user_message)
        assert conversation_history is history
        if conversation_history is not None:
            conversation_history.add_user_message(user_message)
            conversation_history.add_assistant_message("Ordinary reply")
        print("Cortana: Ordinary reply")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    inputs = iter(["Tell me about phishing trends", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        conversation_history=history,
        read_input=lambda: next(inputs),
    )

    capsys.readouterr()
    assert handled == ["Tell me about phishing trends"]
    assert len(history.turns) == 2
    assert history.turns[0].content == "Tell me about phishing trends"
    assert history.turns[1].content == "Ordinary reply"


def test_orchestrator_returns_none_for_unmatched_phrases(tmp_path: Path) -> None:
    """Ambiguous phrases must produce no orchestration match."""
    orchestrator = _orchestrator(tmp_path)
    for phrase in (
        "hello there",
        "please help me investigate",
        "I need to remember when this started",
        "I found a document about phishing last week",
        "show incident report to the team",
        "Can you remember what I said?",
        "We should search documents later",
    ):
        assert orchestrator.try_handle(phrase) is None


# --- C. Memory write -----------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_text",
    [
        ("remember keep the case notes offline", "keep the case notes offline"),
        ("remember: keep the case notes offline", "keep the case notes offline"),
        ("remember - keep the case notes offline", "keep the case notes offline"),
        ("remember:  spaced payload", "spaced payload"),
        ("remember: my preferred editor is Cursor", "my preferred editor is Cursor"),
        ("remember my preferred timezone is Eastern", "my preferred timezone is Eastern"),
        (
            "remember that my project is called Cortana",
            "that my project is called Cortana",
        ),
        ("remember my dog's name is Max", "my dog's name is Max"),
        ("remember the server is in Virginia", "the server is in Virginia"),
        ("remember the backup window is 2 AM", "the backup window is 2 AM"),
    ],
)
def test_memory_write_anchored_forms(
    tmp_path: Path,
    message: str,
    expected_text: str,
) -> None:
    """Explicit anchored remember forms write through MemoryStore."""
    store = _memory_store(tmp_path)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle(message)

    assert result is not None
    assert result.domain == "memory"
    assert result.action == "write"
    assert "Memory saved" in result.safe_user_message
    memories = store.list_memories()
    assert len(memories) == 1
    assert memories[0].text == expected_text


def test_memory_write_blank_payload_falls_through(tmp_path: Path) -> None:
    """Blank remember payloads are not confident operational matches."""
    orchestrator = _orchestrator(tmp_path)
    assert orchestrator.try_handle("remember") is None
    assert orchestrator.try_handle("remember:") is None
    assert orchestrator.try_handle("remember  ") is None
    assert orchestrator.try_handle("remember:   ") is None


def test_memory_write_collisions_do_not_trigger(tmp_path: Path) -> None:
    """Conversational remember phrases must not write memory."""
    store = _memory_store(tmp_path)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )
    for phrase in (
        "I remember when we started.",
        "Can you remember what I said?",
        "I need to remember when this happened.",
        "Do you remember this?",
        "I need to remember when this started",
        "remember this",
        "remember that",
        "remember it",
        "remember this forever",
        "remember that forever",
        "remember it forever",
        "remember this thing",
        "remember that fact",
        "remember it please",
        "remember this forever please",
        "remember this for me",
        "remember what I just said",
        "remember the above",
        "remember the previous thing",
        "remember what we discussed",
        "remember that one",
    ):
        assert orchestrator.try_handle(phrase) is None
    assert store.list_memories() == []


def test_memory_write_respects_store_length_validation(tmp_path: Path) -> None:
    """MemoryStore length validation remains authoritative."""
    store = _memory_store(tmp_path)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )
    result = orchestrator.try_handle("remember " + ("x" * (MAX_MEMORY_TEXT_LENGTH + 1)))
    assert result is not None
    assert "too long" in result.safe_user_message.lower()
    assert store.list_memories() == []


def test_memory_write_respects_store_count_cap(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=1)
    store.add_memory("kept")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )
    result = orchestrator.try_handle("remember another fact")
    assert result is not None
    assert result.domain == "memory"
    assert result.action == "write"
    assert result.safe_user_message == MemoryCountLimitError(max_memories=1).user_message
    assert [memory.text for memory in store.list_memories()] == ["kept"]


def test_memory_write_does_not_mutate_history_in_loop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Routed memory writes must not enter conversation history."""
    history = ConversationHistory()
    handle_message_calls = 0

    def fake_handle_message(**kwargs: object) -> None:
        nonlocal handle_message_calls
        handle_message_calls += 1

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    inputs = iter(["remember operator note alpha", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        conversation_history=history,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert "Memory saved" in output
    assert history.turns == []
    assert handle_message_calls == 0


# --- D. Memory read ------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["list my memories", "list memories", "show my memories"],
)
def test_memory_read_allowlisted_phrases(tmp_path: Path, phrase: str) -> None:
    """Exact allowlisted memory list phrases work read-only."""
    store = _memory_store(tmp_path)
    store.add_memory("alpha memory")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle(phrase)

    assert result is not None
    assert result.domain == "memory"
    assert result.action == "list"
    assert "alpha memory" in result.safe_user_message
    assert len(store.list_memories()) == 1


def test_memory_read_near_matches_do_not_trigger(tmp_path: Path) -> None:
    """Near-matches must fall through without listing."""
    store = _memory_store(tmp_path)
    store.add_memory("secret")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )
    for phrase in (
        "list all my memories",
        "please list memories",
        "show memories",
        "list my memory",
    ):
        assert orchestrator.try_handle(phrase) is None


def test_memory_read_does_not_mutate_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Memory list routing stays out of conversation history."""
    history = ConversationHistory()
    store = _memory_store(tmp_path)
    store.add_memory("listed only")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("handle_message")),
    )

    inputs = iter(["list my memories", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=store,
        conversation_history=history,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert "listed only" in output
    assert history.turns == []


# --- E. Documents --------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "find document about phishing indicators",
        "find documents about phishing indicators",
        "search document for phishing indicators",
        "search documents for phishing indicators",
        "search my documents for phishing indicators",
    ],
)
def test_document_search_anchored_forms(tmp_path: Path, message: str) -> None:
    """Anchored document search forms use lexical retrieval only."""
    vault = _document_vault(tmp_path)
    _add_document(vault, "Report on phishing indicators and traps.")
    retriever = LexicalDocumentRetriever()
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=vault,
        document_retriever=retriever,
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle(message)

    assert result is not None
    assert result.domain == "documents"
    assert result.action == "search"
    assert "Document search results" in result.safe_user_message
    assert "phishing" in result.safe_user_message.lower()


def test_document_search_collisions_do_not_trigger(tmp_path: Path) -> None:
    """Conversational document phrases must not search the vault."""
    vault = _document_vault(tmp_path)
    _add_document(vault, "phishing playbook contents")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=vault,
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )
    for phrase in (
            "I found a document about phishing yesterday.",
            "I found a document about phishing last week",
            "We should search documents later.",
            "We should search documents later",
            "I remember the document about the server.",
        ):
        assert orchestrator.try_handle(phrase) is None


def test_document_search_has_no_ai_client_dependency(tmp_path: Path) -> None:
    """Document route constructor/signature excludes AI client and settings."""
    signature = inspect.signature(UnifiedAssistantOrchestrator.__init__)
    assert "client" not in signature.parameters
    assert "settings" not in signature.parameters
    assert "openai" not in ORCHESTRATOR_SOURCE.lower()
    assert "generate_response" not in ORCHESTRATOR_SOURCE
    assert "ai_service" not in ORCHESTRATOR_SOURCE


def test_document_search_does_not_mutate_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Document NL search stays out of conversation history."""
    history = ConversationHistory()
    vault = _document_vault(tmp_path)
    _add_document(vault, "Credential phishing indicators noted.")

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("handle_message")),
    )

    inputs = iter(["search documents for phishing", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        conversation_history=history,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert "Document search results" in output
    assert history.turns == []


def test_document_search_output_is_bounded(tmp_path: Path) -> None:
    """Search output uses bounded preview conventions."""
    vault = _document_vault(tmp_path)
    long_text = "phishing " + ("word " * 200)
    _add_document(vault, long_text)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=vault,
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle("find documents about phishing")
    assert result is not None
    assert "..." in result.safe_user_message


# --- F. Incident read ----------------------------------------------------


def test_incident_read_valid_uuid(tmp_path: Path) -> None:
    """Exact show-incident UUID form reads one incident."""
    repo = _incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Phish case",
            summary="Operator summary text",
            severity="medium",
        )
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=repo,
    )

    result = orchestrator.try_handle(f"show incident {incident.incident_id}")

    assert result is not None
    assert result.domain == "security"
    assert result.action == "incident_read"
    assert incident.incident_id in result.safe_user_message
    assert "Phish case" in result.safe_user_message
    assert "Operator summary text" in result.safe_user_message


def test_incident_read_invalid_and_missing_fail_safely(tmp_path: Path) -> None:
    """Malformed and missing IDs fail closed without mutation."""
    repo = _incident_repository(tmp_path)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=repo,
    )

    # Matches the 36-char shape but is not a valid UUID.
    invalid_id = "0" * 36
    invalid = orchestrator.try_handle(f"show incident {invalid_id}")
    assert invalid is not None
    assert "No saved incident found" in invalid.safe_user_message

    # Wrong shape must not route at all.
    malformed = orchestrator.try_handle("show incident not-a-uuid")
    assert malformed is None

    missing_id = str(uuid4())
    missing = orchestrator.try_handle(f"show incident {missing_id}")
    assert missing is not None
    assert "No saved incident found" in missing.safe_user_message
    assert missing_id in missing.safe_user_message


def test_incident_read_collisions_do_not_trigger(tmp_path: Path) -> None:
    """Conversational incident phrases must not read the repository."""
    repo = _incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Hidden",
            summary="Should not appear",
            severity="low",
        )
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=repo,
    )
    for phrase in (
        "show incident report to the team",
        "Can you show incident details?",
        "I want to show incident handling in the demo.",
        f"please show incident {incident.incident_id}",
    ):
        assert orchestrator.try_handle(phrase) is None


def test_incident_read_does_not_mutate_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Incident NL read keeps request and contents out of history."""
    history = ConversationHistory()
    repo = _incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Isolated",
            summary="Must not enter history",
            severity="high",
        )
    )

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("handle_message")),
    )

    inputs = iter([f"show incident {incident.incident_id}", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        incident_repository=repo,
        conversation_history=history,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert "Isolated" in output
    assert history.turns == []
    for turn in history.turns:
        assert "Must not enter history" not in turn.content


# --- G. Guidance ---------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,needle",
    [
        ("search evidence", "/evidence-search"),
        ("search incident evidence", "/evidence-search"),
        ("run a tool", "/tool-"),
        ("scan this file", "/tool-"),
        ("run the playbook", "/playbook-"),
        ("execute the workflow", "/playbook-"),
        ("run ai analysis", "/incident-analysis-"),
        ("analyze this incident with ai", "/incident-analysis-"),
        ("set a reminder", "/reminder-add"),
        ("list reminders", "/reminders"),
        ("show my calendar", "/calendar-events"),
        ("schedule a meeting", "/calendar-create"),
    ],
)
def test_guidance_routes_are_static(
    tmp_path: Path,
    phrase: str,
    needle: str,
) -> None:
    """Guidance phrases return fixed static text only."""
    orchestrator = _orchestrator(tmp_path)
    result = orchestrator.try_handle(phrase)
    assert result is not None
    assert result.domain == "guidance"
    assert needle in result.safe_user_message
    assert result.safe_user_message.startswith("Cortana:")


def test_guidance_does_not_echo_arbitrary_input(tmp_path: Path) -> None:
    """Guidance messages must not interpolate raw user input."""
    orchestrator = _orchestrator(tmp_path)
    poison = "search evidence"
    result = orchestrator.try_handle(poison)
    assert result is not None
    assert "rm -rf" not in result.safe_user_message
    assert result.safe_user_message.startswith("Cortana:")


def test_guidance_has_no_operational_side_effects(tmp_path: Path) -> None:
    """Guidance routes must not mutate memory, vault, or incidents."""
    store = _memory_store(tmp_path)
    vault = _document_vault(tmp_path)
    repo = _incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Baseline",
            summary="Baseline summary text",
            severity="low",
        )
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=vault,
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=repo,
    )

    for phrase in (
        "search evidence",
        "run a tool",
        "run the playbook",
        "run ai analysis",
    ):
        result = orchestrator.try_handle(phrase)
        assert result is not None

    assert store.list_memories() == []
    assert vault.list_documents() == []
    loaded = repo.get_incident(incident.incident_id)
    assert loaded is not None
    assert loaded.title == "Baseline"


def test_guidance_does_not_mutate_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Guidance routing stays out of conversation history."""
    history = ConversationHistory()
    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("handle_message")),
    )

    inputs = iter(["run a tool", "exit"])
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        conversation_history=history,
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert "/tool-" in output
    assert history.turns == []


# --- H. Static architecture / security -----------------------------------


def test_orchestrator_source_forbids_operational_imports() -> None:
    """Orchestrator module must not reference forbidden operational surfaces."""
    for token in FORBIDDEN_ORCHESTRATOR_TOKENS:
        assert token not in ORCHESTRATOR_SOURCE, token


def test_orchestrator_ast_has_no_forbidden_imports() -> None:
    """AST import scan should exclude executors, AI, and process helpers."""
    tree = ast.parse(ORCHESTRATOR_SOURCE)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            for alias in node.names:
                imported.add(alias.name)
                imported.add(f"{module}.{alias.name}" if module else alias.name)

    forbidden_modules = {
        "src.ai_service",
        "src.tool_executor",
        "src.workflow_executor",
        "src.incident_analysis_service",
        "src.reminder_service",
        "src.reminder_repository",
        "src.calendar_service",
        "src.calendar_repository",
        "src.tool_process_runner",
        "src.tool_safe_files",
        "src.tool_process_text_search",
        "src.tool_process_safe_open",
        "openai",
    }
    assert imported.isdisjoint(forbidden_modules)


def test_orchestrator_init_dependency_surface() -> None:
    """Constructor accepts only memory, documents, and optional incidents."""
    parameters = set(inspect.signature(UnifiedAssistantOrchestrator.__init__).parameters)
    assert parameters == {
        "self",
        "memory_store",
        "document_vault",
        "document_retriever",
        "incident_repository",
    }


def test_orchestrator_does_not_synthesize_slash_commands(tmp_path: Path) -> None:
    """Handled routes must not fabricate slash strings for re-parse."""
    store = _memory_store(tmp_path)
    vault = _document_vault(tmp_path)
    _add_document(vault, "phishing indicators")
    repo = _incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for display",
            severity="medium",
        )
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=vault,
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=repo,
    )

    messages = [
        "remember keep this offline",
        "list memories",
        "search documents for phishing",
        f"show incident {incident.incident_id}",
        "search evidence",
        "run a tool",
        "run the playbook",
        "run ai analysis",
    ]
    for message in messages:
        result = orchestrator.try_handle(message)
        assert result is not None
        assert "/remember" not in result.safe_user_message
        assert not result.safe_user_message.lstrip().startswith("/")


# --- Case-insensitive NL matching ----------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "remember keep this offline",
        "Remember keep this offline",
        "REMEMBER keep this offline",
    ],
)
def test_memory_write_is_case_insensitive(tmp_path: Path, message: str) -> None:
    """Remember routing matches regardless of command capitalization."""
    store = _memory_store(tmp_path)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle(message)

    assert result is not None
    assert result.domain == "memory"
    assert result.action == "write"
    assert store.list_memories()[0].text == "keep this offline"


def test_memory_write_preserves_captured_text_capitalization(tmp_path: Path) -> None:
    """Only route matching is case-normalized; stored text keeps user casing."""
    store = _memory_store(tmp_path)
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle("Remember Buy Milk Tomorrow")

    assert result is not None
    assert store.list_memories()[0].text == "Buy Milk Tomorrow"


@pytest.mark.parametrize(
    "message",
    [
        "list memories",
        "List memories",
        "LIST MEMORIES",
    ],
)
def test_memory_read_is_case_insensitive(tmp_path: Path, message: str) -> None:
    """Exact memory-list phrases match case-insensitively."""
    store = _memory_store(tmp_path)
    store.add_memory("listed")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle(message)

    assert result is not None
    assert result.action == "list"
    assert "listed" in result.safe_user_message


@pytest.mark.parametrize(
    "message",
    [
        "search my documents for phishing",
        "Search my documents for phishing",
        "SEARCH MY DOCUMENTS FOR PHISHING",
    ],
)
def test_document_search_is_case_insensitive(tmp_path: Path, message: str) -> None:
    """Document-search command wording matches case-insensitively."""
    vault = _document_vault(tmp_path)
    _add_document(vault, "Report covering phishing indicators.")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=vault,
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle(message)

    assert result is not None
    assert result.domain == "documents"
    assert "Document search results" in result.safe_user_message


def test_document_search_preserves_query_capitalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Captured document-search query keeps the user's original capitalization."""
    vault = _document_vault(tmp_path)
    _add_document(vault, "Phishing Indicators")
    retriever = LexicalDocumentRetriever()
    captured: dict[str, str] = {}
    original_search_vault = LexicalDocumentRetriever.search_vault

    def tracking_search_vault(
        self: LexicalDocumentRetriever,
        query: str,
        vault_arg: JsonDocumentVault,
        *,
        max_results: int | None = None,
    ) -> object:
        captured["query"] = query
        return original_search_vault(
            self,
            query,
            vault_arg,
            max_results=max_results,
        )

    monkeypatch.setattr(
        LexicalDocumentRetriever,
        "search_vault",
        tracking_search_vault,
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=vault,
        document_retriever=retriever,
        incident_repository=_incident_repository(tmp_path),
    )

    result = orchestrator.try_handle("Search my documents for Phishing Alerts")

    assert result is not None
    assert captured["query"] == "Phishing Alerts"


@pytest.mark.parametrize(
    "prefix",
    [
        "show incident",
        "Show incident",
        "SHOW INCIDENT",
    ],
)
def test_incident_read_is_case_insensitive(tmp_path: Path, prefix: str) -> None:
    """Show-incident command wording matches case-insensitively."""
    repo = _incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=_memory_store(tmp_path),
        document_vault=_document_vault(tmp_path),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=repo,
    )

    result = orchestrator.try_handle(f"{prefix} {incident.incident_id}")

    assert result is not None
    assert result.action == "incident_read"
    assert incident.incident_id in result.safe_user_message


@pytest.mark.parametrize(
    "message,needle",
    [
        ("Search Evidence", "/evidence-search"),
        ("RUN A TOOL", "/tool-"),
        ("Run The Playbook", "/playbook-"),
        ("Run AI Analysis", "/incident-analysis-"),
        ("SET A REMINDER", "/reminder-add"),
        ("List Reminders", "/reminders"),
        ("SHOW MY CALENDAR", "/calendar-events"),
        ("Schedule A Meeting", "/calendar-confirm"),
    ],
)
def test_guidance_phrases_are_case_insensitive(
    tmp_path: Path,
    message: str,
    needle: str,
) -> None:
    """Guidance allowlist phrases match case-insensitively."""
    result = _orchestrator(tmp_path).try_handle(message)
    assert result is not None
    assert result.domain == "guidance"
    assert needle in result.safe_user_message


# --- I. Collision suite --------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "I need to remember when this started",
        "I found a document about phishing last week",
        "show incident report to the team",
        "Can you remember what I said?",
        "We should search documents later",
    ],
)
def test_required_collision_phrases_fall_through(
    tmp_path: Path,
    phrase: str,
) -> None:
    """Required collision examples must fall through to ordinary conversation."""
    assert _orchestrator(tmp_path).try_handle(phrase) is None


def test_required_collisions_reach_handle_message_in_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Collision phrases must execute the ordinary conversation path."""
    handled: list[str] = []

    def fake_handle_message(
        *,
        user_message: str,
        **kwargs: object,
    ) -> None:
        handled.append(user_message)

    monkeypatch.setattr(
        "src.conversation_loop.handle_message",
        fake_handle_message,
    )

    collisions = [
        "I need to remember when this started",
        "I found a document about phishing last week",
        "show incident report to the team",
        "Can you remember what I said?",
        "We should search documents later",
        "exit",
    ]
    inputs = iter(collisions)
    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=FakeLogger(),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_document_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        memory_store=_memory_store(tmp_path),
        read_input=lambda: next(inputs),
    )

    assert handled == collisions[:-1]
