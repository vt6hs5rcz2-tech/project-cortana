"""Tests for /search-docs, /ask-docs, /sources, and retrieval session wiring."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

import main as main_module
from src.active_memory import ActiveMemoryContext
from src.ai_service import (
    GROUNDED_DOCUMENT_INSTRUCTIONS,
    AIResponse,
    OpenAIClient,
    ResponsesClient,
)
from src.citation_validation import UNSUPPORTED_CITATION_MARKER
from src.commands import (
    ASK_DOCS_EMPTY_VAULT,
    ASK_DOCS_MISSING_QUESTION,
    ASK_DOCS_NO_EVIDENCE,
    CLEAR_CONFIRMATION,
    HELP_TEXT,
    SEARCH_DOCS_EMPTY_VAULT,
    SEARCH_DOCS_MISSING_QUERY,
    SEARCH_DOCS_NO_RESULTS,
    SOURCES_EMPTY,
    CommandOutcome,
    format_status,
    handle_slash_command,
)
from src.config import (
    MAX_RETRIEVED_CHUNKS,
    MAX_RETRIEVED_CONTEXT_CHARS,
    MAX_SEARCH_RESULT_PREVIEW_CHARS,
)
from src.conversation import ConversationApiInput, ConversationHistory
from src.conversation_loop import run_conversation_loop
from src.document_chunker import DocumentChunker
from src.document_context import DOCUMENT_CONTEXT_PREAMBLE
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.identity import CORTANA_SYSTEM_INSTRUCTIONS
from src.memory import MemoryRecord
from src.memory_store import JsonMemoryStore
from src.retrieval_session import RetrievalSession
from src.settings import Settings

DOC_BOUNDARY = "doc_session_token_01"
MEM_BOUNDARY = "mem_session_token_01"


def _grounded_json(
    answer: str = "Answer from [DOC-1:C1]",
    *,
    support: str = "supported",
    citations: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "support": support,
            "citations": citations if citations is not None else ["[DOC-1:C1]"],
        }
    )


@dataclass
class FakeAIResponse:
    """Minimal AI response used by the fake Responses API."""

    output_text: str


class FakeResponses:
    """Fake Responses API used without network access."""

    def __init__(
        self,
        output_text: str | list[str] | None = None,
    ) -> None:
        if output_text is None:
            self._outputs = [_grounded_json()]
        elif isinstance(output_text, list):
            self._outputs = output_text
        else:
            self._outputs = [output_text]
        self.calls = 0
        self.model: str | None = None
        self.input: ConversationApiInput | None = None
        self.instructions: str | None = None
        self.inputs: list[ConversationApiInput] = []

    def create(
        self,
        *,
        model: str,
        input: ConversationApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        self.calls += 1
        self.model = model
        self.input = input
        self.inputs.append(input)
        self.instructions = instructions
        index = min(self.calls - 1, len(self._outputs) - 1)
        return FakeAIResponse(output_text=self._outputs[index])


class FakeClient:
    """Fake OpenAI client containing the fake Responses API."""

    responses: ResponsesClient

    def __init__(self, output_text: str | list[str] | None = None) -> None:
        self.responses = FakeResponses(output_text=output_text)

    @property
    def fake_responses(self) -> FakeResponses:
        assert isinstance(self.responses, FakeResponses)
        return self.responses


class FakeLogger(logging.Logger):
    """Logger substitute used during retrieval command tests."""

    def __init__(self) -> None:
        super().__init__("ProjectCortanaTest")
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        message = str(msg)
        self.info_messages.append(message % args if args else message)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        message = str(msg)
        self.error_messages.append(message % args if args else message)


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _store(tmp_path: Path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "memories.json")


def _run(
    message: str,
    *,
    tmp_path: Path,
    history: ConversationHistory | None = None,
    memory_store: JsonMemoryStore | None = None,
    active: ActiveMemoryContext | None = None,
    vault: JsonDocumentVault | None = None,
    retriever: LexicalDocumentRetriever | None = None,
    session: RetrievalSession | None = None,
    client: FakeClient | None = None,
) -> tuple[Any, ConversationHistory, JsonDocumentVault, RetrievalSession, FakeClient]:
    conversation_history = history or ConversationHistory()
    document_vault = vault or _vault(tmp_path)
    retrieval_session = session or RetrievalSession(boundary_token=DOC_BOUNDARY)
    fake_client = client or FakeClient()
    result = handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=conversation_history,
        memory_store=memory_store or _store(tmp_path),
        active_memory_context=active or ActiveMemoryContext(boundary_token=MEM_BOUNDARY),
        document_vault=document_vault,
        document_extractor=DefaultTextExtractor(),
        document_retriever=retriever or LexicalDocumentRetriever(),
        retrieval_session=retrieval_session,
        client=cast(OpenAIClient, fake_client),
    )
    return result, conversation_history, document_vault, retrieval_session, fake_client


def _add_document(tmp_path: Path, vault: JsonDocumentVault, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    result, _, _, _, _ = _run(f"/add-document {path}", tmp_path=tmp_path, vault=vault)
    assert result.message is not None
    assert "Document ingested" in result.message
    return path


def test_help_lists_retrieval_commands() -> None:
    """Help text should include search-docs, ask-docs, and sources."""
    assert "/search-docs" in HELP_TEXT
    assert "/ask-docs" in HELP_TEXT
    assert "/doc-summary" in HELP_TEXT
    assert "/docs-compare" in HELP_TEXT
    assert "/sources" in HELP_TEXT


def test_search_docs_missing_empty_no_results_and_no_ai(tmp_path: Path) -> None:
    """ /search-docs should handle empty states locally without AI calls. """
    client = FakeClient()
    result, _, _, _, client = _run(
        "/search-docs",
        tmp_path=tmp_path,
        client=client,
    )
    assert result.message == SEARCH_DOCS_MISSING_QUERY
    assert client.fake_responses.calls == 0

    result, _, vault, _, client = _run(
        "/search-docs firewall",
        tmp_path=tmp_path,
        client=client,
    )
    assert result.message == SEARCH_DOCS_EMPTY_VAULT
    assert client.fake_responses.calls == 0

    _add_document(tmp_path, vault, "notes.txt", "Completely unrelated content.")
    result, _, _, _, client = _run(
        "/search-docs nonexistenttermxyz",
        tmp_path=tmp_path,
        vault=vault,
        client=client,
    )
    assert result.message == SEARCH_DOCS_NO_RESULTS
    assert client.fake_responses.calls == 0


def test_search_docs_ranked_previews_are_bounded_and_path_safe(tmp_path: Path) -> None:
    """Search results should show bounded previews without path exposure."""
    vault = _vault(tmp_path)
    source = _add_document(
        tmp_path,
        vault,
        "policy.txt",
        "Firewall policy " + ("detail " * 80) + "endpoint hardening.",
    )
    client = FakeClient()
    result, _, _, _, client = _run(
        "/search-docs firewall policy",
        tmp_path=tmp_path,
        vault=vault,
        client=client,
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "[DOC-1:C1]" in result.message
    assert "filename=policy.txt" in result.message
    assert "chunk_index=" in result.message
    assert "Firewall policy" in result.message
    assert str(source.resolve()) not in result.message
    assert str(vault.file_path) not in result.message
    assert "detail detail" in result.message
    preview_lines = [
        line.strip()
        for line in result.message.splitlines()
        if line.startswith("    ")
    ]
    assert preview_lines
    assert len(preview_lines[0]) <= MAX_SEARCH_RESULT_PREVIEW_CHARS + 3
    assert client.fake_responses.calls == 0


def test_absolute_path_still_reaches_ordinary_ai_chat(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Absolute path-like input should remain ordinary AI conversation."""
    client = FakeClient(output_text="Path handled")
    logger = FakeLogger()
    inputs = iter(["/etc/passwd", "exit"])

    run_conversation_loop(
        client=cast(OpenAIClient, client),
        settings=_settings(),
        logger=logger,
        memory_store=_store(tmp_path),
        active_memory_context=ActiveMemoryContext(boundary_token=MEM_BOUNDARY),
        document_vault=_vault(tmp_path),
        document_extractor=DefaultTextExtractor(),
        document_chunker=DocumentChunker(),
        document_retriever=LexicalDocumentRetriever(),
        retrieval_session=RetrievalSession(boundary_token=DOC_BOUNDARY),
        read_input=lambda: next(inputs),
    )

    output = capsys.readouterr().out
    assert client.fake_responses.calls == 1
    assert client.fake_responses.input == "/etc/passwd"
    assert client.fake_responses.instructions == CORTANA_SYSTEM_INSTRUCTIONS
    assert "Path handled" in output


def test_ask_docs_missing_empty_and_no_evidence_skip_ai(tmp_path: Path) -> None:
    """ /ask-docs should not call AI without a question, vault, or evidence. """
    client = FakeClient()
    result, _, _, session, client = _run("/ask-docs", tmp_path=tmp_path, client=client)
    assert result.message == ASK_DOCS_MISSING_QUESTION
    assert client.fake_responses.calls == 0
    assert session.has_source_manifest is False

    result, _, vault, session, client = _run(
        "/ask-docs What is the firewall rule?",
        tmp_path=tmp_path,
        client=client,
    )
    assert result.message == ASK_DOCS_EMPTY_VAULT
    assert client.fake_responses.calls == 0

    _add_document(tmp_path, vault, "other.txt", "Unrelated gardening advice.")
    result, history, _, session, client = _run(
        "/ask-docs What is the firewall rule?",
        tmp_path=tmp_path,
        vault=vault,
        client=client,
    )
    assert result.message == ASK_DOCS_NO_EVIDENCE
    assert client.fake_responses.calls == 0
    assert session.has_source_manifest is False
    assert history.turns == []


def test_ask_docs_sends_only_retrieved_chunks_and_records_sources(
    tmp_path: Path,
) -> None:
    """Successful /ask-docs should send selected chunks and update session state."""
    vault = _vault(tmp_path)
    _add_document(
        tmp_path,
        vault,
        "firewall.txt",
        "The firewall blocks outbound SSH from guest networks.",
    )
    _add_document(
        tmp_path,
        vault,
        "gardening.txt",
        "Tomatoes prefer warm soil and consistent watering.",
    )
    client = FakeClient(
        output_text=_grounded_json("Outbound SSH is blocked [DOC-1:C1].")
    )
    active = ActiveMemoryContext(boundary_token=MEM_BOUNDARY)
    memory_store = _store(tmp_path)
    memory = memory_store.add_memory("Analyst prefers concise answers.")
    active.activate(memory)

    result, history, _, session, client = _run(
        "/ask-docs What does the firewall block?",
        tmp_path=tmp_path,
        vault=vault,
        active=active,
        memory_store=memory_store,
        client=client,
    )

    assert result.message is not None
    assert "Outbound SSH is blocked [DOC-1:C1]." in result.message
    assert "Support: supported" in result.message
    assert client.fake_responses.calls == 1
    assert client.fake_responses.instructions == GROUNDED_DOCUMENT_INSTRUCTIONS
    api_input = client.fake_responses.input
    assert isinstance(api_input, list)
    assert api_input[0]["role"] == "developer"
    assert DOCUMENT_CONTEXT_PREAMBLE in api_input[0]["content"]
    assert "firewall blocks outbound SSH" in api_input[0]["content"]
    assert "Tomatoes prefer warm soil" not in api_input[0]["content"]
    assert "Analyst prefers concise answers." not in api_input[0]["content"]
    assert len(api_input) == 2
    assert api_input[-1]["role"] == "user"
    assert "What does the firewall block?" in api_input[-1]["content"]
    assert session.has_source_manifest is True
    assert history.completed_turn_count == 0
    assert history.turns == []

    sources, _, _, _, client2 = _run(
        "/sources",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
        client=FakeClient(),
    )
    assert sources.message is not None
    assert "[DOC-1:C1]" in sources.message
    assert "document_id=" in sources.message
    assert "filename=firewall.txt" in sources.message
    assert "chunk_index=" in sources.message
    assert "chars=" in sources.message
    assert "firewall blocks outbound SSH" not in sources.message
    assert client2.fake_responses.calls == 0


def test_ask_docs_detects_invented_citations(tmp_path: Path) -> None:
    """Invented citation labels should be marked unsupported."""
    vault = _vault(tmp_path)
    _add_document(tmp_path, vault, "rules.txt", "MFA is required for VPN access.")
    client = FakeClient(
        output_text=_grounded_json(
            "Required per [DOC-1:C1] and [DOC-9:C9].",
            citations=["[DOC-1:C1]", "[DOC-9:C9]"],
        )
    )

    result, _, _, _, _ = _run(
        "/ask-docs Is MFA required?",
        tmp_path=tmp_path,
        vault=vault,
        client=client,
    )

    assert result.message is not None
    assert "[DOC-1:C1]" in result.message
    assert UNSUPPORTED_CITATION_MARKER in result.message
    assert "[DOC-9:C9]" not in result.message
    assert "Support: partial" in result.message


def test_ask_docs_ai_failure_does_not_corrupt_session(tmp_path: Path) -> None:
    """AI failures should leave the vault and source manifest unchanged."""
    vault = _vault(tmp_path)
    _add_document(tmp_path, vault, "rules.txt", "Rotate API keys every 90 days.")

    class BoomClient:
        class responses:
            @staticmethod
            def create(**kwargs: object) -> AIResponse:
                raise RuntimeError("network down")

    session = RetrievalSession(boundary_token=DOC_BOUNDARY)
    result = handle_slash_command(
        "/ask-docs How often should keys rotate?",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=_store(tmp_path),
        active_memory_context=ActiveMemoryContext(boundary_token=MEM_BOUNDARY),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        document_retriever=LexicalDocumentRetriever(),
        retrieval_session=session,
        client=cast(OpenAIClient, BoomClient()),
    )

    assert result.message == "Cortana: I could not complete that request."
    assert session.has_source_manifest is False
    assert vault.document_count() == 1


def test_ask_docs_malformed_json_fails_closed(tmp_path: Path) -> None:
    """Malformed grounded model output must not be shown raw."""
    vault = _vault(tmp_path)
    _add_document(tmp_path, vault, "rules.txt", "Rotate API keys every 90 days.")
    result, history, _, session, _ = _run(
        "/ask-docs How often should keys rotate?",
        tmp_path=tmp_path,
        vault=vault,
        client=FakeClient(output_text="not-json-at-all"),
    )
    assert result.message == "Cortana: I could not complete that request."
    assert "not-json-at-all" not in (result.message or "")
    assert session.has_source_manifest is False
    assert history.turns == []


def test_sources_empty_clear_and_remove_all_reset(tmp_path: Path) -> None:
    """ /sources empty state, /clear, and remove-all should reset the manifest. """
    result, _, _, session, _ = _run("/sources", tmp_path=tmp_path)
    assert result.message == SOURCES_EMPTY

    vault = _vault(tmp_path)
    _add_document(tmp_path, vault, "a.txt", "Incident response requires triage notes.")
    history = ConversationHistory()
    history.add_user_message("unrelated chat")
    history.add_assistant_message("unrelated answer")
    ask, history, _, session, _ = _run(
        "/ask-docs What does incident response require?",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
        history=history,
        client=FakeClient(output_text=_grounded_json("Triage notes [DOC-1:C1].")),
    )
    assert ask.message is not None
    assert session.has_source_manifest is True
    assert history.completed_turn_count == 1

    clear, history, _, session, _ = _run(
        "/clear",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
        history=history,
    )
    assert clear.message == CLEAR_CONFIRMATION
    assert history.turns == []
    assert session.has_source_manifest is False

    ask, _, _, session, _ = _run(
        "/ask-docs What does incident response require?",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
        client=FakeClient(output_text=_grounded_json("Triage notes [DOC-1:C1].")),
    )
    assert session.has_source_manifest is True
    remove, _, _, session, _ = _run(
        "/remove-all-documents confirm",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
    )
    assert remove.message is not None
    assert "deleted" in remove.message.lower()
    assert session.has_source_manifest is False
    assert vault.document_count() == 0


def test_remove_document_invalidates_stale_manifest_entries(tmp_path: Path) -> None:
    """Deleting one document should drop matching source-manifest entries."""
    vault = _vault(tmp_path)
    _add_document(tmp_path, vault, "keep.txt", "Retention policy keeps logs 30 days.")
    _add_document(tmp_path, vault, "drop.txt", "Retention policy purge schedule weekly.")
    session = RetrievalSession(boundary_token=DOC_BOUNDARY)
    ask, _, vault, session, _ = _run(
        "/ask-docs What is the retention policy?",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
        client=FakeClient(
            output_text=_grounded_json(
                "See [DOC-1:C1] and [DOC-2:C1].",
                citations=["[DOC-1:C1]", "[DOC-2:C1]"],
            )
        ),
    )
    assert ask.message is not None
    assert len(session.source_manifest) >= 1

    target = session.source_manifest[0]
    result, _, _, session, _ = _run(
        f"/remove-document {target.document_id}",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
    )
    assert result.message is not None
    assert target.document_id not in {
        entry.document_id for entry in session.source_manifest
    }


def test_status_reports_retrieval_capabilities_safely(tmp_path: Path) -> None:
    """ /status should report retrieval flags and limits without secrets. """
    vault = _vault(tmp_path)
    source = _add_document(tmp_path, vault, "secret-name.txt", "confidential body text")
    session = RetrievalSession(boundary_token=DOC_BOUNDARY)
    _run(
        "/ask-docs confidential body",
        tmp_path=tmp_path,
        vault=vault,
        session=session,
        client=FakeClient(output_text=_grounded_json("Noted [DOC-1:C1].")),
    )

    status = format_status(
        _settings(),
        ConversationHistory(),
        _store(tmp_path),
        ActiveMemoryContext(boundary_token=MEM_BOUNDARY),
        vault,
        session,
    )
    lowered = status.lower()

    assert "Local document retrieval: enabled" in status
    assert "Semantic retrieval: disabled" in status
    assert (
        "Document context injection: enabled "
        "(explicit /ask-docs, /doc-summary, /docs-compare)"
    ) in status
    assert f"Maximum retrieved chunks: {MAX_RETRIEVED_CHUNKS}" in status
    assert (
        f"Maximum retrieved context characters: {MAX_RETRIEVED_CONTEXT_CHARS}"
        in status
    )
    assert "Current source manifest: present" in status
    assert "Source manifest persistence: disabled" in status
    assert "secret-name.txt" not in status
    assert "confidential body text" not in lowered
    assert "secret question text" not in lowered
    assert str(source.resolve()).lower() not in lowered
    assert str(vault.file_path).lower() not in lowered
    assert "test-api-key" not in lowered
    assert DOC_BOUNDARY not in status


def test_logs_omit_document_and_question_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Safe logs should omit document text and full grounded questions."""
    vault = _vault(tmp_path)
    _add_document(
        tmp_path,
        vault,
        "logcheck.txt",
        "UniqueDocumentBodyTextXYZ for logging privacy.",
    )
    question = "Does UniqueDocumentBodyTextXYZ appear in logs?"

    with caplog.at_level(logging.INFO, logger="ProjectCortana"):
        result, _, _, _, _ = _run(
            f"/ask-docs {question}",
            tmp_path=tmp_path,
            vault=vault,
            client=FakeClient(output_text=_grounded_json("Yes [DOC-1:C1].")),
        )

    assert result.message is not None
    combined = "\n".join(caplog.messages)
    assert "UniqueDocumentBodyTextXYZ" not in combined
    assert question not in combined
    assert "task_type=ask" in combined


def test_new_application_session_starts_with_empty_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Main should inject a fresh empty retrieval session each start."""
    logger = FakeLogger()
    fake_settings = _settings()
    fake_client = cast(OpenAIClient, object())
    received_sessions: list[RetrievalSession] = []

    monkeypatch.setattr(main_module, "setup_logging", lambda: logger)
    monkeypatch.setattr(
        main_module,
        "initialize_ai",
        lambda supplied_logger: (fake_settings, fake_client),
    )
    monkeypatch.setattr(
        main_module,
        "get_default_memory_file_path",
        lambda: tmp_path / "memories.json",
    )
    monkeypatch.setattr(
        main_module,
        "get_default_document_vault_file_path",
        lambda: tmp_path / "documents.json",
    )

    def fake_run_conversation_loop(
        *,
        client: object,
        settings: Settings,
        logger: logging.Logger,
        memory_store: object,
        active_memory_context: object,
        document_vault: object,
        document_extractor: object,
        document_chunker: object,
        document_retriever: object,
        retrieval_session: RetrievalSession,
        **_: object,
    ) -> None:
        received_sessions.append(retrieval_session)

    monkeypatch.setattr(
        main_module,
        "run_conversation_loop",
        fake_run_conversation_loop,
    )

    main_module.main()
    assert len(received_sessions) == 1
    assert received_sessions[0].has_source_manifest is False


def test_retrieved_chunks_do_not_become_memories_or_active_context(
    tmp_path: Path,
) -> None:
    """Grounded answers must not create persistent or active memories."""
    vault = _vault(tmp_path)
    memory_store = _store(tmp_path)
    active = ActiveMemoryContext(boundary_token=MEM_BOUNDARY)
    _add_document(tmp_path, vault, "onlydocs.txt", "Containment steps are documented.")

    _run(
        "/ask-docs What are the containment steps?",
        tmp_path=tmp_path,
        vault=vault,
        memory_store=memory_store,
        active=active,
        client=FakeClient(output_text=_grounded_json("Documented [DOC-1:C1].")),
    )

    assert memory_store.list_memories() == []
    assert active.active_count == 0
    assert isinstance(active.list_active(), list)
    assert all(isinstance(item, MemoryRecord) for item in active.list_active())
