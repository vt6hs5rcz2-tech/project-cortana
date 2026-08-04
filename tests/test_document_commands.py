"""Tests for Knowledge Vault slash commands and AI isolation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pytest

import main as main_module
from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.commands import (
    ADD_DOCUMENT_MISSING_PATH,
    DOCUMENTS_EMPTY,
    DOCUMENT_MISSING_ID,
    HELP_TEXT,
    REMOVE_ALL_DOCUMENTS_NOT_CONFIRMED,
    REMOVE_ALL_DOCUMENTS_PROMPT,
    REMOVE_ALL_DOCUMENTS_SUCCESS,
    REMOVE_DOCUMENT_NOT_FOUND_TEMPLATE,
    CommandOutcome,
    CommandResult,
    format_status,
    handle_slash_command,
)
from src.config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    DOCUMENT_CONTEXT_INJECTION_ENABLED,
    KNOWLEDGE_VAULT_ENABLED,
    MAX_STORED_DOCUMENTS,
)
from src.conversation import ConversationHistory
from src.conversation_loop import run_conversation_loop
from src.document_extractor import DefaultTextExtractor
from src.document_vault import DocumentVault, JsonDocumentVault
from src.identity import CORTANA_SYSTEM_INSTRUCTIONS
from src.memory_store import JsonMemoryStore, MemoryStore
from src.settings import Settings
from tests.document_helpers import TEXT_PDF_BYTES

FAKE_CLIENT = cast(OpenAIClient, object())


class FakeLogger(logging.Logger):
    """Logger substitute used during document command tests."""

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


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _memory_store(tmp_path: Path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "memories.json")


def _vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _extractor() -> DefaultTextExtractor:
    return DefaultTextExtractor()


def _run(
    message: str,
    *,
    tmp_path: Path,
    history: ConversationHistory | None = None,
    memory_store: JsonMemoryStore | None = None,
    active: ActiveMemoryContext | None = None,
    vault: JsonDocumentVault | None = None,
) -> tuple[
    CommandResult,
    ConversationHistory,
    JsonMemoryStore,
    ActiveMemoryContext,
    JsonDocumentVault,
]:
    conversation_history = history or ConversationHistory()
    store = memory_store or _memory_store(tmp_path)
    active_context = active or ActiveMemoryContext()
    document_vault = vault or _vault(tmp_path)
    result = handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=conversation_history,
        memory_store=store,
        active_memory_context=active_context,
        document_vault=document_vault,
        document_extractor=_extractor(),
    )
    return result, conversation_history, store, active_context, document_vault


def test_help_lists_document_commands() -> None:
    """Help text should include Knowledge Vault commands."""
    assert "/add-document" in HELP_TEXT
    assert "/documents" in HELP_TEXT
    assert "/document" in HELP_TEXT
    assert "/remove-document" in HELP_TEXT
    assert "/remove-all-documents" in HELP_TEXT


def test_add_document_success_and_path_variants(tmp_path: Path) -> None:
    """ /add-document should ingest files, including spaced and quoted paths. """
    vault = _vault(tmp_path)
    spaced = tmp_path / "my notes.txt"
    spaced.write_text("spaced content", encoding="utf-8")

    result, history, _, _, vault = _run(
        f'/add-document "{spaced}"',
        tmp_path=tmp_path,
        vault=vault,
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "Document ingested" in result.message
    assert "my notes.txt" in result.message
    assert "Extracted characters: 14" in result.message
    assert history.turns == []
    assert vault.document_count() == 1

    unquoted_space = tmp_path / "another file.txt"
    unquoted_space.write_text("more content", encoding="utf-8")
    result2, _, _, _, vault = _run(
        f"/add-document {unquoted_space}",
        tmp_path=tmp_path,
        vault=vault,
    )
    assert result2.message is not None
    assert "Document ingested" in result2.message
    assert vault.document_count() == 2


def test_add_document_validation_paths(tmp_path: Path) -> None:
    """Missing path, unsupported type, and duplicates should return local errors."""
    result, _, _, _, _ = _run("/add-document", tmp_path=tmp_path)
    assert result.message == ADD_DOCUMENT_MISSING_PATH

    unsupported = tmp_path / "notes.exe"
    unsupported.write_bytes(b"data")
    result2, _, _, _, _ = _run(
        f"/add-document {unsupported}",
        tmp_path=tmp_path,
    )
    assert result2.message is not None
    assert "Unsupported document type" in result2.message

    source = tmp_path / "dup.txt"
    source.write_text("duplicate-body", encoding="utf-8")
    first, _, _, _, vault = _run(f"/add-document {source}", tmp_path=tmp_path)
    assert first.message is not None
    document_id = vault.list_documents()[0].id

    copy = tmp_path / "dup-copy.txt"
    copy.write_text("duplicate-body", encoding="utf-8")
    duplicate, _, _, _, _ = _run(
        f"/add-document {copy}",
        tmp_path=tmp_path,
        vault=vault,
    )
    assert duplicate.message is not None
    assert document_id in duplicate.message


def test_documents_list_and_empty_state(tmp_path: Path) -> None:
    """ /documents should list metadata or report an empty vault. """
    empty, _, _, _, _ = _run("/documents", tmp_path=tmp_path)
    assert empty.message == DOCUMENTS_EMPTY

    source = tmp_path / "listed.txt"
    source.write_text("list me", encoding="utf-8")
    _, _, _, _, vault = _run(f"/add-document {source}", tmp_path=tmp_path)
    listed, _, _, _, _ = _run("/documents", tmp_path=tmp_path, vault=vault)

    assert listed.message is not None
    assert "Stored documents:" in listed.message
    assert "listed.txt" in listed.message
    assert "list me" not in listed.message


def test_document_inspect_success_and_errors(tmp_path: Path) -> None:
    """ /document should show stored source content or clear local errors. """
    missing_arg, _, _, _, _ = _run("/document", tmp_path=tmp_path)
    assert missing_arg.message == DOCUMENT_MISSING_ID

    missing, _, _, _, _ = _run(
        "/document 00000000-0000-0000-0000-000000000000",
        tmp_path=tmp_path,
    )
    assert missing.message is not None
    assert "No stored document found" in missing.message

    source = tmp_path / "inspect.md"
    source.write_text("# inspect body", encoding="utf-8")
    _, _, _, _, vault = _run(f"/add-document {source}", tmp_path=tmp_path)
    document_id = vault.list_documents()[0].id

    inspected, _, _, _, _ = _run(
        f"/document {document_id}",
        tmp_path=tmp_path,
        vault=vault,
    )
    assert inspected.message is not None
    assert "Locally stored Knowledge Vault source content" in inspected.message
    assert "# inspect body" in inspected.message
    assert str(source.resolve()) not in inspected.message


def test_remove_document_and_remove_all_confirmation(tmp_path: Path) -> None:
    """Document deletion commands should require exact confirmation for wipe-all."""
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    _, _, memory_store, active, vault = _run(
        f"/add-document {first}",
        tmp_path=tmp_path,
    )
    memory_store.add_memory("keep this memory")
    active.activate(memory_store.list_memories()[0])
    _, _, _, _, vault = _run(
        f"/add-document {second}",
        tmp_path=tmp_path,
        memory_store=memory_store,
        active=active,
        vault=vault,
    )
    document_id = vault.list_documents()[0].id

    removed, _, _, active_after_remove, vault = _run(
        f"/remove-document {document_id}",
        tmp_path=tmp_path,
        memory_store=memory_store,
        active=active,
        vault=vault,
    )
    assert removed.message is not None
    assert document_id in removed.message
    assert vault.document_count() == 1
    assert memory_store.list_memories()
    assert active_after_remove.active_count == 1

    missing = _run(
        "/remove-document 00000000-0000-0000-0000-000000000000",
        tmp_path=tmp_path,
        vault=vault,
    )[0]
    assert missing.message == REMOVE_DOCUMENT_NOT_FOUND_TEMPLATE.format(
        document_id="00000000-0000-0000-0000-000000000000"
    )

    prompt = _run("/remove-all-documents", tmp_path=tmp_path, vault=vault)[0]
    assert prompt.message == REMOVE_ALL_DOCUMENTS_PROMPT
    assert vault.document_count() == 1

    incorrect = _run(
        "/remove-all-documents yes",
        tmp_path=tmp_path,
        vault=vault,
    )[0]
    assert incorrect.message == REMOVE_ALL_DOCUMENTS_NOT_CONFIRMED
    assert vault.document_count() == 1

    confirmed, history, memory_store, active, vault = _run(
        "/remove-all-documents confirm",
        tmp_path=tmp_path,
        memory_store=memory_store,
        active=active,
        vault=vault,
    )
    assert confirmed.message == REMOVE_ALL_DOCUMENTS_SUCCESS
    assert vault.document_count() == 0
    assert memory_store.list_memories()
    assert active.active_count == 1
    assert history.turns == []


def test_document_commands_avoid_ai_and_preserve_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Document commands must not call AI or alter conversation/memory state."""
    logger = FakeLogger()
    called = False
    source = tmp_path / "ai-free.txt"
    source.write_text("no ai please", encoding="utf-8")
    memory_store = _memory_store(tmp_path)
    memory = memory_store.add_memory("persistent")
    active = ActiveMemoryContext()
    active.activate(memory)
    history = ConversationHistory()
    history.add_user_message("prior")
    history.add_assistant_message("reply")
    vault = _vault(tmp_path)
    inputs = iter(
        [
            f"/add-document {source}",
            "/documents",
            "/remove-all-documents",
            "/remove-all-documents confirm",
            "exit",
        ]
    )

    def fake_generate_response(**kwargs: object) -> str:
        nonlocal called
        called = True
        return "should-not-run"

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        memory_store=memory_store,
        active_memory_context=active,
        document_vault=vault,
        document_extractor=_extractor(),
        read_input=lambda: next(inputs),
        conversation_history=history,
    )
    capsys.readouterr()

    assert called is False
    assert history.completed_turn_count == 1
    assert memory_store.list_memories()[0].text == "persistent"
    assert active.active_count == 1
    assert vault.document_count() == 0


def test_path_like_messages_still_reach_ai(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ordinary absolute-path cybersecurity content should still call the AI."""
    logger = FakeLogger()
    received: list[str] = []
    inputs = iter(["/etc/passwd", "exit"])

    def fake_generate_response(
        *,
        client: object,
        settings: Settings,
        user_message: str,
        conversation_history: ConversationHistory | None = None,
        active_memories: object = None,
        memory_boundary_token: object = None,
    ) -> str:
        received.append(user_message)
        return "path analyzed"

    monkeypatch.setattr(
        "src.conversation_loop.generate_response",
        fake_generate_response,
    )

    run_conversation_loop(
        client=FAKE_CLIENT,
        settings=_settings(),
        logger=logger,
        memory_store=_memory_store(tmp_path),
        active_memory_context=ActiveMemoryContext(),
        document_vault=_vault(tmp_path),
        document_extractor=_extractor(),
        read_input=lambda: next(inputs),
    )

    assert received == ["/etc/passwd"]


def test_document_records_absent_from_ai_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Document vault content must remain absent from AI request construction."""
    logger = FakeLogger()
    source = tmp_path / "secret-doc.txt"
    secret = "vault-secret-text-xyz"
    source.write_text(secret, encoding="utf-8")
    vault = _vault(tmp_path)
    handle_slash_command(
        f"/add-document {source}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=_memory_store(tmp_path),
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=_extractor(),
    )

    captured_input: object | None = None
    captured_instructions: str | None = None

    class FakeResponses:
        def create(
            self,
            *,
            model: str,
            input: object,
            instructions: str | None = None,
        ) -> object:
            nonlocal captured_input, captured_instructions
            captured_input = input
            captured_instructions = instructions

            class Response:
                output_text = "ok"

            return Response()

    class FakeClient:
        responses = FakeResponses()

    from src.ai_service import generate_response

    generate_response(
        client=cast(OpenAIClient, FakeClient()),
        settings=_settings(),
        user_message="Analyze firewall rules",
    )

    serialized = repr(captured_input)
    assert secret not in serialized
    assert vault.list_documents()[0].id not in serialized
    assert captured_instructions == CORTANA_SYSTEM_INSTRUCTIONS
    assert secret not in CORTANA_SYSTEM_INSTRUCTIONS


def test_status_reports_knowledge_vault_safely(tmp_path: Path) -> None:
    """ /status should report vault capacity and keep paths/secrets hidden. """
    vault = _vault(tmp_path)
    source = tmp_path / "status.txt"
    source.write_text("status body", encoding="utf-8")
    handle_slash_command(
        f"/add-document {source}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=_memory_store(tmp_path),
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=_extractor(),
    )

    status = format_status(
        _settings(),
        ConversationHistory(),
        _memory_store(tmp_path),
        ActiveMemoryContext(),
        vault,
    )
    lowered = status.lower()

    assert "Knowledge Vault: enabled" in status
    assert KNOWLEDGE_VAULT_ENABLED is True
    assert "Stored documents: 1" in status
    assert f"Maximum documents: {MAX_STORED_DOCUMENTS}" in status
    assert "Supported document types:" in status
    for extension in sorted(ALLOWED_DOCUMENT_EXTENSIONS):
        assert extension in status
    assert "Document context injection: disabled" in status
    assert DOCUMENT_CONTEXT_INJECTION_ENABLED is False
    assert "documents.json" not in lowered
    assert str(vault.file_path).lower() not in lowered
    assert str(source.resolve()).lower() not in lowered
    assert "status body" not in lowered
    assert "test-api-key" not in lowered
    assert "openai_api_key" not in lowered


def test_startup_injects_one_vault_and_reload_preserves_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Main should create one vault instance and restart should reload documents."""
    logger = FakeLogger()
    fake_settings = _settings()
    fake_client = FAKE_CLIENT
    memory_path = tmp_path / "memories.json"
    vault_path = tmp_path / "documents.json"
    received_vaults: list[DocumentVault] = []

    source = tmp_path / "persist.txt"
    source.write_text("survives restart", encoding="utf-8")
    first_vault = JsonDocumentVault(vault_path)
    handle_slash_command(
        f"/add-document {source}",
        settings=fake_settings,
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(memory_path),
        active_memory_context=ActiveMemoryContext(),
        document_vault=first_vault,
        document_extractor=_extractor(),
    )

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

    def fake_run_conversation_loop(
        *,
        client: object,
        settings: Settings,
        logger: logging.Logger,
        memory_store: MemoryStore,
        active_memory_context: ActiveMemoryContext,
        document_vault: DocumentVault,
        document_extractor: DefaultTextExtractor,
    ) -> None:
        received_vaults.append(document_vault)

    monkeypatch.setattr(
        main_module,
        "run_conversation_loop",
        fake_run_conversation_loop,
    )

    main_module.main()

    assert len(received_vaults) == 1
    assert isinstance(received_vaults[0], JsonDocumentVault)
    reloaded = received_vaults[0].list_documents()
    assert len(reloaded) == 1
    assert reloaded[0].extracted_text == "survives restart"


def test_pdf_ingestion_through_command(tmp_path: Path) -> None:
    """ /add-document should ingest PDF text through the live command path. """
    pdf_path = tmp_path / "brief.pdf"
    pdf_path.write_bytes(TEXT_PDF_BYTES)

    result, _, _, _, vault = _run(f"/add-document {pdf_path}", tmp_path=tmp_path)

    assert result.message is not None
    assert "Document ingested" in result.message
    assert "Hello PDF" in vault.list_documents()[0].extracted_text
