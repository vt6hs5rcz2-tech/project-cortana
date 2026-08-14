"""Permanent Batch 5 tests: persistence caps, DST, evidence rollback, optional calendar."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.calendar_commands import (
    CALENDAR_UNAVAILABLE_MESSAGE,
    CalendarCommandContext,
    handle_calendar_command,
)
from src.calendar_models import CalendarValidationError, local_wall_pair_to_utc
from src.calendar_service import optional_calendar_dependencies_available
from src.commands import handle_slash_command
from src.config import MAX_MEMORY_TEXT_LENGTH
from src.conversation import ConversationHistory
from src.document_extractor import DefaultTextExtractor
from src.document_retrieval import DocumentRetrievalError, LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.evidence_store import LocalEvidenceStore, evidence_storage_filename
from src.incident_repository import (
    IncidentCountLimitError,
    IncidentStorageError,
    JsonIncidentRepository,
)
from src.memory_store import JsonMemoryStore, MemoryStorageError
from src.reminder import (
    DST_AMBIGUOUS_LOCAL_TIME_MESSAGE,
    DST_NONEXISTENT_LOCAL_TIME_MESSAGE,
    ReminderValidationError,
    local_wall_to_utc_iso,
    parse_local_wall_datetime,
)
from src.security_commands import SecurityCommandContext, handle_security_command
from src.security_incident import create_security_incident
from src.settings import Settings
from src.tool_audit import create_tool_audit_entry
from src.tool_repository import JsonToolControlRepository, ToolCountLimitError, ToolStorageError
from tests.security_helpers import incident_repository
from tests.tool_helpers import make_scope


SECRET_LEAK = r"C:\Users\secret\private.env sk-test-123"


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


def test_tool_scope_cap_exact_and_plus_one(tmp_path: Path) -> None:
    repo = JsonToolControlRepository(tmp_path / "tool_control.json", max_scopes=2)
    repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="one"))
    repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="two"))
    with pytest.raises(ToolCountLimitError):
        repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="three"))
    reloaded = JsonToolControlRepository(tmp_path / "tool_control.json", max_scopes=2)
    assert len(reloaded.list_scopes()) == 2


def test_tool_scope_load_rejects_cap_plus_one(tmp_path: Path) -> None:
    writer = JsonToolControlRepository(tmp_path / "tool_control.json", max_scopes=3)
    writer.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="a"))
    writer.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="b"))
    writer.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="c"))
    original = (tmp_path / "tool_control.json").read_text(encoding="utf-8")
    reader = JsonToolControlRepository(tmp_path / "tool_control.json", max_scopes=2)
    with pytest.raises(ToolStorageError):
        reader.list_scopes()
    assert (tmp_path / "tool_control.json").read_text(encoding="utf-8") == original


def test_tool_audit_trims_oldest_on_write(tmp_path: Path) -> None:
    repo = JsonToolControlRepository(
        tmp_path / "tool_control.json",
        max_scopes=10,
        max_audit_entries=2,
    )
    repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="first"))
    repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="second"))
    repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"], name="third"))
    entries = repo.list_audit_entries()
    assert len(entries) == 2
    assert entries[0].safe_details.get("active") is True


def test_tool_append_audit_alternate_path_is_capped(tmp_path: Path) -> None:
    repo = JsonToolControlRepository(tmp_path / "tool_control.json", max_audit_entries=2)
    repo.append_audit_entry(create_tool_audit_entry(action="scope-created"))
    repo.append_audit_entry(create_tool_audit_entry(action="request-created"))
    repo.append_audit_entry(create_tool_audit_entry(action="approval-recorded"))
    assert len(repo.list_audit_entries()) == 2


def test_incident_cap_exact_and_plus_one(tmp_path: Path) -> None:
    repo = JsonIncidentRepository(tmp_path / "incidents.json", max_incidents=2)
    repo.add_incident(create_security_incident(title="One", summary="a", severity="low"))
    repo.add_incident(create_security_incident(title="Two", summary="b", severity="low"))
    with pytest.raises(IncidentCountLimitError):
        repo.add_incident(create_security_incident(title="Three", summary="c", severity="low"))
    reloaded = JsonIncidentRepository(tmp_path / "incidents.json", max_incidents=2)
    assert reloaded.incident_count() == 2


def test_incident_load_rejects_cap_plus_one(tmp_path: Path) -> None:
    writer = JsonIncidentRepository(tmp_path / "incidents.json", max_incidents=2)
    writer.add_incident(create_security_incident(title="One", summary="a", severity="low"))
    writer.add_incident(create_security_incident(title="Two", summary="b", severity="low"))
    original = (tmp_path / "incidents.json").read_text(encoding="utf-8")
    reader = JsonIncidentRepository(tmp_path / "incidents.json", max_incidents=1)
    with pytest.raises(IncidentStorageError):
        reader.list_incidents()
    assert (tmp_path / "incidents.json").read_text(encoding="utf-8") == original


def test_malformed_incident_does_not_affect_memory(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memories.json")
    memory.add_memory("kept")
    incidents = tmp_path / "incidents.json"
    incidents.write_text("{not json", encoding="utf-8")
    with pytest.raises(IncidentStorageError):
        JsonIncidentRepository(incidents).list_incidents()
    assert [item.text for item in memory.list_memories()] == ["kept"]


def test_clear_after_recall_drops_active_keeps_persistent(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    record = store.add_memory("recalled later")
    from src.active_memory import ActiveMemoryContext

    active = ActiveMemoryContext()
    active.activate(record)
    history = ConversationHistory()
    history.add_user_message("hi")
    history.add_assistant_message("hello")
    result = handle_slash_command(
        "/clear",
        settings=_settings(),
        conversation_history=history,
        memory_store=store,
        active_memory_context=active,
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )
    assert result.message is not None
    assert "recalled active" in result.message
    assert active.active_count == 0
    assert store.list_memories()[0].id == record.id


def test_ordinary_and_alias_wall_times_match() -> None:
    summer = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-08-14 09:00"),
        "America/New_York",
    )
    winter = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-01-15 09:00"),
        "America/New_York",
    )
    assert summer.startswith("2026-08-14T13:00:00")
    assert winter.startswith("2026-01-15T14:00:00")
    alias = local_wall_to_utc_iso(
        parse_local_wall_datetime("2026-08-14 09:00"),
        "US/Eastern",
    )
    assert alias == summer
    start, end = local_wall_pair_to_utc(
        start_local="2026-08-14 09:00",
        end_local="2026-08-14 10:00",
        timezone_name="America/New_York",
    )
    assert start == summer
    assert end.startswith("2026-08-14T14:00:00")


def test_dst_messages_are_shared() -> None:
    with pytest.raises(ReminderValidationError) as spring:
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-03-08 02:30"),
            "America/New_York",
        )
    with pytest.raises(CalendarValidationError) as calendar_spring:
        local_wall_pair_to_utc(
            start_local="2026-03-08 02:30",
            end_local="2026-03-08 03:30",
            timezone_name="America/New_York",
        )
    assert str(spring.value) == DST_NONEXISTENT_LOCAL_TIME_MESSAGE
    assert str(calendar_spring.value) == DST_NONEXISTENT_LOCAL_TIME_MESSAGE
    with pytest.raises(ReminderValidationError) as fall:
        local_wall_to_utc_iso(
            parse_local_wall_datetime("2026-11-01 01:30"),
            "America/New_York",
        )
    assert str(fall.value) == DST_AMBIGUOUS_LOCAL_TIME_MESSAGE


def test_evidence_metadata_failure_rolls_back_copy(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"evidence-bytes")
    store = LocalEvidenceStore(tmp_path / "evidence")

    class FailingRepo(JsonIncidentRepository):
        def add_evidence(self, evidence: Any, custody_entries: Any) -> Any:
            raise IncidentStorageError("Cortana: Incident repository data could not be updated safely.")

    repo = FailingRepo(tmp_path / "incidents.json")
    result = handle_security_command(
        "evidence-register",
        SecurityCommandContext(
            message=f"/evidence-register {source} | Title | Description",
            incident_repository=repo,
            evidence_store=store,
        ),
    )
    assert result is not None
    assert "could not be updated safely" in result.message
    assert list((tmp_path / "evidence").glob("*.bin")) == []
    assert source.read_bytes() == b"evidence-bytes"
    assert repo.list_evidence() == []


def test_evidence_copy_failure_does_not_write_metadata(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    store = LocalEvidenceStore(tmp_path / "evidence")
    result = handle_security_command(
        "evidence-register",
        SecurityCommandContext(
            message="/evidence-register missing-file.bin | Title | Description",
            incident_repository=repo,
            evidence_store=store,
        ),
    )
    assert result is not None
    assert "could not be found" in result.message
    assert repo.list_evidence() == []


def test_evidence_success_keeps_binary(tmp_path: Path) -> None:
    source = tmp_path / "keep.bin"
    source.write_bytes(b"kept")
    repo = incident_repository(tmp_path)
    store = LocalEvidenceStore(tmp_path / "evidence")
    result = handle_security_command(
        "evidence-register",
        SecurityCommandContext(
            message=f"/evidence-register {source} | Title | Description",
            incident_repository=repo,
            evidence_store=store,
        ),
    )
    assert result is not None
    assert "Evidence registered" in result.message
    evidence_id = repo.list_evidence()[0].evidence_id
    stored = tmp_path / "evidence" / evidence_storage_filename(evidence_id)
    assert stored.exists()
    assert stored.read_bytes() == b"kept"


def test_calendar_commands_report_unavailable_when_deps_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.calendar_commands.optional_calendar_dependencies_available",
        lambda: False,
    )
    result = handle_calendar_command(
        "calendar-connect",
        CalendarCommandContext(message="/calendar-connect", calendar_service=cast(Any, object())),
    )
    assert result is not None
    assert result.message == CALENDAR_UNAVAILABLE_MESSAGE


def test_optional_calendar_probe_function_exists() -> None:
    assert optional_calendar_dependencies_available() in {True, False}


def test_document_search_does_not_leak_internal_exception(tmp_path: Path) -> None:
    class LeakyRetriever(LexicalDocumentRetriever):
        def search_vault(self, *args: object, **kwargs: object) -> list[object]:
            raise DocumentRetrievalError(SECRET_LEAK)

    vault = JsonDocumentVault(tmp_path / "documents.json")
    from src.document import create_document

    vault.add_document(
        create_document(
            filename="note.txt",
            extension=".txt",
            source_size_bytes=5,
            content_hash="a" * 64,
            extracted_text="hello",
        )
    )
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=vault,
        document_retriever=LeakyRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    result = orchestrator.try_handle("search documents for hello")
    assert result is not None
    assert SECRET_LEAK not in result.safe_user_message
    assert "sk-test-123" not in result.safe_user_message
    assert "private.env" not in result.safe_user_message
    assert result.safe_user_message == "Cortana: I couldn't search the document vault."


def test_memory_load_failure_leaves_documents_usable(tmp_path: Path) -> None:
    memory_path = tmp_path / "memories.json"
    memory_path.write_text("{not json", encoding="utf-8")
    docs = JsonDocumentVault(tmp_path / "documents.json")
    with pytest.raises(MemoryStorageError):
        JsonMemoryStore(memory_path).list_memories()
    assert docs.list_documents() == []


def test_valid_memory_at_char_cap_loads(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json")
    store.add_memory("y" * MAX_MEMORY_TEXT_LENGTH)
    reloaded = JsonMemoryStore(tmp_path / "memories.json")
    assert len(reloaded.list_memories()[0].text) == MAX_MEMORY_TEXT_LENGTH
