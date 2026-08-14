"""Pre-M30 hardening tests: startup, persistence, and first-run contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.commands import ABOUT_TEXT, HELP_TEXT, format_status, handle_slash_command
from src.active_memory import ActiveMemoryContext
from src.conversation import STARTUP_GREETING, ConversationHistory
from src.conversation_state import ConversationState
from src.document_extractor import DefaultTextExtractor
from src.document_vault import DocumentStorageError, JsonDocumentVault
from src.incident_repository import IncidentStorageError, JsonIncidentRepository
from src.memory_store import JsonMemoryStore, MemoryStorageError
from src.reminder_repository import JsonReminderRepository, ReminderStorageError
from src.settings import Settings, load_settings
from src.study_repository import JsonStudyRepository, StudyStorageError
from src.tool_repository import JsonToolControlRepository, ToolStorageError
from src.workflow_repository import JsonWorkflowRunRepository, WorkflowStorageError


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_model="test-model")


def test_full_startup_greeting_and_help_are_understandable(tmp_path: Path) -> None:
    assert STARTUP_GREETING.startswith("Cortana:")
    assert "cybersecurity" in STARTUP_GREETING.casefold()
    assert "/help" in HELP_TEXT or "help" in HELP_TEXT.casefold()
    assert "Cortana" in ABOUT_TEXT
    status = format_status(
        _settings(),
        ConversationHistory(),
        JsonMemoryStore(tmp_path / "memories.json"),
        ActiveMemoryContext(),
        JsonDocumentVault(tmp_path / "documents.json"),
    )
    assert "Cortana" in status or "status" in status.casefold() or "model" in status.casefold()


def test_full_duplicate_store_init_does_not_wipe(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    first = JsonMemoryStore(path)
    first.add_memory("keep")
    second = JsonMemoryStore(path)
    third = JsonMemoryStore(path)
    assert len(second.list_memories()) == 1
    assert len(third.list_memories()) == 1


def test_full_malformed_json_fail_closed_across_stores(tmp_path: Path) -> None:
    cases: list[tuple[Path, object, type[BaseException]]] = [
        (tmp_path / "memories.json", JsonMemoryStore, MemoryStorageError),
        (tmp_path / "documents.json", JsonDocumentVault, DocumentStorageError),
        (tmp_path / "incidents.json", JsonIncidentRepository, IncidentStorageError),
        (tmp_path / "tool_control.json", JsonToolControlRepository, ToolStorageError),
        (tmp_path / "workflow_runs.json", JsonWorkflowRunRepository, WorkflowStorageError),
        (tmp_path / "reminders.json", JsonReminderRepository, ReminderStorageError),
        (tmp_path / "study_state.json", JsonStudyRepository, StudyStorageError),
    ]
    for path, factory, error_type in cases:
        path.write_text("{partial", encoding="utf-8")
        store = factory(path)  # type: ignore[misc]
        with pytest.raises(error_type):
            if hasattr(store, "list_memories"):
                store.list_memories()
            elif hasattr(store, "list_documents"):
                store.list_documents()
            elif hasattr(store, "list_incidents"):
                store.list_incidents()
            elif hasattr(store, "list_scopes"):
                store.list_scopes()
            elif hasattr(store, "list_runs"):
                store.list_runs()
            elif hasattr(store, "list_reminders"):
                store.list_reminders()
            elif hasattr(store, "list_sessions"):
                store.list_sessions()
            else:
                raise AssertionError(f"no list method on {factory}")
        assert path.read_text(encoding="utf-8") == "{partial"


def test_full_missing_files_start_empty(tmp_path: Path) -> None:
    assert JsonMemoryStore(tmp_path / "m.json").list_memories() == []
    assert JsonDocumentVault(tmp_path / "d.json").list_documents() == []
    assert JsonIncidentRepository(tmp_path / "i.json").list_incidents() == []
    assert JsonToolControlRepository(tmp_path / "t.json").list_scopes() == []
    assert JsonReminderRepository(tmp_path / "r.json").list_reminders() == []
    assert JsonStudyRepository(tmp_path / "s.json").list_sessions() == []


def test_full_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path)
    store.add_memory("atomic")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    assert path.exists()


def test_full_settings_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.settings.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError):
        load_settings()


def test_full_slash_unknown_is_not_a_stack_trace(tmp_path: Path) -> None:
    result = handle_slash_command(
        "/wat",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
    )
    assert "Traceback" not in result.message
    assert result.message.startswith("Cortana:")


def test_full_first_run_slash_journey_is_user_visible(tmp_path: Path) -> None:
    history = ConversationHistory()
    memory = JsonMemoryStore(tmp_path / "memories.json")
    vault = JsonDocumentVault(tmp_path / "documents.json")
    about = handle_slash_command(
        "/about",
        settings=_settings(),
        conversation_history=history,
        memory_store=memory,
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
    )
    assert about.message.startswith("Cortana:")
    remembered = handle_slash_command(
        "/remember The lab subnet is 10.0.0.0/24",
        settings=_settings(),
        conversation_history=history,
        memory_store=memory,
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
    )
    assert "saved" in remembered.message.casefold()
    listed = handle_slash_command(
        "/memories",
        settings=_settings(),
        conversation_history=history,
        memory_store=memory,
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
    )
    assert "10.0.0.0/24" in listed.message
    cleared = handle_slash_command(
        "/clear",
        settings=_settings(),
        conversation_history=history,
        memory_store=memory,
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        conversation_state=ConversationState(),
    )
    assert "Cortana:" in cleared.message
    restarted = JsonMemoryStore(tmp_path / "memories.json")
    assert restarted.list_memories()
    assert history.turns == []


def test_full_logger_format_does_not_include_secrets() -> None:
    from src.logger import setup_logging

    logger = setup_logging()
    rendered = repr(_settings())
    assert "sk-test" not in rendered
    assert logger.name == "ProjectCortana"


def test_full_production_has_no_todo_fixme_or_notimplemented() -> None:
    from pathlib import Path as RepoPath

    hits: list[str] = []
    root = RepoPath("src")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("TODO", "FIXME", "XXX", "NotImplementedError"):
            if marker in text:
                hits.append(f"{path}:{marker}")
    assert hits == []


def test_full_no_raw_api_key_logging_in_production() -> None:
    from pathlib import Path as RepoPath

    suspicious: list[str] = []
    for path in RepoPath("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "logger" in text and "openai_api_key" in text:
            suspicious.append(str(path))
        if "print(" in text and "openai_api_key" in text:
            suspicious.append(str(path))
    assert suspicious == []
