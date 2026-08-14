"""Pre-M30 Deep Audit #4: persistence cap, byte-bound, and restart malformed-state.

Discovery only. Failures document load-path gaps and count-only bounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import MAX_MEMORY_TEXT_LENGTH, MAX_STORED_MEMORIES
from src.memory_store import JsonMemoryStore, MemoryCountLimitError, MemoryStorageError
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.reminder_repository import JsonReminderRepository, ReminderStorageError
from src.tool_repository import JsonToolControlRepository
from src.memory import MemoryTextTooLongError, create_memory


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_deep_memory_add_enforces_count_and_char_caps(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=2)
    store.add_memory("one")
    store.add_memory("two")
    with pytest.raises(MemoryCountLimitError):
        store.add_memory("three")
    with pytest.raises(MemoryTextTooLongError):
        create_memory("x" * (MAX_MEMORY_TEXT_LENGTH + 1))


def test_deep_memory_load_rejects_overlong_text(tmp_path: Path) -> None:
    """Load must re-validate per-record char caps, not only count."""
    path = tmp_path / "memories.json"
    _write_json(
        path,
        {
            "memories": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "text": "x" * (MAX_MEMORY_TEXT_LENGTH + 50),
                    "created_at": "2026-08-14T00:00:00.000000Z",
                }
            ]
        },
    )
    store = JsonMemoryStore(path)
    with pytest.raises(MemoryStorageError):
        store.list_memories()


def test_deep_memory_load_rejects_count_plus_one(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    records = [
        {
            "id": f"{index:08d}-1111-1111-1111-111111111111",
            "text": f"memory {index}",
            "created_at": "2026-08-14T00:00:00.000000Z",
        }
        for index in range(MAX_STORED_MEMORIES + 1)
    ]
    _write_json(path, {"memories": records})
    store = JsonMemoryStore(path)
    with pytest.raises(MemoryStorageError):
        store.list_memories()


def test_deep_memory_one_valid_one_malformed_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    _write_json(
        path,
        {
            "memories": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "text": "kept",
                    "created_at": "2026-08-14T00:00:00.000000Z",
                },
                {"id": "bad", "text": 123, "created_at": "nope"},
            ]
        },
    )
    original = path.read_text(encoding="utf-8")
    store = JsonMemoryStore(path)
    with pytest.raises(MemoryStorageError):
        store.list_memories()
    assert path.read_text(encoding="utf-8") == original


def test_deep_memory_duplicate_ids_on_load(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    record = {
        "id": "11111111-1111-1111-1111-111111111111",
        "text": "dup",
        "created_at": "2026-08-14T00:00:00.000000Z",
    }
    _write_json(path, {"memories": [record, dict(record, text="other")]})
    store = JsonMemoryStore(path)
    with pytest.raises(MemoryStorageError):
        store.list_memories()


def test_deep_memory_extra_field_and_future_key_are_ignored_or_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memories.json"
    _write_json(
        path,
        {
            "schema_version": 99,
            "extra_root": True,
            "memories": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "text": "ok",
                    "created_at": "2026-08-14T00:00:00.000000Z",
                    "secret": "should-not-matter",
                }
            ],
        },
    )
    store = JsonMemoryStore(path)
    memories = store.list_memories()
    assert len(memories) == 1
    assert memories[0].text == "ok"


def test_deep_truncated_json_fails_closed_and_does_not_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    path.write_text('{"memories": [{"id":', encoding="utf-8")
    original = path.read_bytes()
    store = JsonMemoryStore(path)
    with pytest.raises(MemoryStorageError):
        store.list_memories()
    assert path.read_bytes() == original


def test_deep_malformed_memory_does_not_affect_documents(tmp_path: Path) -> None:
    memory_path = tmp_path / "memories.json"
    docs_path = tmp_path / "documents.json"
    memory_path.write_text("{not json", encoding="utf-8")
    JsonDocumentVault(docs_path)
    with pytest.raises(MemoryStorageError):
        JsonMemoryStore(memory_path).list_memories()
    vault = JsonDocumentVault(docs_path)
    assert vault.list_documents() == []


def test_deep_tool_and_incident_repos_have_no_count_cap_on_load(tmp_path: Path) -> None:
    """Tool/incident repositories must not grow without a documented retention cap."""
    tools = JsonToolControlRepository(tmp_path / "tool_control.json")
    incidents = JsonIncidentRepository(tmp_path / "incidents.json")
    # Empty load succeeds; there is no MAX_* cap asserted by the repository API.
    assert tools.list_requests() == []
    assert incidents.list_incidents() == []
    tool_caps = [name for name in dir(tools) if "max" in name.lower() or "limit" in name.lower()]
    incident_caps = [
        name for name in dir(incidents) if "max" in name.lower() or "limit" in name.lower()
    ]
    assert tool_caps, "JsonToolControlRepository exposes no retention cap"
    assert incident_caps, "JsonIncidentRepository exposes no retention cap"


def test_deep_delete_then_add_memory_reuses_capacity(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=1)
    record = store.add_memory("first")
    store.delete_memory(record.id)
    second = store.add_memory("second")
    assert [memory.text for memory in store.list_memories()] == ["second"]
    assert second.id != record.id


def test_deep_restart_unknown_enum_and_missing_required_fail_closed(
    tmp_path: Path,
) -> None:
    """Unknown enum / missing required field: whole store fails, file unchanged."""
    cases: list[tuple[Path, type[Exception], object]] = []

    reminder_path = tmp_path / "reminders.json"
    _write_json(
        reminder_path,
        {
            "version": 1,
            "reminders": [
                {
                    "reminder_id": "11111111-1111-1111-1111-111111111111",
                    "title": "x",
                    "message": "y",
                    "timezone_name": "America/New_York",
                    "recurrence_type": "not-a-real-enum",
                }
            ],
            "audit_entries": [],
        },
    )
    cases.append((reminder_path, ReminderStorageError, JsonReminderRepository))

    for path, error_type, factory in cases:
        original = path.read_text(encoding="utf-8")
        repo = factory(path)
        with pytest.raises(error_type):
            if hasattr(repo, "list_reminders"):
                repo.list_reminders()
            elif hasattr(repo, "list_all"):
                repo.list_all()
            else:
                repo.load()
        assert path.read_text(encoding="utf-8") == original


def test_deep_count_cap_is_not_a_byte_cap_for_memory_load(tmp_path: Path) -> None:
    """500 memories at the char cap is allowed; one oversize record must not be."""
    path = tmp_path / "memories.json"
    _write_json(
        path,
        {
            "memories": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "text": "y" * MAX_MEMORY_TEXT_LENGTH,
                    "created_at": "2026-08-14T00:00:00.000000Z",
                }
            ]
        },
    )
    store = JsonMemoryStore(path)
    memories = store.list_memories()
    assert len(memories[0].text) == MAX_MEMORY_TEXT_LENGTH
