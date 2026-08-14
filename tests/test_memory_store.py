"""Tests for Project Cortana JSON-backed memory storage."""

import logging
from pathlib import Path

import pytest

from src.config import MAX_MEMORY_TEXT_LENGTH, MAX_STORED_MEMORIES
from src.memory import MemoryTextTooLongError
from src.memory_store import JsonMemoryStore, MemoryCountLimitError, MemoryStorageError


def _store(tmp_path: Path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "memories.json")


def test_missing_file_returns_empty_memories(tmp_path: Path) -> None:
    """A missing memory file should behave like an empty store."""
    store = _store(tmp_path)

    assert store.list_memories() == []
    assert not store.file_path.exists()


def test_save_and_reload_across_store_instances(tmp_path: Path) -> None:
    """Saved memories should reload from disk in a new store instance."""
    first_store = _store(tmp_path)
    saved = first_store.add_memory("Persist this")

    second_store = _store(tmp_path)
    memories = second_store.list_memories()

    assert len(memories) == 1
    assert memories[0].id == saved.id
    assert memories[0].text == "Persist this"
    assert memories[0].created_at == saved.created_at


def test_multiple_memories_preserve_order(tmp_path: Path) -> None:
    """Multiple memories should retain insertion order."""
    store = _store(tmp_path)
    first = store.add_memory("First")
    second = store.add_memory("Second")
    third = store.add_memory("Third")

    assert [memory.id for memory in store.list_memories()] == [
        first.id,
        second.id,
        third.id,
    ]


def test_delete_one_memory(tmp_path: Path) -> None:
    """Deleting one memory should remove only the matching record on disk."""
    store = _store(tmp_path)
    first = store.add_memory("Keep")
    second = store.add_memory("Remove")

    assert store.delete_memory(second.id) is True

    reloaded = _store(tmp_path).list_memories()
    assert len(reloaded) == 1
    assert reloaded[0].id == first.id
    assert reloaded[0].text == "Keep"


def test_delete_missing_memory(tmp_path: Path) -> None:
    """Deleting a missing memory ID should return False without changing disk."""
    store = _store(tmp_path)
    kept = store.add_memory("Only memory")

    assert store.delete_memory("missing-id") is False

    reloaded = _store(tmp_path).list_memories()
    assert len(reloaded) == 1
    assert reloaded[0].id == kept.id
    assert reloaded[0].text == "Only memory"


def test_delete_all_memories(tmp_path: Path) -> None:
    """Deleting all memories should clear the store and persist emptiness."""
    store = _store(tmp_path)
    store.add_memory("One")
    store.add_memory("Two")

    deleted_count = store.delete_all_memories()
    reloaded = _store(tmp_path)

    assert deleted_count == 2
    assert store.list_memories() == []
    assert reloaded.list_memories() == []


def test_utf8_text_round_trip(tmp_path: Path) -> None:
    """UTF-8 memory text should survive save and reload."""
    store = _store(tmp_path)
    text = "Café 安全 🔐"

    store.add_memory(text)
    reloaded = _store(tmp_path).list_memories()

    assert reloaded[0].text == text


def test_atomic_write_replaces_target_file(tmp_path: Path) -> None:
    """Persisted output should be valid complete JSON at the target path."""
    store = _store(tmp_path)
    store.add_memory("Atomic write")

    raw = store.file_path.read_text(encoding="utf-8")
    leftover_temps = list(tmp_path.glob(".memories-*.tmp"))

    assert '"memories"' in raw
    assert "Atomic write" in raw
    assert leftover_temps == []


def test_parent_directory_created_when_needed(tmp_path: Path) -> None:
    """Saving should create missing parent directories."""
    nested_path = tmp_path / "nested" / "app" / "memories.json"
    store = JsonMemoryStore(nested_path)

    store.add_memory("Create parents")

    assert nested_path.exists()
    assert nested_path.parent.is_dir()


def test_malformed_json_does_not_crash_or_overwrite(tmp_path: Path) -> None:
    """Malformed JSON should raise a safe error and preserve the file."""
    path = tmp_path / "memories.json"
    original = "{not-valid-json"
    path.write_text(original, encoding="utf-8")
    store = JsonMemoryStore(path)

    with pytest.raises(MemoryStorageError):
        store.list_memories()

    assert path.read_text(encoding="utf-8") == original


def test_empty_file_handling(tmp_path: Path) -> None:
    """An empty memory file should not be treated as a valid empty store."""
    path = tmp_path / "memories.json"
    path.write_text("", encoding="utf-8")
    store = JsonMemoryStore(path)

    with pytest.raises(MemoryStorageError):
        store.list_memories()

    assert path.read_text(encoding="utf-8") == ""


def test_invalid_top_level_json_structure(tmp_path: Path) -> None:
    """Invalid top-level JSON should raise without overwriting the file."""
    path = tmp_path / "memories.json"
    original = '["not-an-object"]'
    path.write_text(original, encoding="utf-8")
    store = JsonMemoryStore(path)

    with pytest.raises(MemoryStorageError):
        store.list_memories()

    assert path.read_text(encoding="utf-8") == original


def test_malformed_record_handling(tmp_path: Path) -> None:
    """Malformed individual records should fail safely without repair."""
    path = tmp_path / "memories.json"
    original = '{"memories": [{"id": "abc"}]}'
    path.write_text(original, encoding="utf-8")
    store = JsonMemoryStore(path)

    with pytest.raises(MemoryStorageError):
        store.list_memories()

    assert path.read_text(encoding="utf-8") == original


def test_memory_text_is_not_written_to_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Diagnostic logs must not include saved memory text."""
    path = tmp_path / "memories.json"
    secret_text = "super-secret-memory-content"
    path.write_text("{bad-json", encoding="utf-8")
    store = JsonMemoryStore(path)

    with caplog.at_level(logging.ERROR, logger="ProjectCortana"):
        with pytest.raises(MemoryStorageError):
            store.list_memories()

        store_ok = _store(tmp_path / "ok")
        store_ok.add_memory(secret_text)

    combined_logs = " ".join(record.getMessage() for record in caplog.records)
    assert secret_text not in combined_logs
    assert "{bad-json" not in combined_logs


def test_memory_store_enforces_count_cap(tmp_path: Path) -> None:
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path, max_memories=3)
    first = store.add_memory("one")
    store.add_memory("two")
    store.add_memory("three")
    assert len(store.list_memories()) == 3

    with pytest.raises(MemoryCountLimitError) as error:
        store.add_memory("four")
    assert "capacity" in error.value.user_message.casefold()
    assert "3" in error.value.user_message
    assert [memory.text for memory in store.list_memories()] == ["one", "two", "three"]

    reloaded = JsonMemoryStore(path, max_memories=3)
    assert [memory.text for memory in reloaded.list_memories()] == [
        "one",
        "two",
        "three",
    ]

    assert store.delete_memory(first.id) is True
    added = store.add_memory("four")
    assert added.text == "four"
    assert [memory.text for memory in JsonMemoryStore(path, max_memories=3).list_memories()] == [
        "two",
        "three",
        "four",
    ]


def test_memory_store_text_limit_still_applies_at_capacity(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memories.json", max_memories=1)
    store.add_memory("kept")
    with pytest.raises(MemoryTextTooLongError):
        store.add_memory("x" * (MAX_MEMORY_TEXT_LENGTH + 1))
    assert [memory.text for memory in store.list_memories()] == ["kept"]


def test_memory_store_load_rejects_over_capacity_file(tmp_path: Path) -> None:
    writer = JsonMemoryStore(tmp_path / "memories.json", max_memories=3)
    writer.add_memory("a")
    writer.add_memory("b")
    original = (tmp_path / "memories.json").read_text(encoding="utf-8")
    reader = JsonMemoryStore(tmp_path / "memories.json", max_memories=1)
    with pytest.raises(MemoryStorageError):
        reader.list_memories()
    assert (tmp_path / "memories.json").read_text(encoding="utf-8") == original


def test_max_stored_memories_default_is_five_hundred() -> None:
    assert MAX_STORED_MEMORIES == 500
