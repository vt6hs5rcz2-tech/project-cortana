"""Tests for session-only active memory context."""

from pathlib import Path

import pytest

from src.active_memory import (
    ActiveMemoryCharLimitError,
    ActiveMemoryContext,
    ActiveMemoryCountLimitError,
    DuplicateActiveMemoryError,
    InvalidBoundaryTokenError,
)
from src.config import MAX_ACTIVE_MEMORIES, MAX_ACTIVE_MEMORY_CHARS
from src.memory import MemoryRecord, create_memory
from src.memory_context import format_active_memory_context, outer_boundary_start

DETERMINISTIC_BOUNDARY_TOKEN = "test_session_token_01"


def _record(text: str, memory_id: str = "memory-1") -> MemoryRecord:
    return MemoryRecord(id=memory_id, text=text, created_at="2026-01-01T00:00:00Z")


def test_independent_sessions_receive_different_default_boundary_tokens() -> None:
    """Each new ActiveMemoryContext should generate a distinct default token."""
    first = ActiveMemoryContext()
    second = ActiveMemoryContext()

    assert first.boundary_token != second.boundary_token
    assert len(first.boundary_token) >= 16
    assert len(second.boundary_token) >= 16


def test_boundary_token_can_be_injected_for_deterministic_tests() -> None:
    """Tests may inject a validated boundary token."""
    context = ActiveMemoryContext(boundary_token=DETERMINISTIC_BOUNDARY_TOKEN)

    assert context.boundary_token == DETERMINISTIC_BOUNDARY_TOKEN


def test_session_reuses_boundary_token_consistently() -> None:
    """A session should keep and reuse one boundary token across activations."""
    context = ActiveMemoryContext(boundary_token=DETERMINISTIC_BOUNDARY_TOKEN)
    original_token = context.boundary_token

    context.activate(_record("One", "1"))
    context.activate(_record("Two", "2"))
    context.deactivate("1")
    context.clear()

    assert context.boundary_token == original_token
    assert context.boundary_token == DETERMINISTIC_BOUNDARY_TOKEN


def test_active_memory_formatting_uses_session_boundary_token() -> None:
    """Formatting helpers should embed the session token in boundary markers."""
    context = ActiveMemoryContext(boundary_token=DETERMINISTIC_BOUNDARY_TOKEN)
    record = _record("Exact memory text", "mem-1")
    context.activate(record)

    formatted = format_active_memory_context(
        context.list_active(),
        boundary_token=context.boundary_token,
    )

    assert outer_boundary_start(DETERMINISTIC_BOUNDARY_TOKEN) in formatted
    assert "Exact memory text" in formatted
    assert formatted.count(DETERMINISTIC_BOUNDARY_TOKEN) >= 2


def test_invalid_boundary_token_is_rejected() -> None:
    """Unsafe or malformed boundary tokens must be rejected."""
    with pytest.raises(InvalidBoundaryTokenError):
        ActiveMemoryContext(boundary_token="bad token with spaces!")


def test_activate_one_memory() -> None:
    """Activating a memory should make it available in session context."""
    context = ActiveMemoryContext()
    record = _record("Authorized network range")

    context.activate(record)

    assert context.list_active() == [record]
    assert context.list_active_ids() == [record.id]
    assert context.active_count == 1


def test_preserve_activation_order() -> None:
    """Active memories should remain in activation order."""
    context = ActiveMemoryContext()
    first = _record("First", "id-1")
    second = _record("Second", "id-2")
    third = _record("Third", "id-3")

    context.activate(first)
    context.activate(second)
    context.activate(third)

    assert context.list_active_ids() == ["id-1", "id-2", "id-3"]


def test_reject_duplicate_activation() -> None:
    """The same memory ID cannot be activated twice."""
    context = ActiveMemoryContext()
    record = _record("Duplicate candidate", "dup-id")
    context.activate(record)

    with pytest.raises(DuplicateActiveMemoryError):
        context.activate(_record("Different text", "DUP-ID"))

    assert context.active_count == 1
    assert context.list_active()[0].text == "Duplicate candidate"


def test_deactivate_one_memory() -> None:
    """Deactivating should remove only the selected active memory."""
    context = ActiveMemoryContext()
    first = _record("First", "id-1")
    second = _record("Second", "id-2")
    context.activate(first)
    context.activate(second)

    assert context.deactivate("id-1") is True
    assert context.list_active_ids() == ["id-2"]


def test_deactivate_missing_memory() -> None:
    """Deactivating an inactive ID should return False without changes."""
    context = ActiveMemoryContext()
    context.activate(_record("Present", "present"))

    assert context.deactivate("missing") is False
    assert context.active_count == 1


def test_clear_all_active_memories() -> None:
    """Clearing should remove every active memory and report the count."""
    context = ActiveMemoryContext()
    context.activate(_record("One", "1"))
    context.activate(_record("Two", "2"))

    cleared = context.clear()

    assert cleared == 2
    assert context.active_count == 0
    assert context.list_active() == []


def test_clear_when_already_empty() -> None:
    """Clearing an empty active context should report zero removals."""
    context = ActiveMemoryContext()

    assert context.clear() == 0
    assert context.active_count == 0


def test_active_count_and_character_usage() -> None:
    """Count and character usage should reflect currently active memories."""
    context = ActiveMemoryContext()
    context.activate(_record("abcd", "1"))
    context.activate(_record("efg", "2"))

    assert context.active_count == 2
    assert context.total_character_usage == 7


def test_count_limit_enforcement() -> None:
    """Activation beyond the count limit should be rejected without mutation."""
    context = ActiveMemoryContext()
    for index in range(MAX_ACTIVE_MEMORIES):
        context.activate(_record(f"Memory {index}", f"id-{index}"))

    with pytest.raises(ActiveMemoryCountLimitError):
        context.activate(_record("Overflow", "overflow"))

    assert context.active_count == MAX_ACTIVE_MEMORIES
    assert "overflow" not in context.list_active_ids()


def test_character_limit_enforcement_without_silent_truncation() -> None:
    """Character-limit rejection must leave existing active text unchanged."""
    context = ActiveMemoryContext()
    first_text = "a" * (MAX_ACTIVE_MEMORY_CHARS - 10)
    context.activate(_record(first_text, "small"))

    with pytest.raises(ActiveMemoryCharLimitError):
        context.activate(_record("b" * 20, "too-large"))

    assert context.active_count == 1
    assert context.list_active()[0].text == first_text
    assert len(context.list_active()[0].text) == MAX_ACTIVE_MEMORY_CHARS - 10


def test_session_state_is_not_written_to_disk(tmp_path: Path) -> None:
    """Active-memory selections must remain in-memory only."""
    context = ActiveMemoryContext()
    context.activate(create_memory("Session only memory"))

    created_paths = list(tmp_path.rglob("*"))

    assert created_paths == []
    assert context.active_count == 1
