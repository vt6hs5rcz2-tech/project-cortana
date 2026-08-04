"""Tests for Project Cortana explicit memory model validation."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from src.config import MAX_MEMORY_TEXT_LENGTH
from src.memory import (
    BlankMemoryTextError,
    MemoryRecord,
    MemoryTextTooLongError,
    MemoryValidationError,
    create_memory,
)


def test_create_memory_generates_stable_unique_ids() -> None:
    """Each memory should receive a distinct application-generated ID."""
    first = create_memory("First memory")
    second = create_memory("Second memory")

    assert first.id != second.id
    assert len(first.id) > 0
    assert len(second.id) > 0


def test_create_memory_uses_utc_iso8601_timestamp() -> None:
    """Creation timestamps should be unambiguous UTC ISO 8601 values."""
    record = create_memory("Timestamped memory")

    assert record.created_at.endswith("Z")
    parsed = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    utc_offset = parsed.utcoffset()
    assert utc_offset is not None
    assert utc_offset.total_seconds() == 0


def test_create_memory_trims_surrounding_whitespace() -> None:
    """Surrounding whitespace should be trimmed before saving."""
    record = create_memory("  Keep this text  \n")

    assert record.text == "Keep this text"


def test_create_memory_rejects_blank_text() -> None:
    """Blank or whitespace-only memory text should be rejected."""
    with pytest.raises(BlankMemoryTextError) as error_info:
        create_memory("   ")

    assert isinstance(error_info.value, MemoryValidationError)


def test_create_memory_enforces_maximum_length() -> None:
    """Oversized memory text should be rejected using the centralized limit."""
    oversized = "a" * (MAX_MEMORY_TEXT_LENGTH + 1)

    with pytest.raises(MemoryTextTooLongError) as error_info:
        create_memory(oversized)

    assert isinstance(error_info.value, MemoryValidationError)


def test_create_memory_accepts_maximum_length() -> None:
    """Text at the maximum length should be accepted."""
    record = create_memory("a" * MAX_MEMORY_TEXT_LENGTH)

    assert len(record.text) == MAX_MEMORY_TEXT_LENGTH


def test_memory_record_is_immutable() -> None:
    """Memory records should reject attribute mutation."""
    record = create_memory("Immutable memory")

    with pytest.raises(FrozenInstanceError):
        record.text = "Changed"  # type: ignore[misc]

    assert isinstance(record, MemoryRecord)
