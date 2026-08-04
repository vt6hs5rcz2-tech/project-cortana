"""Explicit persistent memory model for Project Cortana."""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from src.config import MAX_MEMORY_TEXT_LENGTH


class MemoryValidationError(ValueError):
    """Raised when memory text fails local validation."""


class BlankMemoryTextError(MemoryValidationError):
    """Raised when memory text is blank after trimming."""


class MemoryTextTooLongError(MemoryValidationError):
    """Raised when memory text exceeds the configured maximum length."""


@dataclass(frozen=True)
class MemoryRecord:
    """Immutable explicit memory saved by the user."""

    id: str
    text: str
    created_at: str


def _utc_timestamp() -> str:
    """Return the current UTC time in unambiguous ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def create_memory(text: str) -> MemoryRecord:
    """Create a validated immutable memory record from user-provided text."""
    cleaned_text = text.strip()

    if not cleaned_text:
        raise BlankMemoryTextError("Memory text cannot be blank.")

    if len(cleaned_text) > MAX_MEMORY_TEXT_LENGTH:
        raise MemoryTextTooLongError(
            f"Memory text exceeds the maximum length of {MAX_MEMORY_TEXT_LENGTH} characters."
        )

    return MemoryRecord(
        id=str(uuid4()),
        text=cleaned_text,
        created_at=_utc_timestamp(),
    )
