"""Session-only active memory context for temporary AI reference use."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from src.config import MAX_ACTIVE_MEMORIES, MAX_ACTIVE_MEMORY_CHARS
from src.memory import MemoryRecord

_BOUNDARY_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_BOUNDARY_TOKEN_BYTES = 16


class ActiveMemoryError(ValueError):
    """Base error for active-memory session operations."""


class DuplicateActiveMemoryError(ActiveMemoryError):
    """Raised when a memory is already active in the session."""


class ActiveMemoryCountLimitError(ActiveMemoryError):
    """Raised when activating would exceed the active-memory count limit."""


class ActiveMemoryCharLimitError(ActiveMemoryError):
    """Raised when activating would exceed the active-memory character limit."""


class InvalidBoundaryTokenError(ActiveMemoryError):
    """Raised when an active-memory boundary token is unsafe or malformed."""


def generate_boundary_token() -> str:
    """Generate an unpredictable boundary token safe for text markers."""
    return secrets.token_hex(_BOUNDARY_TOKEN_BYTES)


def validate_boundary_token(token: str) -> str:
    """Validate and return a boundary token safe for text markers."""
    if not _BOUNDARY_TOKEN_PATTERN.fullmatch(token):
        raise InvalidBoundaryTokenError(
            "Active-memory boundary token must be 16-64 characters of "
            "letters, digits, underscore, or hyphen."
        )
    return token


def normalize_memory_id(memory_id: str) -> str:
    """Return a consistent comparison key for memory IDs."""
    return memory_id.strip().casefold()


@dataclass
class ActiveMemoryContext:
    """Track which saved memories are active for the current session only."""

    boundary_token: str = field(default="")
    _active: list[MemoryRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize or validate the session-only boundary token."""
        if self.boundary_token:
            self.boundary_token = validate_boundary_token(self.boundary_token)
        else:
            self.boundary_token = generate_boundary_token()

    def activate(self, record: MemoryRecord) -> None:
        """Activate one saved memory for temporary AI context use."""
        record_key = normalize_memory_id(record.id)
        if any(normalize_memory_id(memory.id) == record_key for memory in self._active):
            raise DuplicateActiveMemoryError(
                f"Memory '{record.id}' is already active."
            )

        if len(self._active) >= MAX_ACTIVE_MEMORIES:
            raise ActiveMemoryCountLimitError(
                f"Active memory count limit of {MAX_ACTIVE_MEMORIES} reached."
            )

        projected_characters = self.total_character_usage + len(record.text)
        if projected_characters > MAX_ACTIVE_MEMORY_CHARS:
            raise ActiveMemoryCharLimitError(
                f"Active memory character limit of {MAX_ACTIVE_MEMORY_CHARS} reached."
            )

        self._active.append(record)

    def deactivate(self, memory_id: str) -> bool:
        """Remove one memory from active session context. Return True if removed."""
        target_key = normalize_memory_id(memory_id)
        if not target_key:
            return False

        for index, memory in enumerate(self._active):
            if normalize_memory_id(memory.id) == target_key:
                del self._active[index]
                return True
        return False

    def clear(self) -> int:
        """Clear all active memories and return how many were removed."""
        cleared_count = len(self._active)
        self._active.clear()
        return cleared_count

    def list_active(self) -> list[MemoryRecord]:
        """Return active memories in activation order."""
        return list(self._active)

    def list_active_ids(self) -> list[str]:
        """Return active memory IDs in activation order."""
        return [memory.id for memory in self._active]

    @property
    def active_count(self) -> int:
        """Return the number of currently active memories."""
        return len(self._active)

    @property
    def total_character_usage(self) -> int:
        """Return combined character length of all active memory text."""
        return sum(len(memory.text) for memory in self._active)
