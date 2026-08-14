"""Persistent memory storage interfaces and local JSON implementation."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Protocol

from src.config import MAX_MEMORY_TEXT_LENGTH, MAX_STORED_MEMORIES
from src.memory import MemoryRecord, create_memory

logger = logging.getLogger("ProjectCortana")

MEMORY_LOAD_ERROR_MESSAGE = (
    "Cortana: Saved memory data could not be loaded safely. "
    "The existing memory file was left unchanged for inspection."
)


class MemoryStorageError(RuntimeError):
    """Raised when persistent memory data cannot be loaded or saved safely."""

    def __init__(self, message: str = MEMORY_LOAD_ERROR_MESSAGE) -> None:
        super().__init__(message)
        self.user_message = message


class MemoryCountLimitError(MemoryStorageError):
    """Raised when adding a memory would exceed the configured capacity."""

    def __init__(self, *, max_memories: int = MAX_STORED_MEMORIES) -> None:
        super().__init__(
            "Cortana: Memory capacity reached. "
            f"A maximum of {max_memories} memories can be stored. "
            "Delete an existing memory before adding another."
        )


class MemoryStore(Protocol):
    """Protocol for explicit persistent memory storage."""

    def list_memories(self) -> list[MemoryRecord]:
        """Return saved memories in storage order."""

    def add_memory(self, text: str) -> MemoryRecord:
        """Validate, persist, and return one new memory."""

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one memory by ID. Return True when a record was removed."""

    def delete_all_memories(self) -> int:
        """Delete all memories and return how many were removed."""


class JsonMemoryStore:
    """Local JSON-backed store for explicit persistent memories."""

    def __init__(
        self,
        file_path: Path,
        *,
        max_memories: int = MAX_STORED_MEMORIES,
    ) -> None:
        if max_memories < 1:
            raise ValueError("max_memories must be at least 1.")
        self._file_path = file_path
        self._max_memories = max_memories
        self._memories: list[MemoryRecord] | None = None
        self._load_error: MemoryStorageError | None = None

    @property
    def file_path(self) -> Path:
        """Return the configured memory file path."""
        return self._file_path

    def list_memories(self) -> list[MemoryRecord]:
        """Return a copy of saved memories in storage order."""
        return list(self._ensure_loaded())

    def add_memory(self, text: str) -> MemoryRecord:
        """Validate, append, and persist one new memory."""
        memories = self._ensure_loaded()
        record = create_memory(text)
        if len(memories) >= self._max_memories:
            raise MemoryCountLimitError(max_memories=self._max_memories)
        memories.append(record)
        self._persist(memories)
        return record

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one memory by ID and persist the result."""
        memories = self._ensure_loaded()
        remaining = [memory for memory in memories if memory.id != memory_id]
        if len(remaining) == len(memories):
            return False

        self._persist(remaining)
        return True

    def delete_all_memories(self) -> int:
        """Delete every saved memory and persist an empty collection."""
        memories = self._ensure_loaded()
        deleted_count = len(memories)
        if deleted_count == 0:
            return 0

        self._persist([])
        return deleted_count

    def _ensure_loaded(self) -> list[MemoryRecord]:
        """Load memories once, preserving any prior corruption error."""
        if self._load_error is not None:
            raise self._load_error

        if self._memories is None:
            try:
                self._memories = self._load_from_disk()
            except MemoryStorageError as error:
                self._load_error = error
                raise

        return self._memories

    def _load_from_disk(self) -> list[MemoryRecord]:
        """Load and validate memory records from disk without modifying the file."""
        if not self._file_path.exists():
            return []

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
        except OSError as error:
            logger.error(
                "Unable to read memory file due to OS error type: %s",
                type(error).__name__,
            )
            raise MemoryStorageError from error

        if raw_text.strip() == "":
            logger.error("Memory file is empty and cannot be loaded safely.")
            raise MemoryStorageError

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("Memory file contains malformed JSON.")
            raise MemoryStorageError from None

        return self._parse_payload(payload)

    def _parse_payload(self, payload: object) -> list[MemoryRecord]:
        """Validate top-level JSON structure and convert records."""
        if not isinstance(payload, dict):
            logger.error("Memory file top-level JSON structure is invalid.")
            raise MemoryStorageError

        memories_payload = payload.get("memories")
        if not isinstance(memories_payload, list):
            logger.error("Memory file is missing a valid memories list.")
            raise MemoryStorageError

        records: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for item in memories_payload:
            record = self._parse_record(item)
            if record.id in seen_ids:
                logger.error("Memory file contains duplicate memory IDs.")
                raise MemoryStorageError
            seen_ids.add(record.id)
            records.append(record)
        if len(records) > self._max_memories:
            logger.error(
                "Memory file exceeds configured memory count limit count=%s",
                len(records),
            )
            raise MemoryStorageError
        return records

    def _parse_record(self, item: object) -> MemoryRecord:
        """Validate and convert one memory record from JSON."""
        if not isinstance(item, dict):
            logger.error("Memory file contains a malformed memory record.")
            raise MemoryStorageError

        memory_id = item.get("id")
        text = item.get("text")
        created_at = item.get("created_at")

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
            or not isinstance(text, str)
            or not isinstance(created_at, str)
            or not created_at.strip()
        ):
            logger.error("Memory file contains a malformed memory record.")
            raise MemoryStorageError

        if len(text) > MAX_MEMORY_TEXT_LENGTH:
            logger.error("Memory file contains an overlong memory text record.")
            raise MemoryStorageError

        return MemoryRecord(
            id=memory_id,
            text=text,
            created_at=created_at,
        )

    def _persist(self, memories: list[MemoryRecord]) -> None:
        """Atomically write the current memory collection as UTF-8 JSON."""
        payload = {
            "memories": [
                {
                    "id": memory.id,
                    "text": memory.text,
                    "created_at": memory.created_at,
                }
                for memory in memories
            ]
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._memories = list(memories)

    def _atomic_write(self, content: str) -> None:
        """Write JSON to a temporary file, then replace the target file."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".memories-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, self._file_path)
            temporary_path = None
        except OSError as error:
            logger.error(
                "Unable to write memory file due to OS error type: %s",
                type(error).__name__,
            )
            raise MemoryStorageError(
                "Cortana: Saved memory data could not be updated safely."
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
