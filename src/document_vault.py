"""Knowledge Vault storage interfaces and local JSON implementation."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Protocol

from src.config import MAX_STORED_DOCUMENTS
from src.document import (
    DocumentRecord,
    DocumentValidationError,
    validate_document_record,
)

logger = logging.getLogger("ProjectCortana")

DOCUMENT_LOAD_ERROR_MESSAGE = (
    "Cortana: Knowledge Vault data could not be loaded safely. "
    "The existing document file was left unchanged for inspection."
)
DOCUMENT_SAVE_ERROR_MESSAGE = (
    "Cortana: Knowledge Vault data could not be updated safely."
)
DOCUMENT_COUNT_LIMIT_MESSAGE = (
    "Cortana: Knowledge Vault capacity reached. "
    f"A maximum of {MAX_STORED_DOCUMENTS} documents can be stored."
)
DUPLICATE_DOCUMENT_ID_MESSAGE = (
    "Cortana: Knowledge Vault data could not be loaded safely. "
    "The existing document file was left unchanged for inspection."
)


class DocumentStorageError(RuntimeError):
    """Raised when Knowledge Vault data cannot be loaded or saved safely."""

    def __init__(self, message: str = DOCUMENT_LOAD_ERROR_MESSAGE) -> None:
        super().__init__(message)
        self.user_message = message


class DocumentCountLimitError(DocumentStorageError):
    """Raised when adding a document would exceed the configured capacity."""

    def __init__(self) -> None:
        super().__init__(DOCUMENT_COUNT_LIMIT_MESSAGE)


class DuplicateDocumentHashError(DocumentStorageError):
    """Raised when a document with the same content hash already exists."""

    def __init__(self, existing_document_id: str) -> None:
        self.existing_document_id = existing_document_id
        super().__init__(
            "Cortana: A document with identical content already exists "
            f"({existing_document_id})."
        )


class DocumentVault(Protocol):
    """Protocol for Knowledge Vault document storage."""

    def list_documents(self) -> list[DocumentRecord]:
        """Return stored documents in storage order."""

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document by ID, or None when not found."""

    def add_document(self, record: DocumentRecord) -> DocumentRecord:
        """Persist one validated document record."""

    def find_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        """Return the document with the given content hash, if any."""

    def delete_document(self, document_id: str) -> bool:
        """Delete one document by ID. Return True when a record was removed."""

    def delete_all_documents(self) -> int:
        """Delete all documents and return how many were removed."""

    def document_count(self) -> int:
        """Return the number of stored documents."""


class JsonDocumentVault:
    """Local JSON-backed Knowledge Vault for document metadata and extracted text."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._documents: list[DocumentRecord] | None = None
        self._load_error: DocumentStorageError | None = None

    @property
    def file_path(self) -> Path:
        """Return the configured vault file path."""
        return self._file_path

    def list_documents(self) -> list[DocumentRecord]:
        """Return a copy of stored documents in storage order."""
        return list(self._ensure_loaded())

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document by exact ID match."""
        for document in self._ensure_loaded():
            if document.id == document_id:
                return document
        return None

    def add_document(self, record: DocumentRecord) -> DocumentRecord:
        """Validate capacity and uniqueness, then persist one document."""
        documents = self._ensure_loaded()
        validated = validate_document_record(record)

        existing = self.find_by_content_hash(validated.content_hash)
        if existing is not None:
            raise DuplicateDocumentHashError(existing.id)

        if any(document.id == validated.id for document in documents):
            logger.error("Document add rejected duplicate id.")
            raise DocumentStorageError(DUPLICATE_DOCUMENT_ID_MESSAGE)

        if len(documents) >= MAX_STORED_DOCUMENTS:
            raise DocumentCountLimitError()

        documents.append(validated)
        self._persist(documents)
        return validated

    def find_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        """Return the first document matching the content hash."""
        for document in self._ensure_loaded():
            if document.content_hash == content_hash:
                return document
        return None

    def delete_document(self, document_id: str) -> bool:
        """Delete one document by ID and persist the result."""
        documents = self._ensure_loaded()
        remaining = [
            document for document in documents if document.id != document_id
        ]
        if len(remaining) == len(documents):
            return False

        self._persist(remaining)
        return True

    def delete_all_documents(self) -> int:
        """Delete every stored document and persist an empty collection."""
        documents = self._ensure_loaded()
        deleted_count = len(documents)
        if deleted_count == 0:
            return 0

        self._persist([])
        return deleted_count

    def document_count(self) -> int:
        """Return how many documents are currently stored."""
        return len(self._ensure_loaded())

    def _ensure_loaded(self) -> list[DocumentRecord]:
        """Load documents once, preserving any prior corruption error."""
        if self._load_error is not None:
            raise self._load_error

        if self._documents is None:
            try:
                self._documents = self._load_from_disk()
            except DocumentStorageError as error:
                self._load_error = error
                raise

        return self._documents

    def _load_from_disk(self) -> list[DocumentRecord]:
        """Load and validate document records without modifying the vault file."""
        if not self._file_path.exists():
            return []

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
        except OSError as error:
            logger.error(
                "Unable to read document vault due to OS error type: %s",
                type(error).__name__,
            )
            raise DocumentStorageError from error

        if raw_text.strip() == "":
            logger.error("Document vault file is empty and cannot be loaded safely.")
            raise DocumentStorageError

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("Document vault contains malformed JSON.")
            raise DocumentStorageError from None

        return self._parse_payload(payload)

    def _parse_payload(self, payload: object) -> list[DocumentRecord]:
        """Validate top-level JSON structure and convert records."""
        if not isinstance(payload, dict):
            logger.error("Document vault top-level JSON structure is invalid.")
            raise DocumentStorageError

        documents_payload = payload.get("documents")
        if not isinstance(documents_payload, list):
            logger.error("Document vault is missing a valid documents list.")
            raise DocumentStorageError

        if len(documents_payload) > MAX_STORED_DOCUMENTS:
            logger.error(
                "Document vault exceeds configured document count limit count=%s",
                len(documents_payload),
            )
            raise DocumentStorageError

        records: list[DocumentRecord] = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()

        for item in documents_payload:
            record = self._parse_record(item)
            if record.id in seen_ids:
                logger.error("Document vault contains duplicate document IDs.")
                raise DocumentStorageError
            if record.content_hash in seen_hashes:
                logger.error("Document vault contains duplicate content hashes.")
                raise DocumentStorageError
            seen_ids.add(record.id)
            seen_hashes.add(record.content_hash)
            records.append(record)

        return records

    def _parse_record(self, item: object) -> DocumentRecord:
        """Validate and convert one document record from JSON."""
        if not isinstance(item, dict):
            logger.error("Document vault contains a malformed document record.")
            raise DocumentStorageError

        try:
            record = DocumentRecord(
                id=_require_string(item, "id"),
                filename=_require_string(item, "filename"),
                extension=_require_string(item, "extension"),
                source_size_bytes=_require_int(item, "source_size_bytes"),
                content_hash=_require_string(item, "content_hash"),
                extracted_text=_require_string(item, "extracted_text"),
                extracted_text_length=_require_int(item, "extracted_text_length"),
                ingested_at=_require_string(item, "ingested_at"),
            )
            return validate_document_record(record)
        except (DocumentValidationError, DocumentStorageError, KeyError, TypeError):
            logger.error("Document vault contains a malformed document record.")
            raise DocumentStorageError from None

    def _persist(self, documents: list[DocumentRecord]) -> None:
        """Atomically write the current document collection as UTF-8 JSON."""
        payload = {
            "documents": [
                {
                    "id": document.id,
                    "filename": document.filename,
                    "extension": document.extension,
                    "source_size_bytes": document.source_size_bytes,
                    "content_hash": document.content_hash,
                    "extracted_text": document.extracted_text,
                    "extracted_text_length": document.extracted_text_length,
                    "ingested_at": document.ingested_at,
                }
                for document in documents
            ]
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._documents = list(documents)

    def _atomic_write(self, content: str) -> None:
        """Write JSON to a temporary file, then replace the target file."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".documents-",
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
                "Unable to write document vault due to OS error type: %s",
                type(error).__name__,
            )
            raise DocumentStorageError(DOCUMENT_SAVE_ERROR_MESSAGE) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _require_string(item: dict[object, object], key: str) -> str:
    """Return a required string field from a JSON object."""
    value = item.get(key)
    if not isinstance(value, str):
        raise DocumentStorageError
    return value


def _require_int(item: dict[object, object], key: str) -> int:
    """Return a required integer field from a JSON object."""
    value = item.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DocumentStorageError
    return value
