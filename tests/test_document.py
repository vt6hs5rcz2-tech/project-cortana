"""Tests for Project Cortana Knowledge Vault document model validation."""

from dataclasses import FrozenInstanceError
from datetime import datetime
from hashlib import sha256

import pytest

from src.config import MAX_DOCUMENT_SOURCE_BYTES, MAX_DOCUMENT_TEXT_LENGTH
from src.document import (
    BlankDocumentFilenameError,
    BlankDocumentTextError,
    DocumentRecord,
    DocumentSourceTooLargeError,
    DocumentTextTooLongError,
    DocumentValidationError,
    InvalidDocumentHashError,
    UnsupportedDocumentExtensionError,
    create_document,
    normalize_document_extension,
    validate_document_record,
)


def _valid_hash(payload: bytes = b"content") -> str:
    return sha256(payload).hexdigest()


def test_create_document_generates_uuid_and_utc_timestamp() -> None:
    """Each document should receive a UUID and UTC ISO 8601 ingestion timestamp."""
    record = create_document(
        filename="notes.txt",
        extension=".txt",
        source_size_bytes=12,
        content_hash=_valid_hash(),
        extracted_text="hello world",
    )

    assert len(record.id) == 36
    assert record.ingested_at.endswith("Z")
    parsed = datetime.fromisoformat(record.ingested_at.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    utc_offset = parsed.utcoffset()
    assert utc_offset is not None
    assert utc_offset.total_seconds() == 0


def test_document_record_is_immutable() -> None:
    """Document records should reject attribute mutation."""
    record = create_document(
        filename="notes.txt",
        extension=".TXT",
        source_size_bytes=5,
        content_hash=_valid_hash(b"abcde"),
        extracted_text="abcde",
    )

    with pytest.raises(FrozenInstanceError):
        record.filename = "changed.txt"  # type: ignore[misc]

    assert isinstance(record, DocumentRecord)


def test_extension_normalization() -> None:
    """Extensions should normalize to lowercase dotted forms."""
    assert normalize_document_extension("PDF") == ".pdf"
    assert normalize_document_extension(".Md") == ".md"

    with pytest.raises(UnsupportedDocumentExtensionError):
        normalize_document_extension(".exe")


def test_filename_validation_rejects_blank_and_separators() -> None:
    """Blank filenames and path separators should be rejected."""
    with pytest.raises(BlankDocumentFilenameError):
        create_document(
            filename="   ",
            extension=".txt",
            source_size_bytes=1,
            content_hash=_valid_hash(b"a"),
            extracted_text="a",
        )

    with pytest.raises(BlankDocumentFilenameError):
        create_document(
            filename="folder/notes.txt",
            extension=".txt",
            source_size_bytes=1,
            content_hash=_valid_hash(b"a"),
            extracted_text="a",
        )


def test_sha256_and_size_and_text_validation() -> None:
    """Hash, source size, and extracted-text length should be enforced."""
    with pytest.raises(InvalidDocumentHashError):
        create_document(
            filename="notes.txt",
            extension=".txt",
            source_size_bytes=1,
            content_hash="not-a-hash",
            extracted_text="a",
        )

    with pytest.raises(DocumentSourceTooLargeError):
        create_document(
            filename="notes.txt",
            extension=".txt",
            source_size_bytes=MAX_DOCUMENT_SOURCE_BYTES + 1,
            content_hash=_valid_hash(b"a"),
            extracted_text="a",
        )

    with pytest.raises(BlankDocumentTextError):
        create_document(
            filename="notes.txt",
            extension=".txt",
            source_size_bytes=1,
            content_hash=_valid_hash(b"a"),
            extracted_text="   \n",
        )

    with pytest.raises(DocumentTextTooLongError):
        create_document(
            filename="notes.txt",
            extension=".txt",
            source_size_bytes=1,
            content_hash=_valid_hash(b"a"),
            extracted_text="a" * (MAX_DOCUMENT_TEXT_LENGTH + 1),
        )


def test_validate_document_record_checks_length_consistency() -> None:
    """Loaded records should reject mismatched extracted-text lengths."""
    record = create_document(
        filename="notes.txt",
        extension=".txt",
        source_size_bytes=4,
        content_hash=_valid_hash(b"test"),
        extracted_text="test",
    )
    broken = DocumentRecord(
        id=record.id,
        filename=record.filename,
        extension=record.extension,
        source_size_bytes=record.source_size_bytes,
        content_hash=record.content_hash,
        extracted_text=record.extracted_text,
        extracted_text_length=record.extracted_text_length + 1,
        ingested_at=record.ingested_at,
    )

    with pytest.raises(DocumentValidationError):
        validate_document_record(broken)
