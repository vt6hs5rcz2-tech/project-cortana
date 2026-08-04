"""Immutable Knowledge Vault document model for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from src.config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    MAX_DOCUMENT_SOURCE_BYTES,
    MAX_DOCUMENT_TEXT_LENGTH,
)

SHA256_HEX_LENGTH = 64


class DocumentValidationError(ValueError):
    """Raised when a document record fails local validation."""


class BlankDocumentFilenameError(DocumentValidationError):
    """Raised when a document filename is blank."""


class UnsupportedDocumentExtensionError(DocumentValidationError):
    """Raised when a document extension is not allowed."""


class DocumentSourceTooLargeError(DocumentValidationError):
    """Raised when a source file exceeds the configured size limit."""


class DocumentTextTooLongError(DocumentValidationError):
    """Raised when extracted text exceeds the configured length limit."""


class BlankDocumentTextError(DocumentValidationError):
    """Raised when extracted text is blank or whitespace-only."""


class InvalidDocumentHashError(DocumentValidationError):
    """Raised when a content hash is not a valid SHA-256 hex digest."""


class InvalidDocumentTimestampError(DocumentValidationError):
    """Raised when an ingestion timestamp is not valid UTC ISO 8601."""


class InvalidDocumentIdError(DocumentValidationError):
    """Raised when a document ID is not a valid UUID string."""


@dataclass(frozen=True)
class DocumentRecord:
    """Immutable document metadata and extracted text stored in the Knowledge Vault."""

    id: str
    filename: str
    extension: str
    source_size_bytes: int
    content_hash: str
    extracted_text: str
    extracted_text_length: int
    ingested_at: str


def _utc_timestamp() -> str:
    """Return the current UTC time in unambiguous ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def normalize_document_extension(extension: str) -> str:
    """Normalize a file extension to a lowercase dotted form."""
    cleaned = extension.strip().lower()
    if not cleaned:
        raise UnsupportedDocumentExtensionError("Document extension cannot be blank.")
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    if cleaned not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise UnsupportedDocumentExtensionError(
            f"Unsupported document extension '{cleaned}'."
        )
    return cleaned


def normalize_document_filename(filename: str) -> str:
    """Validate and return a non-blank original filename without path components."""
    cleaned = filename.strip()
    if not cleaned:
        raise BlankDocumentFilenameError("Document filename cannot be blank.")
    if "/" in cleaned or "\\" in cleaned:
        raise BlankDocumentFilenameError(
            "Document filename cannot contain path separators."
        )
    return cleaned


def validate_content_hash(content_hash: str) -> str:
    """Validate and return a lowercase SHA-256 hex digest."""
    cleaned = content_hash.strip().lower()
    if len(cleaned) != SHA256_HEX_LENGTH:
        raise InvalidDocumentHashError("Document content hash must be SHA-256 hex.")
    if any(character not in "0123456789abcdef" for character in cleaned):
        raise InvalidDocumentHashError("Document content hash must be SHA-256 hex.")
    return cleaned


def validate_document_id(document_id: str) -> str:
    """Validate and return a canonical UUID document ID string."""
    cleaned = document_id.strip()
    if not cleaned:
        raise InvalidDocumentIdError("Document ID cannot be blank.")
    try:
        return str(UUID(cleaned))
    except ValueError as error:
        raise InvalidDocumentIdError("Document ID must be a valid UUID.") from error


def validate_ingestion_timestamp(timestamp: str) -> str:
    """Validate and return a UTC ISO 8601 timestamp string."""
    cleaned = timestamp.strip()
    if not cleaned:
        raise InvalidDocumentTimestampError("Ingestion timestamp cannot be blank.")

    try:
        normalized = cleaned.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise InvalidDocumentTimestampError(
            "Ingestion timestamp must be valid UTC ISO 8601."
        ) from error

    if parsed.tzinfo is None:
        raise InvalidDocumentTimestampError(
            "Ingestion timestamp must include a UTC timezone."
        )

    utc_offset = parsed.utcoffset()
    if utc_offset is None or utc_offset.total_seconds() != 0:
        raise InvalidDocumentTimestampError(
            "Ingestion timestamp must be in UTC."
        )

    return cleaned


def validate_source_size(source_size_bytes: int) -> int:
    """Validate source file size against centralized limits."""
    if not isinstance(source_size_bytes, int) or isinstance(source_size_bytes, bool):
        raise DocumentValidationError("Source size must be an integer.")
    if source_size_bytes < 0:
        raise DocumentValidationError("Source size cannot be negative.")
    if source_size_bytes > MAX_DOCUMENT_SOURCE_BYTES:
        raise DocumentSourceTooLargeError(
            f"Source file exceeds the maximum size of {MAX_DOCUMENT_SOURCE_BYTES} bytes."
        )
    return source_size_bytes


def validate_extracted_text(extracted_text: str) -> str:
    """Validate extracted text without silently truncating content."""
    if not isinstance(extracted_text, str):
        raise DocumentValidationError("Extracted text must be a string.")
    if not extracted_text.strip():
        raise BlankDocumentTextError("Extracted text cannot be blank.")
    if len(extracted_text) > MAX_DOCUMENT_TEXT_LENGTH:
        raise DocumentTextTooLongError(
            "Extracted text exceeds the maximum length of "
            f"{MAX_DOCUMENT_TEXT_LENGTH} characters."
        )
    return extracted_text


def create_document(
    *,
    filename: str,
    extension: str,
    source_size_bytes: int,
    content_hash: str,
    extracted_text: str,
) -> DocumentRecord:
    """Create a validated immutable document record for vault storage."""
    normalized_filename = normalize_document_filename(filename)
    normalized_extension = normalize_document_extension(extension)
    validated_size = validate_source_size(source_size_bytes)
    validated_hash = validate_content_hash(content_hash)
    validated_text = validate_extracted_text(extracted_text)

    return DocumentRecord(
        id=str(uuid4()),
        filename=normalized_filename,
        extension=normalized_extension,
        source_size_bytes=validated_size,
        content_hash=validated_hash,
        extracted_text=validated_text,
        extracted_text_length=len(validated_text),
        ingested_at=_utc_timestamp(),
    )


def validate_document_record(record: DocumentRecord) -> DocumentRecord:
    """Validate all fields of a loaded document record."""
    document_id = validate_document_id(record.id)
    filename = normalize_document_filename(record.filename)
    extension = normalize_document_extension(record.extension)
    source_size_bytes = validate_source_size(record.source_size_bytes)
    content_hash = validate_content_hash(record.content_hash)
    extracted_text = validate_extracted_text(record.extracted_text)
    ingested_at = validate_ingestion_timestamp(record.ingested_at)

    if not isinstance(record.extracted_text_length, int) or isinstance(
        record.extracted_text_length, bool
    ):
        raise DocumentValidationError("Extracted text length must be an integer.")
    if record.extracted_text_length != len(extracted_text):
        raise DocumentValidationError(
            "Extracted text length does not match extracted text."
        )

    return DocumentRecord(
        id=document_id,
        filename=filename,
        extension=extension,
        source_size_bytes=source_size_bytes,
        content_hash=content_hash,
        extracted_text=extracted_text,
        extracted_text_length=len(extracted_text),
        ingested_at=ingested_at,
    )
