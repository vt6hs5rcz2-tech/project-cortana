"""Immutable document chunk model for local Knowledge Vault retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.document import (
    BlankDocumentFilenameError,
    DocumentValidationError,
    InvalidDocumentIdError,
    normalize_document_filename,
    validate_document_id,
)


class DocumentChunkValidationError(ValueError):
    """Raised when a document chunk record fails local validation."""


class BlankDocumentChunkError(DocumentChunkValidationError):
    """Raised when chunk text is blank or whitespace-only."""


class InvalidChunkOffsetError(DocumentChunkValidationError):
    """Raised when chunk offsets are inconsistent with chunk text."""


@dataclass(frozen=True)
class DocumentChunk:
    """Immutable chunk of extracted document text with stable identity."""

    id: str
    document_id: str
    document_filename: str
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    text_length: int


def build_chunk_id(document_id: str, chunk_index: int) -> str:
    """Return a deterministic chunk ID derived from document ID and index."""
    validated_document_id = validate_document_id(document_id)
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
        raise DocumentChunkValidationError("Chunk index must be an integer.")
    if chunk_index < 0:
        raise DocumentChunkValidationError("Chunk index cannot be negative.")
    return f"{validated_document_id}:{chunk_index}"


def create_document_chunk(
    *,
    document_id: str,
    document_filename: str,
    chunk_index: int,
    text: str,
    start_offset: int,
    end_offset: int,
) -> DocumentChunk:
    """Create a validated immutable document chunk."""
    validated_document_id = validate_document_id(document_id)
    try:
        validated_filename = normalize_document_filename(document_filename)
    except BlankDocumentFilenameError as error:
        raise DocumentChunkValidationError(str(error)) from error

    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
        raise DocumentChunkValidationError("Chunk index must be an integer.")
    if chunk_index < 0:
        raise DocumentChunkValidationError("Chunk index cannot be negative.")

    if not isinstance(text, str):
        raise DocumentChunkValidationError("Chunk text must be a string.")
    if not text.strip():
        raise BlankDocumentChunkError("Chunk text cannot be blank.")

    if not isinstance(start_offset, int) or isinstance(start_offset, bool):
        raise InvalidChunkOffsetError("Chunk start offset must be an integer.")
    if not isinstance(end_offset, int) or isinstance(end_offset, bool):
        raise InvalidChunkOffsetError("Chunk end offset must be an integer.")
    if start_offset < 0:
        raise InvalidChunkOffsetError("Chunk start offset cannot be negative.")
    if end_offset < start_offset:
        raise InvalidChunkOffsetError(
            "Chunk end offset cannot be less than start offset."
        )
    if end_offset - start_offset != len(text):
        raise InvalidChunkOffsetError(
            "Chunk offsets do not match chunk text length."
        )

    return DocumentChunk(
        id=build_chunk_id(validated_document_id, chunk_index),
        document_id=validated_document_id,
        document_filename=validated_filename,
        chunk_index=chunk_index,
        text=text,
        start_offset=start_offset,
        end_offset=end_offset,
        text_length=len(text),
    )


def validate_document_chunk(chunk: DocumentChunk) -> DocumentChunk:
    """Validate all fields of a document chunk record."""
    try:
        validated = create_document_chunk(
            document_id=chunk.document_id,
            document_filename=chunk.document_filename,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
        )
    except InvalidDocumentIdError as error:
        raise DocumentChunkValidationError(str(error)) from error
    except DocumentValidationError as error:
        raise DocumentChunkValidationError(str(error)) from error

    if chunk.id != validated.id:
        raise DocumentChunkValidationError(
            "Chunk ID does not match document ID and chunk index."
        )
    if chunk.text_length != validated.text_length:
        raise DocumentChunkValidationError(
            "Chunk text length does not match chunk text."
        )
    return validated
