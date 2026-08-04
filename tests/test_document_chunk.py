"""Tests for immutable Knowledge Vault document chunks."""

from __future__ import annotations

import pytest

from src.document_chunk import (
    BlankDocumentChunkError,
    DocumentChunk,
    DocumentChunkValidationError,
    InvalidChunkOffsetError,
    build_chunk_id,
    create_document_chunk,
    validate_document_chunk,
)

DOCUMENT_ID = "11111111-1111-4111-8111-111111111111"


def test_create_document_chunk_is_immutable_and_validated() -> None:
    """Chunk records should be frozen with validated fields and offsets."""
    chunk = create_document_chunk(
        document_id=DOCUMENT_ID,
        document_filename="notes.txt",
        chunk_index=0,
        text="alpha beta",
        start_offset=0,
        end_offset=10,
    )

    assert isinstance(chunk, DocumentChunk)
    assert chunk.id == f"{DOCUMENT_ID}:0"
    assert chunk.text_length == 10
    with pytest.raises(AttributeError):
        chunk.text = "changed"  # type: ignore[misc]


def test_chunk_ids_are_deterministic() -> None:
    """Chunk IDs should derive stably from document ID and index."""
    assert build_chunk_id(DOCUMENT_ID, 3) == f"{DOCUMENT_ID}:3"
    first = create_document_chunk(
        document_id=DOCUMENT_ID,
        document_filename="a.txt",
        chunk_index=2,
        text="same",
        start_offset=10,
        end_offset=14,
    )
    second = create_document_chunk(
        document_id=DOCUMENT_ID,
        document_filename="a.txt",
        chunk_index=2,
        text="same",
        start_offset=10,
        end_offset=14,
    )
    assert first.id == second.id == f"{DOCUMENT_ID}:2"


def test_blank_chunks_and_bad_offsets_are_rejected() -> None:
    """Blank text and inconsistent offsets should raise controlled errors."""
    with pytest.raises(BlankDocumentChunkError):
        create_document_chunk(
            document_id=DOCUMENT_ID,
            document_filename="a.txt",
            chunk_index=0,
            text="   ",
            start_offset=0,
            end_offset=3,
        )

    with pytest.raises(InvalidChunkOffsetError):
        create_document_chunk(
            document_id=DOCUMENT_ID,
            document_filename="a.txt",
            chunk_index=0,
            text="abc",
            start_offset=0,
            end_offset=2,
        )


def test_validate_document_chunk_checks_derived_id() -> None:
    """Validation should reject mismatched derived chunk IDs."""
    valid = create_document_chunk(
        document_id=DOCUMENT_ID,
        document_filename="a.txt",
        chunk_index=1,
        text="hello",
        start_offset=5,
        end_offset=10,
    )
    assert validate_document_chunk(valid) == valid

    broken = DocumentChunk(
        id="not-derived",
        document_id=DOCUMENT_ID,
        document_filename="a.txt",
        chunk_index=1,
        text="hello",
        start_offset=5,
        end_offset=10,
        text_length=5,
    )
    with pytest.raises(DocumentChunkValidationError):
        validate_document_chunk(broken)
