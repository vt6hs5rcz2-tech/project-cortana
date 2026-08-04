"""Tests for deterministic local document chunking."""

from __future__ import annotations

import pytest

from src.document import create_document
from src.document_chunker import (
    BlankDocumentChunkingError,
    ChunkerConfig,
    DocumentChunkLimitError,
    DocumentChunker,
)

DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"


def _hash_for(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_chunking_is_deterministic_and_preserves_text() -> None:
    """Chunking the same text twice should yield identical chunks and offsets."""
    text = (
        "Paragraph one discusses firewalls and network segmentation.\n\n"
        "Paragraph two covers endpoint hardening and patch cadence.\n\n"
        + ("Long uninterrupted tokenstream " * 80)
    )
    chunker = DocumentChunker(
        ChunkerConfig(target_chunk_size=120, overlap=20, min_chunk_length=10)
    )

    first = chunker.chunk_text(
        document_id=DOCUMENT_ID,
        document_filename="guide.txt",
        text=text,
    )
    second = chunker.chunk_text(
        document_id=DOCUMENT_ID,
        document_filename="guide.txt",
        text=text,
    )

    assert first == second
    assert first
    for chunk in first:
        assert text[chunk.start_offset:chunk.end_offset] == chunk.text
        assert chunk.text_length == len(chunk.text)


def test_prefers_paragraph_and_sentence_boundaries() -> None:
    """Chunker should prefer paragraph breaks before hard cuts when practical."""
    prefix = "A" * 40
    text = prefix + "\n\n" + ("B" * 80)
    chunker = DocumentChunker(
        ChunkerConfig(target_chunk_size=60, overlap=10, min_chunk_length=10)
    )

    chunks = chunker.chunk_text(
        document_id=DOCUMENT_ID,
        document_filename="bounds.txt",
        text=text,
    )

    assert chunks[0].end_offset == len(prefix) + 2
    assert chunks[0].text.endswith("\n\n")


def test_long_uninterrupted_text_and_overlap_do_not_loop() -> None:
    """Overlap should advance safely without infinite loops or silent omission."""
    text = "X" * 500
    chunker = DocumentChunker(
        ChunkerConfig(target_chunk_size=100, overlap=25, min_chunk_length=10)
    )

    chunks = chunker.chunk_text(
        document_id=DOCUMENT_ID,
        document_filename="stream.txt",
        text=text,
    )

    assert len(chunks) >= 5
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(text)
    # Overlap means covered ranges advance; consecutive starts must increase.
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert right.start_offset > left.start_offset
        assert right.start_offset <= left.end_offset


def test_empty_input_raises_controlled_error() -> None:
    """Whitespace-only text should raise a controlled blank-text error."""
    chunker = DocumentChunker()
    with pytest.raises(BlankDocumentChunkingError):
        chunker.chunk_text(
            document_id=DOCUMENT_ID,
            document_filename="empty.txt",
            text="   \n",
        )


def test_maximum_chunks_is_not_silently_truncated() -> None:
    """Exceeding max chunks should raise rather than omit searchable text."""
    text = ("word " * 2000).strip()
    chunker = DocumentChunker(
        ChunkerConfig(
            target_chunk_size=50,
            overlap=0,
            min_chunk_length=5,
            max_chunks_per_document=3,
        )
    )

    with pytest.raises(DocumentChunkLimitError):
        chunker.chunk_text(
            document_id=DOCUMENT_ID,
            document_filename="big.txt",
            text=text,
        )


def test_chunk_document_uses_record_fields() -> None:
    """Document records should chunk using stored ID, filename, and text."""
    text = "Alpha. " * 40
    document = create_document(
        filename="memo.txt",
        extension=".txt",
        source_size_bytes=len(text.encode("utf-8")),
        content_hash=_hash_for(text),
        extracted_text=text,
    )
    chunker = DocumentChunker(
        ChunkerConfig(target_chunk_size=80, overlap=10, min_chunk_length=10)
    )

    chunks = chunker.chunk_document(document)
    assert chunks
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert all(chunk.document_filename == "memo.txt" for chunk in chunks)
