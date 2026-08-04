"""Deterministic local document chunking for Knowledge Vault retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    CHUNK_OVERLAP,
    MAX_CHUNKS_PER_DOCUMENT,
    MIN_CHUNK_LENGTH,
    TARGET_CHUNK_SIZE,
)
from src.document import DocumentRecord
from src.document_chunk import DocumentChunk, create_document_chunk


class DocumentChunkingError(ValueError):
    """Raised when document text cannot be chunked within safe limits."""


class BlankDocumentChunkingError(DocumentChunkingError):
    """Raised when extracted text is blank or whitespace-only."""


class DocumentChunkLimitError(DocumentChunkingError):
    """Raised when chunking would exceed the maximum chunks per document."""


@dataclass(frozen=True)
class ChunkerConfig:
    """Centralized chunking parameters for deterministic local chunking."""

    target_chunk_size: int = TARGET_CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP
    min_chunk_length: int = MIN_CHUNK_LENGTH
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT

    def __post_init__(self) -> None:
        """Validate chunker configuration values."""
        if self.target_chunk_size <= 0:
            raise DocumentChunkingError("Target chunk size must be positive.")
        if self.overlap < 0:
            raise DocumentChunkingError("Chunk overlap cannot be negative.")
        if self.overlap >= self.target_chunk_size:
            raise DocumentChunkingError(
                "Chunk overlap must be smaller than the target chunk size."
            )
        if self.min_chunk_length <= 0:
            raise DocumentChunkingError("Minimum chunk length must be positive.")
        if self.max_chunks_per_document <= 0:
            raise DocumentChunkingError(
                "Maximum chunks per document must be positive."
            )


class DocumentChunker:
    """Pure, deterministic chunker over extracted document text."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialize the chunker with centralized or injected limits."""
        self._config = config or ChunkerConfig()

    @property
    def config(self) -> ChunkerConfig:
        """Return the active chunker configuration."""
        return self._config

    def chunk_text(
        self,
        *,
        document_id: str,
        document_filename: str,
        text: str,
    ) -> list[DocumentChunk]:
        """Chunk raw extracted text into validated immutable chunk records."""
        if not isinstance(text, str):
            raise DocumentChunkingError("Document text must be a string.")
        if not text.strip():
            raise BlankDocumentChunkingError(
                "Extracted text cannot be blank."
            )

        ranges = self._compute_ranges(text)
        if len(ranges) > self._config.max_chunks_per_document:
            raise DocumentChunkLimitError(
                "Document exceeds the maximum of "
                f"{self._config.max_chunks_per_document} chunks."
            )

        chunks: list[DocumentChunk] = []
        for index, (start, end) in enumerate(ranges):
            chunk_text = text[start:end]
            chunks.append(
                create_document_chunk(
                    document_id=document_id,
                    document_filename=document_filename,
                    chunk_index=index,
                    text=chunk_text,
                    start_offset=start,
                    end_offset=end,
                )
            )
        return chunks

    def chunk_document(self, document: DocumentRecord) -> list[DocumentChunk]:
        """Chunk one Knowledge Vault document record."""
        return self.chunk_text(
            document_id=document.id,
            document_filename=document.filename,
            text=document.extracted_text,
        )

    def _compute_ranges(self, text: str) -> list[tuple[int, int]]:
        """Return deterministic (start, end) character ranges for ``text``."""
        target = self._config.target_chunk_size
        overlap = self._config.overlap
        min_length = self._config.min_chunk_length
        text_length = len(text)

        if text_length <= target:
            return [(0, text_length)]

        ranges: list[tuple[int, int]] = []
        start = 0

        while start < text_length:
            ideal_end = min(start + target, text_length)
            if ideal_end >= text_length:
                end = text_length
            else:
                end = self._prefer_boundary(text, start, ideal_end)

            if end <= start:
                end = min(start + target, text_length)

            # Avoid leaving an unusable trailing fragment when possible.
            remaining = text_length - end
            if 0 < remaining < min_length and end < text_length:
                end = text_length

            ranges.append((start, end))

            if end >= text_length:
                break

            next_start = end - overlap
            if next_start <= start:
                next_start = end
            if next_start >= text_length:
                break
            start = next_start

        return ranges

    def _prefer_boundary(self, text: str, start: int, ideal_end: int) -> int:
        """Prefer paragraph, then sentence, then whitespace boundaries."""
        window_start = start + max(self._config.min_chunk_length, 1)
        if window_start >= ideal_end:
            return ideal_end

        search_region = text[window_start:ideal_end]

        paragraph_break = search_region.rfind("\n\n")
        if paragraph_break != -1:
            return window_start + paragraph_break + 2

        for separator in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
            sentence_break = search_region.rfind(separator)
            if sentence_break != -1:
                return window_start + sentence_break + len(separator)

        whitespace_break = max(
            search_region.rfind(" "),
            search_region.rfind("\n"),
            search_region.rfind("\t"),
        )
        if whitespace_break != -1:
            return window_start + whitespace_break + 1

        return ideal_end
