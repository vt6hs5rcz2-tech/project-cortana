"""Deterministic local lexical retrieval over Knowledge Vault documents."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.config import (
    MAX_RETRIEVED_CHUNKS,
    MAX_RETRIEVED_CONTEXT_CHARS,
)
from src.document import DocumentRecord
from src.document_chunk import DocumentChunk
from src.document_chunker import DocumentChunker, DocumentChunkingError
from src.document_vault import DocumentVault

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


class DocumentRetrievalError(ValueError):
    """Raised when local document retrieval cannot complete safely."""


class BlankRetrievalQueryError(DocumentRetrievalError):
    """Raised when a retrieval query is blank or has no usable tokens."""


class RetrievalContextLimitError(DocumentRetrievalError):
    """Raised when selected retrieval context would exceed safe limits."""


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked chunk result from local lexical retrieval."""

    chunk: DocumentChunk
    score: float
    matched_terms: tuple[str, ...]
    citation_label: str


@dataclass(frozen=True)
class SourceManifestEntry:
    """Compact session mapping from a citation label to chunk location."""

    citation_label: str
    document_id: str
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class RetrievedDocumentContext:
    """Immutable retrieved-document context for one grounded AI request."""

    results: tuple[RetrievalResult, ...]
    boundary_token: str
    source_manifest: tuple[SourceManifestEntry, ...]

    @property
    def citation_labels(self) -> frozenset[str]:
        """Return the exact citation labels supplied with this context."""
        return frozenset(result.citation_label for result in self.results)

    @property
    def total_context_characters(self) -> int:
        """Return combined character length of retrieved chunk text."""
        return sum(len(result.chunk.text) for result in self.results)


@dataclass(frozen=True)
class RetrieverConfig:
    """Centralized retrieval result limits."""

    max_retrieved_chunks: int = MAX_RETRIEVED_CHUNKS
    max_retrieved_context_chars: int = MAX_RETRIEVED_CONTEXT_CHARS

    def __post_init__(self) -> None:
        """Validate retriever configuration values."""
        if self.max_retrieved_chunks <= 0:
            raise DocumentRetrievalError(
                "Maximum retrieved chunks must be positive."
            )
        if self.max_retrieved_context_chars <= 0:
            raise DocumentRetrievalError(
                "Maximum retrieved context characters must be positive."
            )


def tokenize_query(query: str) -> tuple[str, ...]:
    """Tokenize and normalize a query into lowercase lexical terms."""
    if not isinstance(query, str):
        raise BlankRetrievalQueryError("Retrieval query must be a string.")
    tokens = tuple(_TOKEN_PATTERN.findall(query.casefold()))
    return tokens


def build_citation_label(document_number: int, chunk_index: int) -> str:
    """Build a compact citation label for one chunk in a retrieval response."""
    if document_number < 1:
        raise DocumentRetrievalError("Document number must be at least 1.")
    if chunk_index < 0:
        raise DocumentRetrievalError("Chunk index cannot be negative.")
    return f"[DOC-{document_number}:C{chunk_index + 1}]"


def build_source_manifest(
    results: Sequence[RetrievalResult],
) -> tuple[SourceManifestEntry, ...]:
    """Build a source manifest from ranked retrieval results."""
    entries: list[SourceManifestEntry] = []
    for result in results:
        entries.append(
            SourceManifestEntry(
                citation_label=result.citation_label,
                document_id=result.chunk.document_id,
                filename=result.chunk.document_filename,
                chunk_index=result.chunk.chunk_index,
                start_offset=result.chunk.start_offset,
                end_offset=result.chunk.end_offset,
            )
        )
    return tuple(entries)


class LexicalDocumentRetriever:
    """Local lexical retriever over stored Knowledge Vault documents."""

    def __init__(
        self,
        *,
        chunker: DocumentChunker | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        """Initialize the retriever with injected chunker and limits."""
        self._chunker = chunker or DocumentChunker()
        self._config = config or RetrieverConfig()

    @property
    def chunker(self) -> DocumentChunker:
        """Return the document chunker used by this retriever."""
        return self._chunker

    @property
    def config(self) -> RetrieverConfig:
        """Return the active retriever configuration."""
        return self._config

    def search(
        self,
        query: str,
        documents: Sequence[DocumentRecord],
        *,
        max_results: int | None = None,
    ) -> list[RetrievalResult]:
        """Rank document chunks for ``query`` using deterministic lexical scoring."""
        tokens = tokenize_query(query)
        if not tokens:
            raise BlankRetrievalQueryError(
                "Retrieval query must contain at least one searchable term."
            )

        limit = max_results if max_results is not None else self._config.max_retrieved_chunks
        if limit <= 0:
            raise DocumentRetrievalError("Maximum results must be positive.")

        scored: list[tuple[float, tuple[str, ...], DocumentChunk]] = []
        for document in documents:
            try:
                chunks = self._chunker.chunk_document(document)
            except DocumentChunkingError as error:
                raise DocumentRetrievalError(str(error)) from error

            for chunk in chunks:
                score, matched = self._score_chunk(chunk.text, tokens, query)
                if score > 0:
                    scored.append((score, matched, chunk))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[2].document_filename.casefold(),
                item[2].document_id,
                item[2].chunk_index,
            )
        )

        selected = self._select_within_limits(scored, limit)
        return self._assign_citation_labels(selected)

    def search_vault(
        self,
        query: str,
        vault: DocumentVault,
        *,
        max_results: int | None = None,
    ) -> list[RetrievalResult]:
        """Search documents currently stored in the local Knowledge Vault."""
        return self.search(
            query,
            vault.list_documents(),
            max_results=max_results,
        )

    def retrieve_within_documents(
        self,
        query: str,
        documents: Sequence[DocumentRecord],
        document_ids: Sequence[str],
        *,
        max_chunks_per_document: int,
        max_total_chunks: int | None = None,
        max_total_chars: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve matching chunks only within explicitly required document IDs.

        ``document_ids`` is required and must be non-empty. Citation labels use
        stable DOC-N numbering from the explicit ID order, not score rank.
        """
        if not document_ids:
            raise DocumentRetrievalError(
                "Scoped retrieval requires at least one explicit document ID."
            )
        if max_chunks_per_document <= 0:
            raise DocumentRetrievalError(
                "Maximum chunks per document must be positive."
            )

        tokens = tokenize_query(query)
        if not tokens:
            raise BlankRetrievalQueryError(
                "Retrieval query must contain at least one searchable term."
            )

        by_id = {document.id: document for document in documents}
        total_chunk_limit = (
            max_total_chunks
            if max_total_chunks is not None
            else self._config.max_retrieved_chunks
        )
        total_char_limit = (
            max_total_chars
            if max_total_chars is not None
            else self._config.max_retrieved_context_chars
        )
        if total_chunk_limit <= 0:
            raise DocumentRetrievalError("Maximum total chunks must be positive.")
        if total_char_limit <= 0:
            raise DocumentRetrievalError(
                "Maximum total context characters must be positive."
            )

        per_document_char_budget = max(1, total_char_limit // len(document_ids))
        results: list[RetrievalResult] = []
        total_chars = 0

        for document_number, document_id in enumerate(document_ids, start=1):
            document = by_id.get(document_id)
            if document is None:
                raise DocumentRetrievalError(
                    f"Document '{document_id}' was not found for scoped retrieval."
                )

            try:
                chunks = self._chunker.chunk_document(document)
            except DocumentChunkingError as error:
                raise DocumentRetrievalError(str(error)) from error

            scored: list[tuple[float, tuple[str, ...], DocumentChunk]] = []
            for chunk in chunks:
                score, matched = self._score_chunk(chunk.text, tokens, query)
                if score > 0:
                    scored.append((score, matched, chunk))

            scored.sort(
                key=lambda item: (
                    -item[0],
                    item[2].chunk_index,
                )
            )

            selected_for_document = 0
            document_chars = 0
            for score, matched, chunk in scored:
                if selected_for_document >= max_chunks_per_document:
                    break
                if len(results) >= total_chunk_limit:
                    break
                if document_chars + len(chunk.text) > per_document_char_budget:
                    if selected_for_document == 0 and len(chunk.text) > per_document_char_budget:
                        raise RetrievalContextLimitError(
                            "A single matching chunk exceeds the per-document "
                            "retrieved context character budget."
                        )
                    break
                if total_chars + len(chunk.text) > total_char_limit:
                    if not results:
                        raise RetrievalContextLimitError(
                            "A single matching chunk exceeds the maximum retrieved "
                            "context character limit."
                        )
                    break

                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=score,
                        matched_terms=matched,
                        citation_label=build_citation_label(
                            document_number,
                            chunk.chunk_index,
                        ),
                    )
                )
                selected_for_document += 1
                document_chars += len(chunk.text)
                total_chars += len(chunk.text)

        return results

    def _select_within_limits(
        self,
        scored: Sequence[tuple[float, tuple[str, ...], DocumentChunk]],
        limit: int,
    ) -> list[tuple[float, tuple[str, ...], DocumentChunk]]:
        """Select top results without exceeding chunk or character limits."""
        selected: list[tuple[float, tuple[str, ...], DocumentChunk]] = []
        total_chars = 0

        for item in scored:
            if len(selected) >= limit:
                break
            chunk = item[2]
            projected = total_chars + len(chunk.text)
            if projected > self._config.max_retrieved_context_chars:
                if not selected:
                    raise RetrievalContextLimitError(
                        "A single matching chunk exceeds the maximum retrieved "
                        "context character limit."
                    )
                break
            selected.append(item)
            total_chars = projected

        return selected

    def _assign_citation_labels(
        self,
        selected: Sequence[tuple[float, tuple[str, ...], DocumentChunk]],
    ) -> list[RetrievalResult]:
        """Assign deterministic DOC-N citation labels preserving rank order."""
        document_numbers: dict[str, int] = {}
        next_document_number = 1
        results: list[RetrievalResult] = []

        for score, matched, chunk in selected:
            document_number = document_numbers.get(chunk.document_id)
            if document_number is None:
                document_number = next_document_number
                document_numbers[chunk.document_id] = document_number
                next_document_number += 1

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched,
                    citation_label=build_citation_label(
                        document_number,
                        chunk.chunk_index,
                    ),
                )
            )
        return results

    def _score_chunk(
        self,
        chunk_text: str,
        query_tokens: Sequence[str],
        raw_query: str,
    ) -> tuple[float, tuple[str, ...]]:
        """Score one chunk with term frequency and optional phrase bonus."""
        lowered = chunk_text.casefold()
        chunk_tokens = _TOKEN_PATTERN.findall(lowered)
        if not chunk_tokens:
            return 0.0, ()

        token_counts: dict[str, int] = {}
        for token in chunk_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        matched: list[str] = []
        score = 0.0
        for term in query_tokens:
            count = token_counts.get(term, 0)
            if count > 0:
                matched.append(term)
                score += float(count)

        if not matched:
            return 0.0, ()

        # Favor chunks that contain more distinct query terms.
        score += float(len(set(matched))) * 0.5

        normalized_query = " ".join(query_tokens)
        if len(query_tokens) > 1 and normalized_query in " ".join(chunk_tokens):
            score += 2.0
        elif raw_query.strip() and raw_query.strip().casefold() in lowered:
            score += 1.0

        return score, tuple(dict.fromkeys(matched))
