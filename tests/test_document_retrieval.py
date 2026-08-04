"""Tests for deterministic local lexical document retrieval."""

from __future__ import annotations

import hashlib

import pytest

from src.document import DocumentRecord, create_document
from src.document_chunker import ChunkerConfig, DocumentChunker
from src.document_retrieval import (
    BlankRetrievalQueryError,
    LexicalDocumentRetriever,
    RetrieverConfig,
    build_citation_label,
    build_source_manifest,
)
from src.document_vault import JsonDocumentVault


def _hash_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(filename: str, text: str) -> DocumentRecord:
    return create_document(
        filename=filename,
        extension=".txt",
        source_size_bytes=len(text.encode("utf-8")),
        content_hash=_hash_for(text),
        extracted_text=text,
    )


def _retriever() -> LexicalDocumentRetriever:
    return LexicalDocumentRetriever(
        chunker=DocumentChunker(
            ChunkerConfig(target_chunk_size=200, overlap=20, min_chunk_length=10)
        ),
        config=RetrieverConfig(max_retrieved_chunks=5, max_retrieved_context_chars=2000),
    )


def test_single_and_multi_term_case_insensitive_search() -> None:
    """Lexical search should match case-insensitively and favor multi-term hits."""
    docs = [
        _document("alpha.txt", "The firewall policy blocks outbound traffic."),
        _document(
            "beta.txt",
            "Firewall policy and patch cadence improve endpoint hardening.",
        ),
    ]
    retriever = _retriever()

    single = retriever.search("FIREWALL", docs)
    assert single
    assert any("firewall" in term for result in single for term in result.matched_terms)

    multi = retriever.search("firewall policy", docs)
    assert multi
    assert multi[0].chunk.document_filename in {"alpha.txt", "beta.txt"}
    assert "firewall" in multi[0].matched_terms
    assert "policy" in multi[0].matched_terms


def test_phrase_bonus_and_deterministic_ranking() -> None:
    """Repeated searches should preserve ranking and citation mapping."""
    docs = [
        _document("zulu.txt", "incident response playbook details triage steps."),
        _document("alpha.txt", "The incident response playbook is required reading."),
        _document("alpha.txt", "Unrelated material about cryptography keys."),
    ]
    # Force identical filenames with distinct IDs for tie-break coverage.
    docs[1] = DocumentRecord(
        id=docs[1].id,
        filename="alpha.txt",
        extension=".txt",
        source_size_bytes=docs[1].source_size_bytes,
        content_hash=docs[1].content_hash,
        extracted_text=docs[1].extracted_text,
        extracted_text_length=docs[1].extracted_text_length,
        ingested_at=docs[1].ingested_at,
    )
    docs[2] = DocumentRecord(
        id=docs[2].id,
        filename="alpha.txt",
        extension=".txt",
        source_size_bytes=docs[2].source_size_bytes,
        content_hash=docs[2].content_hash,
        extracted_text=docs[2].extracted_text,
        extracted_text_length=docs[2].extracted_text_length,
        ingested_at=docs[2].ingested_at,
    )

    retriever = _retriever()
    first = retriever.search("incident response playbook", docs)
    second = retriever.search("incident response playbook", docs)

    assert first
    assert [(item.citation_label, item.chunk.id, item.score) for item in first] == [
        (item.citation_label, item.chunk.id, item.score) for item in second
    ]
    assert first[0].citation_label.startswith("[DOC-")
    manifest = build_source_manifest(first)
    assert manifest[0].document_id == first[0].chunk.document_id
    assert manifest[0].filename == first[0].chunk.document_filename
    assert "/" not in manifest[0].filename
    assert "\\" not in manifest[0].filename


def test_no_match_and_duplicate_filenames() -> None:
    """No meaningful match should return empty results; duplicate names stay distinct."""
    docs = [
        _document("same.txt", "Alpha security control description."),
        _document("same.txt", "Beta security control description."),
    ]
    retriever = _retriever()

    assert retriever.search("nonexistenttermxyz", docs) == []

    results = retriever.search("security control", docs)
    assert len(results) >= 2
    assert results[0].chunk.document_id != results[1].chunk.document_id
    assert results[0].chunk.document_filename == results[1].chunk.document_filename


def test_maximum_result_count_and_context_character_limit() -> None:
    """Retriever should enforce chunk count and context character limits."""
    docs = [
        _document("one.txt", "token alpha " * 8),
        _document("two.txt", "token beta " * 8),
        _document("three.txt", "token gamma " * 8),
    ]
    retriever = LexicalDocumentRetriever(
        chunker=DocumentChunker(
            ChunkerConfig(target_chunk_size=80, overlap=0, min_chunk_length=5)
        ),
        config=RetrieverConfig(max_retrieved_chunks=2, max_retrieved_context_chars=120),
    )

    results = retriever.search("token", docs, max_results=2)
    assert len(results) <= 2
    assert sum(len(item.chunk.text) for item in results) <= 120
    assert results


def test_search_vault_local_only(tmp_path: object) -> None:
    """Vault search should read only stored documents and reject blank queries."""
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    vault = JsonDocumentVault(tmp_path / "documents.json")
    vault.add_document(_document("local.txt", "Credential rotation schedule weekly."))
    retriever = _retriever()

    results = retriever.search_vault("credential rotation", vault)
    assert results
    assert results[0].citation_label == build_citation_label(
        1,
        results[0].chunk.chunk_index,
    )

    with pytest.raises(BlankRetrievalQueryError):
        retriever.search("   ", vault.list_documents())
