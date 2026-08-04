"""Tests for in-memory retrieval session state."""

from src.document_retrieval import SourceManifestEntry
from src.retrieval_session import RetrievalSession, fingerprint_query


def _entry(document_id: str, label: str = "[DOC-1:C1]") -> SourceManifestEntry:
    return SourceManifestEntry(
        citation_label=label,
        document_id=document_id,
        filename="notes.txt",
        chunk_index=0,
        start_offset=0,
        end_offset=12,
    )


def test_session_starts_empty_and_records_manifest() -> None:
    """New sessions begin empty and can record a grounded result."""
    session = RetrievalSession(boundary_token="session_token_0001")
    assert session.has_source_manifest is False
    assert session.source_manifest == []
    assert session.query_fingerprint is None

    session.record_grounded_result(
        query="What is the firewall rule?",
        source_manifest=[_entry("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")],
        citation_labels={"[DOC-1:C1]"},
    )

    assert session.has_source_manifest is True
    assert session.query_fingerprint == fingerprint_query(
        "What is the firewall rule?"
    )
    assert session.valid_citation_labels == frozenset({"[DOC-1:C1]"})
    assert session.source_manifest[0].filename == "notes.txt"


def test_clear_and_remove_document_invalidate_manifest() -> None:
    """Clear and document removal should invalidate stale manifest entries."""
    doc_a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    doc_b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    session = RetrievalSession(boundary_token="session_token_0002")
    session.record_grounded_result(
        query="policy question",
        source_manifest=[
            _entry(doc_a, "[DOC-1:C1]"),
            _entry(doc_b, "[DOC-2:C1]"),
        ],
        citation_labels={"[DOC-1:C1]", "[DOC-2:C1]"},
    )

    removed = session.remove_document(doc_a)
    assert removed == 1
    assert [entry.document_id for entry in session.source_manifest] == [doc_b]
    assert session.valid_citation_labels == frozenset({"[DOC-2:C1]"})

    session.clear()
    assert session.has_source_manifest is False
    assert session.query_fingerprint is None
