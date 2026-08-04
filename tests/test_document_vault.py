"""Tests for Project Cortana JSON-backed Knowledge Vault storage."""

import hashlib
import logging
import json
from pathlib import Path

import pytest

from src.config import MAX_STORED_DOCUMENTS
from src.document import DocumentRecord, create_document
from src.document_vault import DocumentStorageError, JsonDocumentVault


def _vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _record(text: str, filename: str = "notes.txt") -> DocumentRecord:
    payload = text.encode("utf-8")
    return create_document(
        filename=filename,
        extension=".txt",
        source_size_bytes=len(payload),
        content_hash=hashlib.sha256(payload).hexdigest(),
        extracted_text=text,
    )


def test_missing_file_returns_empty_vault(tmp_path: Path) -> None:
    """A missing vault file should behave like an empty store."""
    vault = _vault(tmp_path)

    assert vault.list_documents() == []
    assert vault.document_count() == 0
    assert not vault.file_path.exists()


def test_save_and_reload_across_vault_instances(tmp_path: Path) -> None:
    """Saved documents should reload from disk in a new vault instance."""
    first = _vault(tmp_path)
    saved = first.add_document(_record("Persist this"))

    second = _vault(tmp_path)
    documents = second.list_documents()

    assert len(documents) == 1
    assert documents[0].id == saved.id
    assert documents[0].extracted_text == "Persist this"
    assert documents[0].ingested_at == saved.ingested_at


def test_preserve_order_and_find_by_hash(tmp_path: Path) -> None:
    """Documents should retain insertion order and be findable by content hash."""
    vault = _vault(tmp_path)
    first = vault.add_document(_record("First", "a.txt"))
    second = vault.add_document(_record("Second", "b.txt"))

    assert [document.id for document in vault.list_documents()] == [first.id, second.id]
    assert vault.find_by_content_hash(second.content_hash) == second
    assert vault.get_document(first.id) == first


def test_delete_one_and_delete_missing_and_delete_all(tmp_path: Path) -> None:
    """Delete operations should persist correctly across reloads."""
    vault = _vault(tmp_path)
    first = vault.add_document(_record("Keep", "keep.txt"))
    second = vault.add_document(_record("Remove", "remove.txt"))

    assert vault.delete_document(second.id) is True
    assert vault.delete_document("missing-id") is False
    reloaded = _vault(tmp_path)
    assert [document.id for document in reloaded.list_documents()] == [first.id]

    assert reloaded.delete_all_documents() == 1
    assert _vault(tmp_path).list_documents() == []


def test_utf8_document_text_round_trip_and_parent_dirs(tmp_path: Path) -> None:
    """UTF-8 extracted text should survive save/reload and create parents."""
    nested = tmp_path / "nested" / "app" / "documents.json"
    vault = JsonDocumentVault(nested)
    text = "Café 安全 🔐"

    vault.add_document(_record(text))
    reloaded = JsonDocumentVault(nested).list_documents()

    assert nested.exists()
    assert reloaded[0].extracted_text == text


def test_atomic_replacement_and_temp_cleanup(tmp_path: Path) -> None:
    """Persisted output should be complete JSON with temporary files cleaned up."""
    vault = _vault(tmp_path)
    vault.add_document(_record("Atomic write"))

    raw = vault.file_path.read_text(encoding="utf-8")
    leftover_temps = list(tmp_path.glob(".documents-*.tmp"))

    assert '"documents"' in raw
    assert "Atomic write" in raw
    assert leftover_temps == []


def test_malformed_json_and_empty_file_preserved(tmp_path: Path) -> None:
    """Corrupt vault files should raise safely and remain unchanged."""
    path = tmp_path / "documents.json"
    original = "{not-valid-json"
    path.write_text(original, encoding="utf-8")
    vault = JsonDocumentVault(path)

    with pytest.raises(DocumentStorageError):
        vault.list_documents()
    assert path.read_text(encoding="utf-8") == original

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("", encoding="utf-8")
    empty_vault = JsonDocumentVault(empty_path)
    with pytest.raises(DocumentStorageError):
        empty_vault.list_documents()
    assert empty_path.read_text(encoding="utf-8") == ""


def test_invalid_structure_malformed_records_and_duplicates(tmp_path: Path) -> None:
    """Invalid structure, malformed records, and duplicates should fail closed."""
    invalid_top = tmp_path / "invalid.json"
    invalid_top.write_text('["not-an-object"]', encoding="utf-8")
    with pytest.raises(DocumentStorageError):
        JsonDocumentVault(invalid_top).list_documents()
    assert invalid_top.read_text(encoding="utf-8") == '["not-an-object"]'

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"documents": [{"id": "abc"}]}', encoding="utf-8")
    with pytest.raises(DocumentStorageError):
        JsonDocumentVault(malformed).list_documents()

    valid = create_document(
        filename="a.txt",
        extension=".txt",
        source_size_bytes=1,
        content_hash=hashlib.sha256(b"a").hexdigest(),
        extracted_text="a",
    )
    duplicate_ids = {
        "documents": [
            {
                "id": valid.id,
                "filename": valid.filename,
                "extension": valid.extension,
                "source_size_bytes": valid.source_size_bytes,
                "content_hash": valid.content_hash,
                "extracted_text": valid.extracted_text,
                "extracted_text_length": valid.extracted_text_length,
                "ingested_at": valid.ingested_at,
            },
            {
                "id": valid.id,
                "filename": "b.txt",
                "extension": ".txt",
                "source_size_bytes": 1,
                "content_hash": hashlib.sha256(b"b").hexdigest(),
                "extracted_text": "b",
                "extracted_text_length": 1,
                "ingested_at": valid.ingested_at,
            },
        ]
    }
    duplicate_path = tmp_path / "dup-id.json"
    duplicate_path.write_text(json.dumps(duplicate_ids), encoding="utf-8")
    with pytest.raises(DocumentStorageError):
        JsonDocumentVault(duplicate_path).list_documents()

    second = create_document(
        filename="c.txt",
        extension=".txt",
        source_size_bytes=1,
        content_hash=hashlib.sha256(b"c").hexdigest(),
        extracted_text="c",
    )
    duplicate_hashes = {
        "documents": [
            {
                "id": valid.id,
                "filename": valid.filename,
                "extension": valid.extension,
                "source_size_bytes": valid.source_size_bytes,
                "content_hash": valid.content_hash,
                "extracted_text": valid.extracted_text,
                "extracted_text_length": valid.extracted_text_length,
                "ingested_at": valid.ingested_at,
            },
            {
                "id": second.id,
                "filename": second.filename,
                "extension": second.extension,
                "source_size_bytes": second.source_size_bytes,
                "content_hash": valid.content_hash,
                "extracted_text": second.extracted_text,
                "extracted_text_length": second.extracted_text_length,
                "ingested_at": second.ingested_at,
            },
        ]
    }
    duplicate_hash_path = tmp_path / "dup-hash.json"
    duplicate_hash_path.write_text(json.dumps(duplicate_hashes), encoding="utf-8")
    with pytest.raises(DocumentStorageError):
        JsonDocumentVault(duplicate_hash_path).list_documents()


def test_count_limit_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding beyond the configured document capacity should be rejected."""
    monkeypatch.setattr("src.document_vault.MAX_STORED_DOCUMENTS", 1)
    vault = _vault(tmp_path)
    vault.add_document(_record("one"))

    with pytest.raises(DocumentStorageError):
        vault.add_document(_record("two"))


def test_loaded_count_over_limit_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault files exceeding the configured count should not load."""
    monkeypatch.setattr("src.document_vault.MAX_STORED_DOCUMENTS", 1)
    records = []
    for index in range(2):
        text = f"doc-{index}"
        record = create_document(
            filename=f"{index}.txt",
            extension=".txt",
            source_size_bytes=len(text),
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            extracted_text=text,
        )
        records.append(
            {
                "id": record.id,
                "filename": record.filename,
                "extension": record.extension,
                "source_size_bytes": record.source_size_bytes,
                "content_hash": record.content_hash,
                "extracted_text": record.extracted_text,
                "extracted_text_length": record.extracted_text_length,
                "ingested_at": record.ingested_at,
            }
        )
    path = tmp_path / "documents.json"
    path.write_text(json.dumps({"documents": records}), encoding="utf-8")

    with pytest.raises(DocumentStorageError):
        JsonDocumentVault(path).list_documents()
    assert path.exists()


def test_document_text_is_not_written_to_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Diagnostic logs must not include document text or raw corrupt JSON."""
    path = tmp_path / "documents.json"
    secret_text = "super-secret-document-content"
    path.write_text("{bad-json", encoding="utf-8")
    vault = JsonDocumentVault(path)

    with caplog.at_level(logging.ERROR, logger="ProjectCortana"):
        with pytest.raises(DocumentStorageError):
            vault.list_documents()

        ok_vault = _vault(tmp_path / "ok")
        ok_vault.add_document(_record(secret_text))

    combined_logs = " ".join(record.getMessage() for record in caplog.records)
    assert secret_text not in combined_logs
    assert "{bad-json" not in combined_logs
    assert MAX_STORED_DOCUMENTS >= 1
