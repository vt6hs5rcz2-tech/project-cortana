"""Tests for secure local Knowledge Vault document ingestion."""

import logging
import os
from pathlib import Path

import pytest

from src.config import MAX_DOCUMENT_SOURCE_BYTES
from src.document_extractor import DefaultTextExtractor
from src.document_ingestion import DocumentIngestionError, ingest_local_document
from src.document_vault import DuplicateDocumentHashError, JsonDocumentVault
from tests.document_helpers import TEXT_PDF_BYTES


def _vault(tmp_path: Path) -> JsonDocumentVault:
    return JsonDocumentVault(tmp_path / "documents.json")


def _extractor() -> DefaultTextExtractor:
    return DefaultTextExtractor()


def test_txt_markdown_and_pdf_ingestion(tmp_path: Path) -> None:
    """Supported document types should ingest extracted text and metadata."""
    vault = _vault(tmp_path)
    extractor = _extractor()

    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Plain text body", encoding="utf-8")
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Markdown body", encoding="utf-8")
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(TEXT_PDF_BYTES)

    txt_record = ingest_local_document(
        str(txt_path),
        vault=vault,
        extractor=extractor,
    )
    md_record = ingest_local_document(
        str(md_path),
        vault=vault,
        extractor=extractor,
    )
    pdf_record = ingest_local_document(
        str(pdf_path),
        vault=vault,
        extractor=extractor,
    )

    assert txt_record.filename == "notes.txt"
    assert txt_record.extension == ".txt"
    assert txt_record.extracted_text == "Plain text body"
    assert md_record.extension == ".md"
    assert md_record.extracted_text == "# Markdown body"
    assert pdf_record.extension == ".pdf"
    assert "Hello PDF" in pdf_record.extracted_text
    assert not hasattr(txt_record, "source_path")
    assert "source_path" not in txt_record.__dict__


def test_unsupported_missing_directory_and_oversized_rejection(
    tmp_path: Path,
) -> None:
    """Unsafe or unsupported paths should be rejected with clear local errors."""
    vault = _vault(tmp_path)
    extractor = _extractor()

    unsupported = tmp_path / "malware.exe"
    unsupported.write_bytes(b"MZ")
    with pytest.raises(DocumentIngestionError):
        ingest_local_document(str(unsupported), vault=vault, extractor=extractor)

    with pytest.raises(DocumentIngestionError):
        ingest_local_document(
            str(tmp_path / "missing.txt"),
            vault=vault,
            extractor=extractor,
        )

    with pytest.raises(DocumentIngestionError):
        ingest_local_document(str(tmp_path), vault=vault, extractor=extractor)

    oversized = tmp_path / "big.txt"
    oversized.write_bytes(b"a" * (MAX_DOCUMENT_SOURCE_BYTES + 1))
    with pytest.raises(DocumentIngestionError) as error_info:
        ingest_local_document(str(oversized), vault=vault, extractor=extractor)
    assert "maximum size" in error_info.value.user_message


def test_duplicate_content_reports_existing_id_and_same_name_allowed(
    tmp_path: Path,
) -> None:
    """Duplicate hashes should be rejected; same filename with new content allowed."""
    vault = _vault(tmp_path)
    extractor = _extractor()

    first_path = tmp_path / "report.txt"
    first_path.write_text("same-bytes", encoding="utf-8")
    first = ingest_local_document(str(first_path), vault=vault, extractor=extractor)

    duplicate_path = tmp_path / "copy.txt"
    duplicate_path.write_text("same-bytes", encoding="utf-8")
    with pytest.raises(DuplicateDocumentHashError) as error_info:
        ingest_local_document(str(duplicate_path), vault=vault, extractor=extractor)
    assert first.id in error_info.value.user_message

    renamed = tmp_path / "report.txt"
    renamed.write_text("different-bytes", encoding="utf-8")
    second = ingest_local_document(str(renamed), vault=vault, extractor=extractor)
    assert second.filename == "report.txt"
    assert second.id != first.id


def test_source_path_not_stored_or_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ingestion logs and records must not expose absolute source paths."""
    vault = _vault(tmp_path)
    extractor = _extractor()
    source = tmp_path / "private-notes.txt"
    source.write_text("classified notes", encoding="utf-8")
    absolute = str(source.resolve())

    with caplog.at_level("INFO", logger="ProjectCortana"):
        record = ingest_local_document(absolute, vault=vault, extractor=extractor)

    payload = vault.file_path.read_text(encoding="utf-8")
    combined_logs = " ".join(item.getMessage() for item in caplog.records)

    assert absolute not in payload
    assert absolute not in combined_logs
    assert record.filename == "private-notes.txt"
    assert "classified notes" not in combined_logs


def test_symlink_to_valid_document_is_rejected(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Symbolic links to otherwise valid documents must be rejected safely."""
    vault = _vault(tmp_path)
    extractor = _extractor()
    target = tmp_path / "target-notes.txt"
    target.write_text("symlink target content", encoding="utf-8")
    link_path = tmp_path / "linked-notes.txt"

    try:
        os.symlink(target, link_path)
    except OSError as error:
        pytest.skip(
            "Operating system refused symlink creation "
            f"({type(error).__name__}: {error}). "
            "Enable symlink privileges or Developer Mode to run this coverage."
        )

    assert link_path.is_symlink()
    absolute_link = str(link_path.resolve())
    absolute_target = str(target.resolve())

    with caplog.at_level(logging.ERROR, logger="ProjectCortana"):
        with pytest.raises(DocumentIngestionError) as error_info:
            ingest_local_document(str(link_path), vault=vault, extractor=extractor)

    assert "Symbolic links" in error_info.value.user_message
    assert vault.document_count() == 0
    assert absolute_link not in error_info.value.user_message
    assert absolute_target not in error_info.value.user_message
    assert "symlink target content" not in error_info.value.user_message

    combined_logs = " ".join(record.getMessage() for record in caplog.records)
    assert absolute_link not in combined_logs
    assert absolute_target not in combined_logs
    assert "symlink target content" not in combined_logs
