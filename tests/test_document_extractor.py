"""Tests for Knowledge Vault text extraction."""

import pytest

from src.config import MAX_DOCUMENT_TEXT_LENGTH
from src.document_extractor import DefaultTextExtractor, DocumentExtractionError
from tests.document_helpers import TEXT_PDF_BYTES, blank_pdf_bytes, encrypted_pdf_bytes


def _extractor() -> DefaultTextExtractor:
    return DefaultTextExtractor()


def test_utf8_round_trip_and_bom_handling() -> None:
    """TXT/Markdown extraction should preserve UTF-8 text and strip BOM."""
    extractor = _extractor()
    payload = "Café 安全".encode("utf-8")
    bom_payload = b"\xef\xbb\xbf" + payload

    assert extractor.extract_text(extension=".txt", file_bytes=payload) == "Café 安全"
    assert extractor.extract_text(extension=".txt", file_bytes=bom_payload) == "Café 安全"
    assert (
        extractor.extract_text(extension=".md", file_bytes=b"# Title\n\n- item\n")
        == "# Title\n\n- item\n"
    )


def test_invalid_encoding_and_empty_text_rejection() -> None:
    """Invalid UTF-8 and blank extracted text should raise safe errors."""
    extractor = _extractor()

    with pytest.raises(DocumentExtractionError):
        extractor.extract_text(extension=".txt", file_bytes=b"\xff\xfe\x00")

    with pytest.raises(DocumentExtractionError):
        extractor.extract_text(extension=".txt", file_bytes=b"")

    with pytest.raises(DocumentExtractionError):
        extractor.extract_text(extension=".md", file_bytes=b"   \n\t")


def test_maximum_extracted_length_rejected() -> None:
    """Oversized extracted text should be rejected without truncation."""
    extractor = _extractor()
    oversized = ("a" * (MAX_DOCUMENT_TEXT_LENGTH + 1)).encode("utf-8")

    with pytest.raises(DocumentExtractionError) as error_info:
        extractor.extract_text(extension=".txt", file_bytes=oversized)

    assert "maximum allowed length" in error_info.value.user_message


def test_pdf_valid_malformed_encrypted_and_empty() -> None:
    """PDF extraction should handle valid, malformed, encrypted, and empty PDFs safely."""
    extractor = _extractor()

    extracted = extractor.extract_text(extension=".pdf", file_bytes=TEXT_PDF_BYTES)
    assert "Hello PDF" in extracted

    with pytest.raises(DocumentExtractionError):
        extractor.extract_text(extension=".pdf", file_bytes=b"%PDF-1.4 corrupted")

    with pytest.raises(DocumentExtractionError) as encrypted_error:
        extractor.extract_text(extension=".pdf", file_bytes=encrypted_pdf_bytes())
    assert "Encrypted" in encrypted_error.value.user_message

    with pytest.raises(DocumentExtractionError) as empty_error:
        extractor.extract_text(extension=".pdf", file_bytes=blank_pdf_bytes())
    assert "No usable text" in empty_error.value.user_message


def test_pdf_extraction_errors_do_not_crash_session() -> None:
    """Extractor failures should raise controlled errors instead of crashing."""
    extractor = _extractor()

    try:
        extractor.extract_text(extension=".pdf", file_bytes=b"not-a-pdf")
    except DocumentExtractionError as error:
        assert error.user_message.startswith("Cortana:")
    else:
        pytest.fail("Expected DocumentExtractionError")


def test_no_ocr_for_blank_pdf() -> None:
    """Blank PDFs must fail closed without OCR inventing text."""
    extractor = _extractor()

    with pytest.raises(DocumentExtractionError):
        extractor.extract_text(extension=".pdf", file_bytes=blank_pdf_bytes())
