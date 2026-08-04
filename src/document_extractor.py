"""Text extraction interfaces for Knowledge Vault document ingestion."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Protocol

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import ALLOWED_DOCUMENT_EXTENSIONS, MAX_DOCUMENT_TEXT_LENGTH
from src.document import (
    BlankDocumentTextError,
    DocumentTextTooLongError,
    UnsupportedDocumentExtensionError,
    normalize_document_extension,
)

logger = logging.getLogger("ProjectCortana")


class DocumentExtractionError(RuntimeError):
    """Raised when document text cannot be extracted safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class TextExtractor(Protocol):
    """Protocol for extracting plain text from supported document bytes."""

    def extract_text(self, *, extension: str, file_bytes: bytes) -> str:
        """Extract plain text for one supported document type."""


class DefaultTextExtractor:
    """Extract plain text from TXT, Markdown, and PDF document bytes."""

    def extract_text(self, *, extension: str, file_bytes: bytes) -> str:
        """Extract and validate text for a supported document extension."""
        try:
            normalized_extension = normalize_document_extension(extension)
        except UnsupportedDocumentExtensionError as error:
            raise DocumentExtractionError(
                "Cortana: Unsupported document type. "
                f"Supported types: {_supported_extensions_display()}."
            ) from error

        if normalized_extension in {".txt", ".md"}:
            text = self._extract_utf8_text(file_bytes)
        elif normalized_extension == ".pdf":
            text = self._extract_pdf_text(file_bytes)
        else:
            raise DocumentExtractionError(
                "Cortana: Unsupported document type. "
                f"Supported types: {_supported_extensions_display()}."
            )

        return self._validate_extracted_text(text)

    def _extract_utf8_text(self, file_bytes: bytes) -> str:
        """Decode UTF-8 text, including UTF-8 BOM, without silent truncation."""
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            logger.error(
                "Document text decoding failed error_type=%s",
                type(error).__name__,
            )
            raise DocumentExtractionError(
                "Cortana: The document could not be decoded as UTF-8 text."
            ) from error

    def _extract_pdf_text(self, file_bytes: bytes) -> str:
        """Extract text-only content from a PDF without OCR or script execution."""
        try:
            reader = PdfReader(BytesIO(file_bytes), strict=False)
        except PdfReadError as error:
            logger.error(
                "PDF read failed error_type=%s",
                type(error).__name__,
            )
            raise DocumentExtractionError(
                "Cortana: The PDF could not be read safely."
            ) from error
        except Exception as error:  # noqa: BLE001 - keep session resilient
            logger.error(
                "PDF extraction failed error_type=%s",
                type(error).__name__,
            )
            raise DocumentExtractionError(
                "Cortana: The PDF could not be read safely."
            ) from error

        if reader.is_encrypted:
            logger.error("PDF extraction rejected encrypted document.")
            raise DocumentExtractionError(
                "Cortana: Encrypted PDFs are not supported."
            )

        page_texts: list[str] = []
        try:
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
        except Exception as error:  # noqa: BLE001 - keep session resilient
            logger.error(
                "PDF page extraction failed error_type=%s",
                type(error).__name__,
            )
            raise DocumentExtractionError(
                "Cortana: No usable text could be extracted from the PDF."
            ) from error

        return "\n".join(page_texts)

    def _validate_extracted_text(self, text: str) -> str:
        """Reject blank or oversized extracted text without truncation."""
        if not text.strip():
            raise DocumentExtractionError(
                "Cortana: No usable text could be extracted from the document."
            ) from BlankDocumentTextError("Extracted text cannot be blank.")

        if len(text) > MAX_DOCUMENT_TEXT_LENGTH:
            raise DocumentExtractionError(
                "Cortana: Extracted text exceeds the maximum allowed length of "
                f"{MAX_DOCUMENT_TEXT_LENGTH} characters."
            ) from DocumentTextTooLongError(
                f"Extracted text exceeds {MAX_DOCUMENT_TEXT_LENGTH} characters."
            )

        return text


def _supported_extensions_display() -> str:
    """Return a stable display string for supported document extensions."""
    return ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
