"""Secure local document ingestion for the Knowledge Vault."""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path

from src.config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    MAX_DOCUMENT_SOURCE_BYTES,
)
from src.document import (
    DocumentRecord,
    DocumentValidationError,
    UnsupportedDocumentExtensionError,
    create_document,
    normalize_document_extension,
    normalize_document_filename,
)
from src.document_extractor import DocumentExtractionError, TextExtractor
from src.document_vault import (
    DocumentCountLimitError,
    DocumentStorageError,
    DocumentVault,
    DuplicateDocumentHashError,
)
from src.path_argument_utils import strip_path_argument_quotes

logger = logging.getLogger("ProjectCortana")


class DocumentIngestionError(RuntimeError):
    """Raised when a local document cannot be ingested safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


def ingest_local_document(
    path_argument: str,
    *,
    vault: DocumentVault,
    extractor: TextExtractor,
) -> DocumentRecord:
    """Ingest one explicitly supplied local file into the Knowledge Vault."""
    cleaned_argument = strip_path_argument_quotes(path_argument)
    if not cleaned_argument.strip():
        raise DocumentIngestionError(
            "Cortana: Please provide a file path. Usage: /add-document <path>"
        )

    file_bytes, filename, extension, source_size_bytes = _read_regular_file_bytes(
        cleaned_argument
    )

    try:
        normalized_extension = normalize_document_extension(extension)
        normalized_filename = normalize_document_filename(filename)
    except UnsupportedDocumentExtensionError as error:
        raise DocumentIngestionError(
            "Cortana: Unsupported document type. "
            f"Supported types: {_supported_extensions_display()}."
        ) from error
    except DocumentValidationError as error:
        raise DocumentIngestionError(
            "Cortana: The document filename is invalid."
        ) from error

    content_hash = hashlib.sha256(file_bytes).hexdigest()

    try:
        existing = vault.find_by_content_hash(content_hash)
    except DocumentStorageError:
        raise
    if existing is not None:
        raise DuplicateDocumentHashError(existing.id)

    try:
        extracted_text = extractor.extract_text(
            extension=normalized_extension,
            file_bytes=file_bytes,
        )
    except DocumentExtractionError:
        raise
    except Exception as error:
        logger.error(
            "Document extraction failed unexpectedly error_type=%s extension=%s",
            type(error).__name__,
            normalized_extension,
        )
        raise DocumentIngestionError(
            "Cortana: The document could not be processed safely."
        ) from error

    record = create_document(
        filename=normalized_filename,
        extension=normalized_extension,
        source_size_bytes=source_size_bytes,
        content_hash=content_hash,
        extracted_text=extracted_text,
    )

    try:
        stored = vault.add_document(record)
    except (DuplicateDocumentHashError, DocumentCountLimitError, DocumentStorageError):
        raise
    except Exception as error:
        logger.error(
            "Document vault add failed unexpectedly error_type=%s",
            type(error).__name__,
        )
        raise DocumentIngestionError(
            "Cortana: Knowledge Vault data could not be updated safely."
        ) from error

    logger.info(
        "Document ingested id=%s filename=%s extension=%s "
        "source_bytes=%s extracted_chars=%s",
        stored.id,
        stored.filename,
        stored.extension,
        stored.source_size_bytes,
        stored.extracted_text_length,
    )
    return stored


def _read_regular_file_bytes(
    path_argument: str,
) -> tuple[bytes, str, str, int]:
    """Open and read one regular local file with conservative path checks."""
    candidate = Path(path_argument)

    try:
        if candidate.is_symlink():
            raise DocumentIngestionError(
                "Cortana: Symbolic links and reparse points cannot be ingested."
            )
    except OSError as error:
        logger.error(
            "Document path inspection failed error_type=%s",
            type(error).__name__,
        )
        raise DocumentIngestionError(
            "Cortana: The specified file could not be accessed."
        ) from error

    try:
        if not candidate.exists():
            raise DocumentIngestionError(
                "Cortana: The specified file could not be found."
            )
    except OSError as error:
        logger.error(
            "Document path existence check failed error_type=%s",
            type(error).__name__,
        )
        raise DocumentIngestionError(
            "Cortana: The specified file could not be accessed."
        ) from error

    try:
        if candidate.is_dir():
            raise DocumentIngestionError(
                "Cortana: Directories cannot be ingested. Provide a single file path."
            )
    except OSError as error:
        logger.error(
            "Document path type check failed error_type=%s",
            type(error).__name__,
        )
        raise DocumentIngestionError(
            "Cortana: The specified file could not be accessed."
        ) from error

    filename = candidate.name
    extension = candidate.suffix

    try:
        file_descriptor = os.open(
            os.fspath(candidate),
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as error:
        raise DocumentIngestionError(
            "Cortana: The specified file could not be found."
        ) from error
    except OSError as error:
        logger.error(
            "Document open failed error_type=%s",
            type(error).__name__,
        )
        raise DocumentIngestionError(
            "Cortana: The specified file could not be accessed."
        ) from error

    try:
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise DocumentIngestionError(
                "Cortana: Only regular files can be ingested."
            )

        source_size_bytes = int(file_stat.st_size)
        if source_size_bytes > MAX_DOCUMENT_SOURCE_BYTES:
            raise DocumentIngestionError(
                "Cortana: The source file exceeds the maximum size of "
                f"{MAX_DOCUMENT_SOURCE_BYTES} bytes."
            )

        file_bytes = os.read(file_descriptor, source_size_bytes + 1)
        if len(file_bytes) > MAX_DOCUMENT_SOURCE_BYTES:
            raise DocumentIngestionError(
                "Cortana: The source file exceeds the maximum size of "
                f"{MAX_DOCUMENT_SOURCE_BYTES} bytes."
            )
    finally:
        os.close(file_descriptor)

    return file_bytes, filename, extension, len(file_bytes)


def _supported_extensions_display() -> str:
    """Return a stable display string for supported document extensions."""
    return ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
