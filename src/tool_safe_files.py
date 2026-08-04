"""Safe read-only local file helpers for defensive tools."""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path

from src.config import MAX_TOOL_FILE_BYTES
from src.document_ingestion import strip_path_argument_quotes
from src.tool_common import ToolValidationError

logger = logging.getLogger("ProjectCortana")

HASH_CHUNK_SIZE = 1024 * 1024


class SafeFileError(ToolValidationError):
    """Raised when a file target cannot be accessed safely."""


def resolve_regular_file(path_argument: str, *, max_bytes: int = MAX_TOOL_FILE_BYTES) -> Path:
    """Validate and return a resolved regular-file path without following symlinks."""
    cleaned = strip_path_argument_quotes(path_argument).strip()
    if not cleaned:
        raise SafeFileError("A file path is required.")

    path = Path(cleaned)
    _reject_symlink_or_invalid(path)

    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        logger.error(
            "Safe file resolve failed error_type=%s",
            type(error).__name__,
        )
        raise SafeFileError("The specified file could not be accessed.") from error

    _reject_symlink_or_invalid(resolved)
    _assert_regular_and_sized(resolved, max_bytes=max_bytes)
    return resolved


def stream_sha256(
    path_argument: str,
    *,
    max_bytes: int = MAX_TOOL_FILE_BYTES,
) -> tuple[str, int, str]:
    """Stream SHA-256 for a regular file.

    Returns ``(sha256_hex, size_bytes, filename_only)``.
    """
    path = resolve_regular_file(path_argument, max_bytes=max_bytes)
    digest = hashlib.sha256()
    total = 0
    file_descriptor = _open_regular_readonly(path, max_bytes=max_bytes)
    try:
        while True:
            chunk = os.read(file_descriptor, HASH_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SafeFileError(
                    f"The file exceeds the maximum size of {max_bytes} bytes."
                )
            digest.update(chunk)
    finally:
        os.close(file_descriptor)

    return digest.hexdigest(), total, path.name


def read_text_lines(
    path_argument: str,
    *,
    max_bytes: int = MAX_TOOL_FILE_BYTES,
) -> tuple[list[str], str]:
    """Read UTF-8 text lines from a regular file with controlled decode errors.

    Returns ``(lines, filename_only)``.
    """
    path = resolve_regular_file(path_argument, max_bytes=max_bytes)
    file_descriptor = _open_regular_readonly(path, max_bytes=max_bytes)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_descriptor, HASH_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SafeFileError(
                    f"The file exceeds the maximum size of {max_bytes} bytes."
                )
            chunks.append(chunk)
    finally:
        os.close(file_descriptor)

    text = b"".join(chunks).decode("utf-8", errors="replace")
    return text.splitlines(), path.name


def filename_only(path_argument: str) -> str:
    """Return only the final path segment for display."""
    cleaned = strip_path_argument_quotes(path_argument).strip().replace("\\", "/")
    if not cleaned:
        return "[filename]"
    return cleaned.rstrip("/").rsplit("/", 1)[-1]


def _reject_symlink_or_invalid(path: Path) -> None:
    """Reject symbolic links, directories, and missing paths."""
    try:
        if path.is_symlink():
            raise SafeFileError(
                "Symbolic links and reparse points cannot be used as tool targets."
            )
        if not path.exists():
            raise SafeFileError("The specified file could not be found.")
        if path.is_dir():
            raise SafeFileError("Directories cannot be used as tool targets.")
    except SafeFileError:
        raise
    except OSError as error:
        logger.error(
            "Safe file path inspection failed error_type=%s",
            type(error).__name__,
        )
        raise SafeFileError("The specified file could not be accessed.") from error


def _assert_regular_and_sized(path: Path, *, max_bytes: int) -> None:
    """Ensure the path refers to a regular file within size limits."""
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        logger.error(
            "Safe file stat failed error_type=%s",
            type(error).__name__,
        )
        raise SafeFileError("The specified file could not be accessed.") from error

    if not stat.S_ISREG(file_stat.st_mode):
        raise SafeFileError("Only regular files can be used as tool targets.")
    if file_stat.st_size > max_bytes:
        raise SafeFileError(
            f"The file exceeds the maximum size of {max_bytes} bytes."
        )


def _open_regular_readonly(path: Path, *, max_bytes: int) -> int:
    """Open a path read-only and ensure the descriptor refers to a regular file."""
    try:
        file_descriptor = os.open(
            os.fspath(path),
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as error:
        raise SafeFileError("The specified file could not be found.") from error
    except OSError as error:
        logger.error(
            "Safe file open failed error_type=%s",
            type(error).__name__,
        )
        raise SafeFileError("The specified file could not be accessed.") from error

    try:
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(file_descriptor)
            raise SafeFileError("Only regular files can be used as tool targets.")
        if file_stat.st_size > max_bytes:
            os.close(file_descriptor)
            raise SafeFileError(
                f"The file exceeds the maximum size of {max_bytes} bytes."
            )
    except SafeFileError:
        raise
    except OSError as error:
        os.close(file_descriptor)
        logger.error(
            "Safe file fstat failed error_type=%s",
            type(error).__name__,
        )
        raise SafeFileError("The specified file could not be accessed.") from error

    return file_descriptor
