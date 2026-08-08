"""Local evidence file storage with streaming SHA-256 verification."""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path
from typing import Protocol
from uuid import UUID

from src.config import MAX_EVIDENCE_SOURCE_BYTES
from src.document_ingestion import strip_path_argument_quotes
from src.security_common import InvalidSecurityIdError, validate_security_id

logger = logging.getLogger("ProjectCortana")

HASH_CHUNK_SIZE = 1024 * 1024


class EvidenceStoreError(RuntimeError):
    """Raised when evidence bytes cannot be stored or verified safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class EvidenceStore(Protocol):
    """Protocol for local evidence byte storage."""

    def copy_from_path(
        self,
        path_argument: str,
        *,
        evidence_id: str,
    ) -> tuple[str, int, str]:
        """Copy a source file into evidence storage.

        Returns ``(sha256_hex, source_size_bytes, original_filename)``.
        """

    def verify_stored_hash(self, evidence_id: str, expected_sha256: str) -> str:
        """Stream-hash a stored evidence copy and compare to the expected digest.

        Returns ``\"match\"``, ``\"mismatch\"``, or ``\"missing\"``.
        """

    def has_stored_copy(self, evidence_id: str) -> bool:
        """Return whether a stored evidence object exists for the ID."""

    def resolve_stored_evidence_path(self, evidence_id: str) -> Path:
        """Return the absolute path of one stored evidence object for tool use.

        Intended only for authorized defensive tool operations. Callers must
        authorize the path with existing scope checks and must not display,
        audit, or persist the path outside ordinary tool-request machinery.
        """


class LocalEvidenceStore:
    """User-local evidence directory that stores opaque byte copies only."""

    def __init__(self, directory_path: Path) -> None:
        self._directory_path = directory_path

    @property
    def directory_path(self) -> Path:
        """Return the configured evidence directory path."""
        return self._directory_path

    def copy_from_path(
        self,
        path_argument: str,
        *,
        evidence_id: str,
    ) -> tuple[str, int, str]:
        """Copy bytes from a regular file into exclusive evidence storage."""
        cleaned_argument = strip_path_argument_quotes(path_argument)
        if not cleaned_argument.strip():
            raise EvidenceStoreError(
                "Cortana: Please provide a file path. "
                "Usage: /evidence-register <path> | <title> | <description>"
            )

        canonical_id = _canonical_evidence_id(evidence_id)
        destination = self._object_path(canonical_id)
        temporary_path = self._temporary_path(canonical_id)

        if destination.exists():
            raise EvidenceStoreError(
                "Cortana: An evidence object with that ID already exists in storage."
            )

        source_path = Path(cleaned_argument)
        original_filename = source_path.name
        self._directory_path.mkdir(parents=True, exist_ok=True)

        try:
            copy_hash, copy_size = self._stream_copy_regular_file(
                source_path,
                temporary_path,
            )
            self._finalize_exclusive(temporary_path, destination, canonical_id)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        verified = self.verify_stored_hash(canonical_id, copy_hash)
        if verified != "match":
            destination.unlink(missing_ok=True)
            raise EvidenceStoreError(
                "Cortana: Evidence copy verification failed before storage was finalized."
            )

        logger.info(
            "Evidence copied evidence_id=%s size_bytes=%s result=stored",
            canonical_id,
            copy_size,
        )
        return copy_hash, copy_size, original_filename

    def verify_stored_hash(self, evidence_id: str, expected_sha256: str) -> str:
        """Verify a stored evidence object against the recorded SHA-256 digest."""
        canonical_id = _canonical_evidence_id(evidence_id)
        path = self._object_path(canonical_id)

        try:
            if not path.exists():
                return "missing"
            if path.is_symlink():
                raise EvidenceStoreError(
                    "Cortana: Stored evidence object is not a regular file."
                )
        except OSError as error:
            logger.error(
                "Evidence verify path check failed evidence_id=%s error_type=%s",
                canonical_id,
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: Stored evidence could not be verified safely."
            ) from error

        try:
            actual_hash, _size = self._stream_hash_open_path(path)
        except EvidenceStoreError:
            raise
        except OSError as error:
            logger.error(
                "Evidence verify hash failed evidence_id=%s error_type=%s",
                canonical_id,
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: Stored evidence could not be verified safely."
            ) from error

        if actual_hash == expected_sha256.strip().lower():
            return "match"
        return "mismatch"

    def has_stored_copy(self, evidence_id: str) -> bool:
        """Return whether a stored evidence object exists for the ID."""
        path = self._object_path(_canonical_evidence_id(evidence_id))
        try:
            return path.exists() and not path.is_symlink() and path.is_file()
        except OSError:
            return False

    def resolve_stored_evidence_path(self, evidence_id: str) -> Path:
        """Return a safe absolute path to one existing stored evidence object."""
        canonical_id = _canonical_evidence_id(evidence_id)
        path = self._object_path(canonical_id)
        try:
            if path.is_symlink():
                raise EvidenceStoreError(
                    "Cortana: Stored evidence object is not a regular file."
                )
            if not path.exists():
                raise EvidenceStoreError(
                    "Cortana: Stored evidence object was not found."
                )
            if not path.is_file():
                raise EvidenceStoreError(
                    "Cortana: Stored evidence object is not a regular file."
                )
            resolved = path.resolve(strict=False)
            if resolved.is_symlink():
                raise EvidenceStoreError(
                    "Cortana: Stored evidence object is not a regular file."
                )
            if not resolved.is_file():
                raise EvidenceStoreError(
                    "Cortana: Stored evidence object is not a regular file."
                )
            return resolved
        except EvidenceStoreError:
            raise
        except OSError as error:
            logger.error(
                "Evidence resolve path failed evidence_id=%s error_type=%s",
                canonical_id,
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: Stored evidence could not be resolved safely."
            ) from error

    def _object_path(self, evidence_id: str) -> Path:
        """Return the safe internal object path derived only from the evidence ID."""
        return self._directory_path / f"{evidence_id}.bin"

    def _temporary_path(self, evidence_id: str) -> Path:
        """Return a same-directory temporary path for an incomplete copy."""
        return self._directory_path / f".{evidence_id}.partial"

    def _finalize_exclusive(
        self,
        temporary_path: Path,
        destination: Path,
        evidence_id: str,
    ) -> None:
        """Move a completed temporary copy into place without silent overwrite."""
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            placeholder_fd = os.open(os.fspath(destination), flags, 0o600)
            os.close(placeholder_fd)
        except FileExistsError as error:
            raise EvidenceStoreError(
                "Cortana: An evidence object with that ID already exists in storage."
            ) from error
        except OSError as error:
            logger.error(
                "Evidence exclusive create failed evidence_id=%s error_type=%s",
                evidence_id,
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: Evidence could not be stored safely."
            ) from error

        try:
            os.replace(temporary_path, destination)
        except OSError as error:
            destination.unlink(missing_ok=True)
            logger.error(
                "Evidence copy finalize failed evidence_id=%s error_type=%s",
                evidence_id,
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: Evidence could not be stored safely."
            ) from error

    def _stream_copy_regular_file(
        self,
        source_path: Path,
        temporary_path: Path,
    ) -> tuple[str, int]:
        """Copy source bytes to a temporary path while streaming SHA-256."""
        self._reject_symlink_or_missing(source_path)
        temporary_path.unlink(missing_ok=True)

        digest = hashlib.sha256()
        total = 0
        source_fd = self._open_regular_readonly(source_path)
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            dest_fd = os.open(os.fspath(temporary_path), flags, 0o600)
            try:
                while True:
                    chunk = os.read(source_fd, HASH_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_EVIDENCE_SOURCE_BYTES:
                        raise EvidenceStoreError(
                            "Cortana: The evidence source exceeds the maximum size of "
                            f"{MAX_EVIDENCE_SOURCE_BYTES} bytes."
                        )
                    digest.update(chunk)
                    os.write(dest_fd, chunk)
                os.fsync(dest_fd)
            finally:
                os.close(dest_fd)
        except EvidenceStoreError:
            temporary_path.unlink(missing_ok=True)
            raise
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            logger.error(
                "Evidence copy failed error_type=%s",
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: Evidence could not be stored safely."
            ) from error
        finally:
            os.close(source_fd)

        return digest.hexdigest(), total

    def _stream_hash_open_path(self, path: Path) -> tuple[str, int]:
        """Stream SHA-256 for an already-validated path."""
        digest = hashlib.sha256()
        total = 0
        file_descriptor = self._open_regular_readonly(path)
        try:
            while True:
                chunk = os.read(file_descriptor, HASH_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_EVIDENCE_SOURCE_BYTES:
                    raise EvidenceStoreError(
                        "Cortana: The evidence source exceeds the maximum size of "
                        f"{MAX_EVIDENCE_SOURCE_BYTES} bytes."
                    )
                digest.update(chunk)
        finally:
            os.close(file_descriptor)
        return digest.hexdigest(), total

    def _open_regular_readonly(self, path: Path) -> int:
        """Open a path read-only and ensure the descriptor refers to a regular file."""
        try:
            file_descriptor = os.open(
                os.fspath(path),
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError as error:
            raise EvidenceStoreError(
                "Cortana: The specified evidence file could not be found."
            ) from error
        except OSError as error:
            logger.error(
                "Evidence open failed error_type=%s",
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: The specified evidence file could not be accessed."
            ) from error

        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                os.close(file_descriptor)
                raise EvidenceStoreError(
                    "Cortana: Only regular files can be registered as evidence."
                )
            if file_stat.st_size > MAX_EVIDENCE_SOURCE_BYTES:
                os.close(file_descriptor)
                raise EvidenceStoreError(
                    "Cortana: The evidence source exceeds the maximum size of "
                    f"{MAX_EVIDENCE_SOURCE_BYTES} bytes."
                )
        except EvidenceStoreError:
            raise
        except OSError as error:
            os.close(file_descriptor)
            logger.error(
                "Evidence fstat failed error_type=%s",
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: The specified evidence file could not be accessed."
            ) from error

        return file_descriptor

    def _reject_symlink_or_missing(self, path: Path) -> None:
        """Reject symbolic links/reparse points and missing paths without leaking paths."""
        try:
            if path.is_symlink():
                raise EvidenceStoreError(
                    "Cortana: Symbolic links and reparse points cannot be registered "
                    "as evidence."
                )
            if not path.exists():
                raise EvidenceStoreError(
                    "Cortana: The specified evidence file could not be found."
                )
            if path.is_dir():
                raise EvidenceStoreError(
                    "Cortana: Directories cannot be registered as evidence. "
                    "Provide a single file path."
                )
        except EvidenceStoreError:
            raise
        except OSError as error:
            logger.error(
                "Evidence path inspection failed error_type=%s",
                type(error).__name__,
            )
            raise EvidenceStoreError(
                "Cortana: The specified evidence file could not be accessed."
            ) from error


def _canonical_evidence_id(evidence_id: str) -> str:
    """Return a canonical UUID evidence ID for internal filenames."""
    try:
        return validate_security_id(evidence_id, field_name="Evidence ID")
    except InvalidSecurityIdError as error:
        raise EvidenceStoreError(
            "Cortana: Evidence ID must be a valid UUID."
        ) from error


def evidence_storage_filename(evidence_id: str) -> str:
    """Return the safe internal filename for an evidence ID (for tests only)."""
    canonical = str(UUID(validate_security_id(evidence_id, field_name="Evidence ID")))
    return f"{canonical}.bin"
