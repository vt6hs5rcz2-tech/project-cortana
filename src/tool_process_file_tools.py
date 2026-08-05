"""Child-side process-isolated file integrity and text-search callables.

These callables use the Milestone 14 safe-open foundation only. They never
reopen by path after the verified handle is obtained and never buffer the full
file for hashing or text search.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config import MAX_TOOL_FILE_BYTES
from src.tool_common import validate_sha256_digest
from src.tool_process_file_auth import (
    FileAuthorization,
    safe_filename_from_authorization,
    validate_file_authorization,
    validate_file_tool_process_parameters,
)
from src.tool_process_safe_open import (
    FileChangedDuringRead,
    FileTooLarge,
    IdentityMismatch,
    SafeOpenError,
    hash_sha256_from_safe_handle,
    safe_open_for_read,
)
from src.tool_process_text_search import stream_text_search_from_safe_handle


def run_file_sha256(
    parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash one parent-authorized file through a verified Windows handle."""
    cleaned = validate_file_tool_process_parameters(
        "impl_file_sha256",
        dict(parameters),
    )
    authorization = validate_file_authorization(cleaned["file_authorization"])
    result = _hash_authorized_file(authorization)
    return {
        "sha256_hex": result["sha256_hex"],
        "size_bytes": result["size_bytes"],
        "filename_only": result["filename_only"],
    }


def run_compare_sha256(
    parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one parent-authorized file digest to an expected SHA-256."""
    cleaned = validate_file_tool_process_parameters(
        "impl_compare_sha256",
        dict(parameters),
    )
    authorization = validate_file_authorization(cleaned["file_authorization"])
    expected = validate_sha256_digest(str(cleaned["expected_sha256"]))
    result = _hash_authorized_file(authorization)
    digest = str(result["sha256_hex"])
    return {
        "sha256_hex": digest,
        "expected_sha256": expected,
        "matches": digest == expected,
        "size_bytes": result["size_bytes"],
        "filename_only": result["filename_only"],
    }


def run_text_search(
    parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Search one parent-authorized file through a verified Windows handle."""
    cleaned = validate_file_tool_process_parameters(
        "impl_text_search",
        dict(parameters),
    )
    authorization = validate_file_authorization(cleaned["file_authorization"])
    query = str(cleaned["query"])
    max_matches = int(cleaned["max_matches"])
    opened = None
    try:
        opened = safe_open_for_read(
            authorization.canonical_path,
            expected_identity=authorization.to_windows_identity(),
            authorized_root=authorization.authorized_root,
        )
        searched = stream_text_search_from_safe_handle(
            opened,
            query=query,
            max_matches=max_matches,
            max_bytes=MAX_TOOL_FILE_BYTES,
        )
        return {
            "filename": safe_filename_from_authorization(authorization),
            "match_count": searched.match_count,
            "truncated": searched.truncated,
            "matches": [
                {
                    "line_number": item.line_number,
                    "preview": item.preview,
                }
                for item in searched.matches
            ],
        }
    except (IdentityMismatch, FileChangedDuringRead, FileTooLarge):
        raise
    except SafeOpenError:
        raise
    finally:
        if opened is not None:
            opened.close()


def _hash_authorized_file(authorization: FileAuthorization) -> dict[str, Any]:
    """Open once with safe_open_for_read and stream-hash through that handle."""
    opened = None
    try:
        opened = safe_open_for_read(
            authorization.canonical_path,
            expected_identity=authorization.to_windows_identity(),
            authorized_root=authorization.authorized_root,
        )
        hashed = hash_sha256_from_safe_handle(
            opened,
            max_bytes=MAX_TOOL_FILE_BYTES,
        )
        return {
            "sha256_hex": hashed.sha256_hex,
            "size_bytes": hashed.size_bytes,
            "filename_only": safe_filename_from_authorization(authorization),
        }
    except (IdentityMismatch, FileChangedDuringRead, FileTooLarge):
        raise
    except SafeOpenError:
        raise
    finally:
        if opened is not None:
            opened.close()
