"""Windows-native safe file-opening foundation for future process-isolated file tools.

This module is foundation-only in Milestone 14. It is not wired into tool
eligibility for file-sha256, compare-sha256, or any other file-touching tool.
Plain ``open()`` / ``os.open()`` are not treated as the secure boundary here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from src.config import PROCESS_FILE_TOOL_ISOLATION_ENABLED
from src.tool_common import ToolValidationError

# Windows FILE_ATTRIBUTE_REPARSE_POINT / FILE_FLAG_OPEN_REPARSE_POINT
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_DEVICE = 0x40
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000

_RESERVED_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)

_DRIVE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


class SafeOpenError(ToolValidationError):
    """Raised when a Windows safe-open validation or open fails closed."""


class SafeOpenUnavailableError(SafeOpenError):
    """Raised when the Windows safe-open path cannot be used."""


@dataclass(frozen=True)
class WindowsFileIdentity:
    """Immutable parent-authorized Windows file identity."""

    canonical_path: str
    volume_serial_number: int
    file_index_high: int
    file_index_low: int
    size_bytes: int
    last_write_time_filetime: int


@dataclass(frozen=True)
class SafeOpenHandle:
    """Verified open handle plus identity; caller must close deterministically."""

    handle: Any
    identity: WindowsFileIdentity

    def close(self) -> None:
        """Close the underlying Windows handle."""
        close_windows_handle(self.handle)


def assert_file_tool_isolation_readiness() -> None:
    """Fail closed when the file-tool isolation readiness flag is enabled but unavailable.

    This does not change registry eligibility. It only validates that the
    foundation can load on Windows with pywin32 when the readiness flag is on.
    """
    if not PROCESS_FILE_TOOL_ISOLATION_ENABLED:
        return
    if os.name != "nt":
        raise SafeOpenUnavailableError(
            "Process file-tool isolation readiness requires Windows."
        )
    _require_win32_modules()


def validate_windows_path_format(path_value: str) -> str:
    """Reject forbidden Windows path forms before any open attempt."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise SafeOpenError("A file path is required.")
    cleaned = path_value.strip()
    if "\x00" in cleaned:
        raise SafeOpenError("Path contains a null byte.")

    normalized = cleaned.replace("/", "\\")
    if normalized.startswith("\\\\"):
        # Reject UNC and device namespaces. Extended \\?\C:\ paths are also
        # rejected in v1 to keep the allowlist narrow and reviewable.
        raise SafeOpenError("UNC and device namespace paths are not permitted.")
    if not _DRIVE_PATH_PATTERN.match(normalized):
        raise SafeOpenError("Only local drive-letter paths are permitted.")

    # Alternate data stream: colon after the drive letter colon.
    # The drive prefix itself (for example ``C:``) is not an ADS marker.
    if ":" in normalized[2:]:
        raise SafeOpenError("Alternate data streams are not permitted.")

    for segment in _windows_path_segments(normalized):
        if _is_reserved_device_segment(segment):
            raise SafeOpenError("Reserved device names are not permitted.")
    return normalized


def _windows_path_segments(normalized_drive_path: str) -> list[str]:
    """Return meaningful path segments after the drive prefix.

    The drive prefix (``C:``) is excluded. Empty components from repeated
    separators are rejected.
    """
    remainder = normalized_drive_path[2:]
    if remainder.startswith("\\"):
        remainder = remainder[1:]
    if not remainder:
        return []
    segments = remainder.split("\\")
    if any(segment == "" for segment in segments):
        raise SafeOpenError("Empty path components are not permitted.")
    return segments


def _is_reserved_device_segment(segment: str) -> bool:
    """Return whether one path segment is a Windows reserved device name.

    Windows treats reserved names case-insensitively and may normalize trailing
    spaces/dots. Names such as ``CON``, ``CON ``, ``CON.txt``, ``CON .txt``,
    and ``NUL.`` must all be rejected.
    """
    # Strip only trailing spaces/dots that Windows normalizes away. Do not strip
    # ordinary leading whitespace from otherwise valid names.
    cleaned = segment.rstrip(" .")
    if not cleaned:
        return False
    base = cleaned.split(".", 1)[0].rstrip(" .").upper()
    return base in _RESERVED_DEVICE_NAMES


def create_windows_file_identity(
    *,
    canonical_path: str,
    volume_serial_number: int,
    file_index_high: int,
    file_index_low: int,
    size_bytes: int,
    last_write_time_filetime: int,
) -> WindowsFileIdentity:
    """Create one validated immutable Windows file identity record."""
    path = validate_windows_path_format(canonical_path)
    for name, value in (
        ("volume_serial_number", volume_serial_number),
        ("file_index_high", file_index_high),
        ("file_index_low", file_index_low),
        ("size_bytes", size_bytes),
        ("last_write_time_filetime", last_write_time_filetime),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SafeOpenError(f"{name} must be an integer.")
        if value < 0:
            raise SafeOpenError(f"{name} cannot be negative.")
    return WindowsFileIdentity(
        canonical_path=path,
        volume_serial_number=volume_serial_number,
        file_index_high=file_index_high,
        file_index_low=file_index_low,
        size_bytes=size_bytes,
        last_write_time_filetime=last_write_time_filetime,
    )


def capture_windows_file_identity(path_value: str) -> WindowsFileIdentity:
    """Open a path with reparse-aware APIs and capture its Windows identity."""
    path = validate_windows_path_format(path_value)
    opened = safe_open_for_read(path, expected_identity=None)
    try:
        return opened.identity
    finally:
        opened.close()


def safe_open_for_read(
    path_value: str,
    *,
    expected_identity: WindowsFileIdentity | None,
    authorized_root: str | None = None,
) -> SafeOpenHandle:
    """Open a regular file without following reparse points and verify identity.

    When ``expected_identity`` is provided, the opened handle identity must match
    exactly. When ``authorized_root`` is provided, containment is checked before
    open using the caller path, then checked again after open using a path
    independently derived from the open handle via ``GetFinalPathNameByHandle``.
    Volume serial / file-index identity comparison remains the primary
    post-open object-identity guarantee.
    """
    path = validate_windows_path_format(path_value)
    root: str | None = None
    if authorized_root is not None:
        root = validate_windows_path_format(authorized_root)
        if not _path_is_under_root(path, root):
            raise SafeOpenError("Path is outside the authorized root.")

    win32file, win32con, _win32api = _require_win32_modules()
    handle: Any | None = None
    try:
        handle = win32file.CreateFile(
            path,
            win32con.GENERIC_READ,
            win32con.FILE_SHARE_READ,
            None,
            win32con.OPEN_EXISTING,
            _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        handle_path = _final_path_from_handle(handle)
        identity = _identity_from_handle(handle, canonical_path=handle_path)
        attributes = _file_attributes(handle)
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise SafeOpenError("Reparse points are not permitted.")
        if attributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise SafeOpenError("Directories are not permitted.")
        if attributes & _FILE_ATTRIBUTE_DEVICE:
            raise SafeOpenError("Device files are not permitted.")
        if expected_identity is not None and not _identities_match(
            identity,
            expected_identity,
        ):
            raise SafeOpenError("Opened file identity does not match authorization.")
        if root is not None and not _path_is_under_root(handle_path, root):
            raise SafeOpenError("Path is outside the authorized root.")
        result = SafeOpenHandle(handle=handle, identity=identity)
        handle = None
        return result
    except SafeOpenError:
        raise
    except Exception as error:
        raise SafeOpenError("Secure file open failed.") from error
    finally:
        if handle is not None:
            close_windows_handle(handle)


def read_bytes_from_safe_handle(
    opened: SafeOpenHandle,
    *,
    max_bytes: int,
) -> bytes:
    """Read bytes through a verified handle only."""
    if max_bytes < 1:
        raise SafeOpenError("max_bytes must be positive.")
    win32file, _win32con, _win32api = _require_win32_modules()
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            _error_code, data = win32file.ReadFile(opened.handle, 1024 * 1024)
            if not data:
                break
            total += len(data)
            if total > max_bytes:
                raise SafeOpenError("File exceeds the maximum permitted size.")
            chunks.append(bytes(data))
    except SafeOpenError:
        raise
    except Exception as error:
        raise SafeOpenError("Secure file read failed.") from error
    return b"".join(chunks)


def close_windows_handle(handle: Any) -> None:
    """Close a Windows handle without raising to callers."""
    if handle is None:
        return
    try:
        _win32file, _win32con, win32api = _require_win32_modules()
        win32api.CloseHandle(handle)
    except Exception:
        pass


def _require_win32_modules() -> tuple[Any, Any, Any]:
    if os.name != "nt":
        raise SafeOpenUnavailableError(
            "Windows safe-open requires a Windows platform."
        )
    try:
        import win32api  # type: ignore[import-untyped]
        import win32con  # type: ignore[import-untyped]
        import win32file  # type: ignore[import-untyped]
    except ImportError as error:
        raise SafeOpenUnavailableError(
            "Windows safe-open requires pywin32."
        ) from error
    return win32file, win32con, win32api


def _identity_from_handle(handle: Any, *, canonical_path: str) -> WindowsFileIdentity:
    win32file, _win32con, _win32api = _require_win32_modules()
    try:
        info = win32file.GetFileInformationByHandle(handle)
    except Exception as error:
        raise SafeOpenError("Unable to read Windows file identity.") from error
    # pywin32 returns:
    # (attrs, create, access, write, volume, sizeHigh, sizeLow, links, indexHigh, indexLow)
    (
        _attrs,
        _created,
        _accessed,
        written,
        volume,
        size_high,
        size_low,
        _links,
        index_high,
        index_low,
    ) = info
    size_bytes = (int(size_high) << 32) | int(size_low)
    write_filetime = _filetime_to_int(written)
    return create_windows_file_identity(
        canonical_path=canonical_path,
        volume_serial_number=int(volume),
        file_index_high=int(index_high),
        file_index_low=int(index_low),
        size_bytes=size_bytes,
        last_write_time_filetime=write_filetime,
    )


def _final_path_from_handle(handle: Any) -> str:
    """Derive and normalize the opened path from the Windows file handle."""
    win32file, _win32con, _win32api = _require_win32_modules()
    try:
        raw_path = win32file.GetFinalPathNameByHandle(handle, 0)
    except Exception as error:
        raise SafeOpenError("Unable to derive opened file path from handle.") from error
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SafeOpenError("Unable to derive opened file path from handle.")
    return _normalize_handle_path(raw_path)


def _normalize_handle_path(raw_path: str) -> str:
    """Normalize GetFinalPathNameByHandle output to a local drive-letter path."""
    normalized = raw_path.strip().replace("/", "\\")
    if normalized.startswith("\\\\?\\UNC\\") or normalized.startswith("//?/UNC/"):
        raise SafeOpenError("UNC and device namespace paths are not permitted.")
    if normalized.startswith("\\\\?\\"):
        normalized = normalized[4:]
    elif normalized.startswith("\\\\.\\"):
        raise SafeOpenError("UNC and device namespace paths are not permitted.")
    if normalized.startswith("\\\\"):
        raise SafeOpenError("UNC and device namespace paths are not permitted.")
    return validate_windows_path_format(normalized)


def _file_attributes(handle: Any) -> int:
    win32file, _win32con, _win32api = _require_win32_modules()
    info = win32file.GetFileInformationByHandle(handle)
    return int(info[0])


def _filetime_to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    # pywintypes.Time objects expose .timestamp() in seconds.
    try:
        return int(value.timestamp() * 10_000_000)
    except Exception:
        return 0


def _identities_match(
    left: WindowsFileIdentity,
    right: WindowsFileIdentity,
) -> bool:
    return (
        left.volume_serial_number == right.volume_serial_number
        and left.file_index_high == right.file_index_high
        and left.file_index_low == right.file_index_low
        and left.size_bytes == right.size_bytes
    )


def _path_is_under_root(path_value: str, root_value: str) -> bool:
    """Return whether path_value is under root_value using Windows path rules."""
    path_norm = path_value.replace("/", "\\").rstrip("\\").casefold()
    root_norm = root_value.replace("/", "\\").rstrip("\\").casefold()
    if path_norm == root_norm:
        return True
    return path_norm.startswith(root_norm + "\\")
