"""Tests for Milestone 14 Windows safe file-opening foundation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from src.config import PROCESS_FILE_TOOL_ISOLATION_ENABLED
from src.tool_process_safe_open import (
    SafeOpenError,
    SafeOpenUnavailableError,
    assert_file_tool_isolation_readiness,
    capture_windows_file_identity,
    create_windows_file_identity,
    read_bytes_from_safe_handle,
    safe_open_for_read,
    validate_windows_path_format,
)
from src.tool_registry import build_default_tool_registry


def _fake_win32_modules(
    *,
    attributes: int = 0,
    volume: int = 1,
    size_bytes: int = 4,
    index_high: int = 1,
    index_low: int = 2,
    handle_path: str = r"C:\Cases\sample.txt",
    read_data: bytes = b"",
) -> Any:
    class FakeHandle:
        pass

    class Win32File:
        @staticmethod
        def CreateFile(*_args: Any, **_kwargs: Any) -> FakeHandle:
            return FakeHandle()

        @staticmethod
        def GetFileInformationByHandle(_handle: Any) -> tuple[Any, ...]:
            return (
                attributes,
                0,
                0,
                0,
                volume,
                0,
                size_bytes,
                1,
                index_high,
                index_low,
            )

        @staticmethod
        def GetFinalPathNameByHandle(_handle: Any, _flags: int) -> str:
            return handle_path

        @staticmethod
        def ReadFile(_handle: Any, _size: int) -> tuple[int, bytes]:
            return 0, read_data

    class Win32Con:
        GENERIC_READ = 1
        FILE_SHARE_READ = 1
        OPEN_EXISTING = 3
        FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000

    class Win32Api:
        @staticmethod
        def CloseHandle(_handle: Any) -> None:
            return None

    return Win32File(), Win32Con(), Win32Api()


def test_file_tool_isolation_flag_defaults_false() -> None:
    assert PROCESS_FILE_TOOL_ISOLATION_ENABLED is False


def test_non_file_tools_remain_ineligible_for_file_isolation() -> None:
    registry = build_default_tool_registry()
    assert registry.require("file-sha256").process_isolation == "eligible"
    assert registry.require("compare-sha256").process_isolation == "eligible"
    assert registry.require("text-search").process_isolation == "eligible"
    assert registry.require("incident-summary").process_isolation == "prohibited"


def test_forbidden_path_formats() -> None:
    with pytest.raises(SafeOpenError):
        validate_windows_path_format(r"\\server\share\file.txt")
    with pytest.raises(SafeOpenError):
        validate_windows_path_format(r"\\.\C:\file.txt")
    with pytest.raises(SafeOpenError):
        validate_windows_path_format(r"C:\temp\file.txt:stream")
    with pytest.raises(SafeOpenError):
        validate_windows_path_format(r"C:\temp\NUL")
    with pytest.raises(SafeOpenError):
        validate_windows_path_format(r"C:\temp\COM1.txt")
    with pytest.raises(SafeOpenError):
        validate_windows_path_format("relative\\file.txt")


@pytest.mark.parametrize(
    "path_value",
    [
        r"C:\temp\CON ",
        r"C:\temp\CON .txt",
        r"C:\temp\NUL.",
        r"C:\temp\PRN .log",
        r"C:\temp\AUX...",
        r"C:\temp\COM1 ",
        r"C:\temp\COM9.txt ",
        r"C:\temp\LPT1.",
        r"C:\temp\LPT9 .txt",
        r"C:\temp\con",
        r"C:\temp\nul.",
        r"C:\temp\Com1.txt",
        r"C:\temp\lpt9 .TXT",
    ],
)
def test_reserved_names_with_trailing_spaces_dots_and_extensions_rejected(
    path_value: str,
) -> None:
    with pytest.raises(SafeOpenError, match="Reserved device"):
        validate_windows_path_format(path_value)


@pytest.mark.parametrize(
    "path_value",
    [
        r"C:\authorized\NUL\file.txt",
        r"C:\CON\file.txt",
        r"C:\authorized\COM1\data.bin",
        r"C:\authorized\LPT9.\file.txt",
        r"C:\authorized\PRN .dir\file.txt",
        r"C:\authorized\con\file.txt",
        r"C:\authorized\Com9.txt\nested\file.bin",
    ],
)
def test_reserved_names_in_intermediate_segments_rejected(path_value: str) -> None:
    with pytest.raises(SafeOpenError, match="Reserved device"):
        validate_windows_path_format(path_value)


@pytest.mark.parametrize(
    "path_value",
    [
        r"C:\Cases\sample.txt",
        r"C:\authorized\reports\file.txt",
        r"C:\authorized\config.ini",
        r"D:\data\notes.md",
        r"C:\authorized\CONSOLE\file.txt",
        r"C:\authorized\NULLABLE\file.txt",
        r"C:\authorized\COM10\file.txt",
        r"C:\authorized\LPT10\file.txt",
    ],
)
def test_ordinary_valid_path_segments_accepted(path_value: str) -> None:
    assert validate_windows_path_format(path_value) == path_value.replace("/", "\\")


def test_identity_model_validation() -> None:
    identity = create_windows_file_identity(
        canonical_path=r"C:\Cases\sample.txt",
        volume_serial_number=1,
        file_index_high=2,
        file_index_low=3,
        size_bytes=10,
        last_write_time_filetime=100,
    )
    assert identity.file_index_low == 3
    with pytest.raises(SafeOpenError):
        create_windows_file_identity(
            canonical_path=r"\\server\share\a.txt",
            volume_serial_number=1,
            file_index_high=2,
            file_index_low=3,
            size_bytes=10,
            last_write_time_filetime=100,
        )


def test_identity_mismatch_always_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = create_windows_file_identity(
        canonical_path=r"C:\Cases\sample.txt",
        volume_serial_number=111,
        file_index_high=1,
        file_index_low=2,
        size_bytes=4,
        last_write_time_filetime=9,
    )

    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32_modules(
            volume=222,
            size_bytes=4,
            index_high=1,
            index_low=2,
            handle_path=r"C:\Cases\sample.txt",
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    with pytest.raises(SafeOpenError, match="identity"):
        safe_open_for_read(
            r"C:\Cases\sample.txt",
            expected_identity=expected,
        )


def test_reparse_point_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32_modules(
            attributes=0x400,
            handle_path=r"C:\Cases\link.txt",
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    with pytest.raises(SafeOpenError, match="Reparse"):
        safe_open_for_read(r"C:\Cases\link.txt", expected_identity=None)


def test_post_open_root_uses_handle_derived_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authorized-root post-open check uses GetFinalPathNameByHandle, not input."""
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32_modules(
            handle_path=r"\\?\C:\Outside\escaped.txt",
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    with pytest.raises(SafeOpenError, match="authorized root"):
        safe_open_for_read(
            r"C:\Cases\sample.txt",
            expected_identity=None,
            authorized_root=r"C:\Cases",
        )


def test_post_open_root_accepts_handle_path_under_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32_modules(
            handle_path=r"\\?\C:\Cases\sample.txt",
            size_bytes=0,
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    opened = safe_open_for_read(
        r"C:\Cases\sample.txt",
        expected_identity=None,
        authorized_root=r"C:\Cases",
    )
    try:
        assert opened.identity.canonical_path == r"C:\Cases\sample.txt"
    finally:
        opened.close()


def test_handle_path_extended_prefix_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32_modules(
            handle_path=r"\\?\C:\Cases\sample.txt",
            size_bytes=0,
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    opened = safe_open_for_read(r"C:\Cases\sample.txt", expected_identity=None)
    try:
        assert opened.identity.canonical_path == r"C:\Cases\sample.txt"
    finally:
        opened.close()


def test_handle_path_unc_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32_modules(
            handle_path=r"\\?\UNC\server\share\file.txt",
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    with pytest.raises(SafeOpenError):
        safe_open_for_read(r"C:\Cases\sample.txt", expected_identity=None)


def test_readiness_flag_fails_closed_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open.PROCESS_FILE_TOOL_ISOLATION_ENABLED",
        True,
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "posix")
    with pytest.raises(SafeOpenUnavailableError):
        assert_file_tool_isolation_readiness()


def test_source_does_not_use_plain_open_as_boundary() -> None:
    source = Path("src/tool_process_safe_open.py").read_text(encoding="utf-8")
    assert "CreateFile" in source
    assert "GetFinalPathNameByHandle" in source
    assert "_FILE_FLAG_OPEN_REPARSE_POINT" in source
    # Ensure plain open helpers are not used as executable calls.
    assert "os.open(" not in source.replace("os.open()", "")
    assert "builtins.open" not in source


@pytest.mark.skipif(os.name != "nt", reason="Windows safe-open integration")
def test_windows_capture_and_open_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("hello-safe-open", encoding="utf-8")
    # validate_windows_path_format requires drive letter path.
    path_value = str(target.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")
    identity = capture_windows_file_identity(path_value)
    assert identity.size_bytes == len("hello-safe-open")
    opened = safe_open_for_read(path_value, expected_identity=identity)
    try:
        data = read_bytes_from_safe_handle(opened, max_bytes=1024)
        assert data == b"hello-safe-open"
    finally:
        opened.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows TOCTOU integration")
def test_windows_toctou_replacement_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original.txt"
    replacement = tmp_path / "replacement.txt"
    original.write_text("original-bytes", encoding="utf-8")
    replacement.write_text("replacement-bytes", encoding="utf-8")
    original_path = str(original.resolve())
    if not (len(original_path) >= 3 and original_path[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")

    identity = capture_windows_file_identity(original_path)

    # Replace the path after identity capture (delete + recreate different file).
    original.unlink()
    replacement.replace(original)

    with pytest.raises(SafeOpenError):
        opened = safe_open_for_read(original_path, expected_identity=identity)
        opened.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows symlink integration")
def test_windows_symlink_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("target", encoding="utf-8")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on this host")
    path_value = str(link.resolve(strict=False))
    # Use the link path string before resolve-follows.
    link_path = str(link)
    if not (len(link_path) >= 3 and link_path[1] == ":"):
        # tmp_path should be drive-letter on Windows.
        link_path = str(link.resolve())
    with pytest.raises(SafeOpenError):
        safe_open_for_read(str(link), expected_identity=None)


def test_no_tool_dispatch_in_safe_open_module() -> None:
    source = Path("src/tool_process_safe_open.py").read_text(encoding="utf-8")
    assert "CHILD_DISPATCH" not in source
    assert "ToolRegistry" not in source
    assert "subprocess" not in source


def test_safe_open_callers_are_allowlisted() -> None:
    """Only reviewed Milestone 15/16 modules may import the safe-open foundation."""
    allowed = {
        Path("src") / "tool_process_file_auth.py",
        Path("src") / "tool_process_file_tools.py",
        Path("src") / "tool_process_adapter.py",
        Path("src") / "tool_process_text_search.py",
    }
    callers: list[Path] = []
    for path in Path("src").rglob("*.py"):
        if path.name == "tool_process_safe_open.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "tool_process_safe_open" in text or "safe_open_for_read" in text:
            callers.append(path)
    assert set(callers) == allowed
