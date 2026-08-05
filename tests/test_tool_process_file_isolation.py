"""Tests for Milestone 15 process-isolated file integrity tools."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.config import (
    MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS,
    MAX_PROCESS_IPC_REQUEST_BYTES,
    MAX_TOOL_FILE_BYTES,
    PROCESS_FILE_TOOL_ISOLATION_ENABLED,
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
    PROCESS_RESOURCE_LIMITS_ENABLED,
)
from src.tool_approval import create_tool_approval
from src.tool_audit import create_tool_audit_entry
from src.tool_definition import create_parameter_definition, create_tool_definition
from src.tool_executor import DefensiveToolExecutor, _select_execution_route
from src.tool_process_adapter import ToolProcessAdapter
from src.tool_process_common import (
    PROCESS_SAFE_FILE_IMPLEMENTATION_IDS,
    PROCESS_SAFE_IMPLEMENTATION_IDS,
    PROCESS_SAFE_TOOL_IDS,
    ToolProcessError,
)
from src.tool_process_envelope import create_process_execution_request
from src.tool_process_file_auth import (
    FILE_AUTHORIZATION_KEYS,
    validate_file_authorization,
    validate_file_tool_process_parameters,
)
from src.tool_process_file_tools import run_compare_sha256, run_file_sha256
from src.tool_process_runner import CHILD_DISPATCH_IMPLEMENTATION_IDS
from src.tool_process_safe_open import (
    FileChangedDuringRead,
    FileHashResult,
    FileTooLarge,
    IdentityMismatch,
    SafeOpenError,
    SafeOpenUnavailableError,
    create_windows_file_identity,
    hash_sha256_from_safe_handle,
    safe_open_for_read,
)
from src.tool_registry import build_default_tool_registry
from src.tool_request import create_tool_execution_request
from src.tool_safe_files import HASH_CHUNK_SIZE
from tests.tool_helpers import make_scope, tool_repository


def _enable_file_process_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (
        "src.config",
        "src.tool_executor",
        "src.tool_process_adapter",
    ):
        monkeypatch.setattr(f"{module}.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED", True)
        monkeypatch.setattr(f"{module}.PROCESS_FILE_TOOL_ISOLATION_ENABLED", True)
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.config.PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED",
        True,
    )


def _sample_authorization(
    *,
    path: str = r"C:\Cases\SENTINEL_M15_PATH\sample.txt",
    root: str = r"C:\Cases\SENTINEL_M15_PATH",
) -> dict[str, Any]:
    return {
        "canonical_path": path,
        "authorized_root": root,
        "volume_serial_number": 11,
        "file_index_high": 22,
        "file_index_low": 33,
        "expected_size_bytes": 5,
        "baseline_last_write_time_filetime": 100,
    }


def _fake_win32(
    *,
    attributes: int = 0,
    volume: int = 11,
    size_bytes: int = 5,
    index_high: int = 22,
    index_low: int = 33,
    write_time: int = 100,
    handle_path: str = r"C:\Cases\SENTINEL_M15_PATH\sample.txt",
    payload: bytes = b"hello",
    mutate_after_read: dict[str, Any] | None = None,
) -> Any:
    class FakeHandle:
        pass

    state = {"offset": 0, "reads_done": False}

    class Win32File:
        @staticmethod
        def CreateFile(*_args: Any, **_kwargs: Any) -> FakeHandle:
            return FakeHandle()

        @staticmethod
        def GetFileInformationByHandle(_handle: Any) -> tuple[Any, ...]:
            info_volume = volume
            info_size = size_bytes
            info_high = index_high
            info_low = index_low
            info_write = write_time
            if state["reads_done"] and mutate_after_read is not None:
                info_volume = mutate_after_read.get("volume", info_volume)
                info_size = mutate_after_read.get("size_bytes", info_size)
                info_high = mutate_after_read.get("index_high", info_high)
                info_low = mutate_after_read.get("index_low", info_low)
                info_write = mutate_after_read.get("write_time", info_write)
            return (
                attributes,
                0,
                0,
                info_write,
                info_volume,
                0,
                info_size,
                1,
                info_high,
                info_low,
            )

        @staticmethod
        def GetFinalPathNameByHandle(_handle: Any, _flags: int) -> str:
            return handle_path

        @staticmethod
        def ReadFile(_handle: Any, size: int) -> tuple[int, bytes]:
            if state["offset"] >= len(payload):
                state["reads_done"] = True
                return 0, b""
            chunk = payload[state["offset"] : state["offset"] + size]
            state["offset"] += len(chunk)
            if state["offset"] >= len(payload):
                state["reads_done"] = True
            return 0, chunk

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


def test_flags_default_and_dual_gate() -> None:
    assert PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED is False
    assert PROCESS_FILE_TOOL_ISOLATION_ENABLED is False
    assert PROCESS_RESOURCE_LIMITS_ENABLED is False
    assert MAX_PROCESS_IPC_REQUEST_BYTES >= 24_576
    assert MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS == 4_096


def test_eligibility_only_reviewed_file_tools() -> None:
    registry = build_default_tool_registry()
    assert registry.require("file-sha256").process_isolation == "eligible"
    assert registry.require("compare-sha256").process_isolation == "eligible"
    assert registry.require("text-search").process_isolation == "eligible"
    assert PROCESS_SAFE_FILE_IMPLEMENTATION_IDS <= PROCESS_SAFE_IMPLEMENTATION_IDS
    assert {"file-sha256", "compare-sha256", "text-search"} <= PROCESS_SAFE_TOOL_IDS
    assert CHILD_DISPATCH_IMPLEMENTATION_IDS == PROCESS_SAFE_IMPLEMENTATION_IDS


def test_unapproved_file_tool_still_cannot_be_eligible() -> None:
    with pytest.raises(Exception):
        create_tool_definition(
            tool_id="file-isolated-bad",
            name="Bad",
            description="Must remain prohibited",
            category="file-integrity",
            version="1.0.0",
            risk_level="low",
            execution_mode="internal-python",
            supported_objective_types=("hash",),
            supported_target_types=("local-file",),
            parameter_schema=(
                create_parameter_definition(
                    name="path",
                    parameter_type="file-path",
                    required=True,
                    description="path",
                ),
            ),
            requires_approval=True,
            implementation_identifier="impl_system_summary",
            process_isolation="eligible",
        )


def test_routing_requires_both_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = build_default_tool_registry().require("file-sha256")
    assert _select_execution_route(definition) == "in_process"

    monkeypatch.setattr(
        "src.tool_executor.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_FILE_TOOL_ISOLATION_ENABLED",
        False,
    )
    assert _select_execution_route(definition) == "in_process"

    monkeypatch.setattr(
        "src.tool_executor.PROCESS_FILE_TOOL_ISOLATION_ENABLED",
        True,
    )
    assert _select_execution_route(definition) == "process"

    summary = build_default_tool_registry().require("system-summary")
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_FILE_TOOL_ISOLATION_ENABLED",
        False,
    )
    assert _select_execution_route(summary) == "process"


def test_file_authorization_exact_keys_and_rejects() -> None:
    auth = _sample_authorization()
    validated = validate_file_authorization(auth)
    assert frozenset(validated.to_exact_dict()) == FILE_AUTHORIZATION_KEYS

    missing = dict(auth)
    del missing["file_index_low"]
    with pytest.raises(ToolProcessError):
        validate_file_authorization(missing)

    extra = dict(auth)
    extra["extra"] = 1
    with pytest.raises(ToolProcessError):
        validate_file_authorization(extra)

    bad_type = dict(auth)
    bad_type["volume_serial_number"] = "11"
    with pytest.raises(ToolProcessError):
        validate_file_authorization(bad_type)

    negative = dict(auth)
    negative["file_index_high"] = -1
    with pytest.raises(ToolProcessError):
        validate_file_authorization(negative)

    oversized = dict(auth)
    oversized["canonical_path"] = "C:\\" + ("a" * 5000)
    with pytest.raises(ToolProcessError):
        validate_file_authorization(oversized)

    with pytest.raises(ToolProcessError):
        validate_file_tool_process_parameters(
            "impl_file_sha256",
            {
                "file_authorization": auth,
                "file_authorization_2": auth,
            },
        )

    with pytest.raises(ToolProcessError):
        validate_file_tool_process_parameters(
            "impl_compare_sha256",
            {"file_authorization": auth},
        )

    with pytest.raises(Exception):
        validate_file_tool_process_parameters(
            "impl_compare_sha256",
            {
                "file_authorization": auth,
                "expected_sha256": "not-a-digest",
            },
        )


def test_envelope_accepts_file_authorization_shape() -> None:
    auth = _sample_authorization()
    request = create_process_execution_request(
        correlation_id=str(uuid4()),
        implementation_identifier="impl_file_sha256",
        normalized_parameters={"file_authorization": auth},
        execution_timeout_seconds=5,
        max_output_characters=1000,
    )
    assert "path" not in request.normalized_parameters
    assert "file_authorization" in request.normalized_parameters

    compare = create_process_execution_request(
        correlation_id=str(uuid4()),
        implementation_identifier="impl_compare_sha256",
        normalized_parameters={
            "file_authorization": auth,
            "expected_sha256": "a" * 64,
        },
        execution_timeout_seconds=5,
        max_output_characters=1000,
    )
    assert compare.normalized_parameters["expected_sha256"] == "a" * 64


def test_streaming_hash_no_full_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"abc" * 1000
    modules = _fake_win32(
        volume=1,
        size_bytes=len(payload),
        index_high=1,
        index_low=2,
        write_time=9,
        handle_path=r"C:\Cases\sample.txt",
        payload=payload,
    )
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: modules,
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    file_identity = create_windows_file_identity(
        canonical_path=r"C:\Cases\sample.txt",
        volume_serial_number=1,
        file_index_high=1,
        file_index_low=2,
        size_bytes=len(payload),
        last_write_time_filetime=9,
    )

    class Opened:
        handle = object()

        def close(self) -> None:
            return None

    opened = Opened()
    opened.identity = file_identity  # type: ignore[attr-defined]

    source = Path("src/tool_process_safe_open.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    hash_fn: ast.FunctionDef | None = None
    for statement in tree.body:
        if (
            isinstance(statement, ast.FunctionDef)
            and statement.name == "hash_sha256_from_safe_handle"
        ):
            hash_fn = statement
            break
    assert hash_fn is not None
    joined = False
    list_append = False
    for walk_node in ast.walk(hash_fn):
        if isinstance(walk_node, ast.Call) and isinstance(
            walk_node.func, ast.Attribute
        ):
            if walk_node.func.attr == "join":
                joined = True
            if walk_node.func.attr == "append":
                list_append = True
    assert joined is False
    assert list_append is False
    rendered = ast.unparse(hash_fn)
    assert "chunks" not in rendered
    assert "assert_safe_handle_unchanged_after_read" in rendered
    assert "GetFileInformationByHandle" not in rendered
    assert "File identity changed during read." not in rendered
    assert "Bytes read do not match open-time size." not in rendered

    result = hash_sha256_from_safe_handle(opened)  # type: ignore[arg-type]
    assert isinstance(result, FileHashResult)
    assert result.sha256_hex == hashlib.sha256(payload).hexdigest()
    assert result.size_bytes == len(payload)
    assert result.final_size_bytes == len(payload)
    assert result.final_last_write_time_filetime == 9
    assert HASH_CHUNK_SIZE == 1024 * 1024


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"size_bytes": 6}, "size"),
        ({"size_bytes": 3}, "size"),
        ({"write_time": 999}, "last-write"),
        ({"volume": 99}, "identity"),
        ({"index_low": 999}, "identity"),
    ],
)
def test_post_read_change_detection(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
    match: str,
) -> None:
    payload = b"hello"
    modules = _fake_win32(
        volume=1,
        size_bytes=len(payload),
        index_high=1,
        index_low=2,
        write_time=9,
        handle_path=r"C:\Cases\sample.txt",
        payload=payload,
        mutate_after_read=mutation,
    )
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: modules,
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    file_identity = create_windows_file_identity(
        canonical_path=r"C:\Cases\sample.txt",
        volume_serial_number=1,
        file_index_high=1,
        file_index_low=2,
        size_bytes=len(payload),
        last_write_time_filetime=9,
    )

    class Opened:
        handle = object()

    opened = Opened()
    opened.identity = file_identity  # type: ignore[attr-defined]

    with pytest.raises(FileChangedDuringRead, match=match):
        hash_sha256_from_safe_handle(opened)  # type: ignore[arg-type]


def test_file_too_large_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32(
            volume=1,
            size_bytes=MAX_TOOL_FILE_BYTES + 1,
            index_high=1,
            index_low=2,
            write_time=9,
            handle_path=r"C:\Cases\sample.txt",
            payload=b"x",
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    file_identity = create_windows_file_identity(
        canonical_path=r"C:\Cases\sample.txt",
        volume_serial_number=1,
        file_index_high=1,
        file_index_low=2,
        size_bytes=MAX_TOOL_FILE_BYTES + 1,
        last_write_time_filetime=9,
    )

    class Opened:
        handle = object()

    opened = Opened()
    opened.identity = file_identity  # type: ignore[attr-defined]

    with pytest.raises(FileTooLarge):
        hash_sha256_from_safe_handle(opened)  # type: ignore[arg-type]


def test_identity_mismatch_error_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32(
            volume=222,
            size_bytes=5,
            index_high=1,
            index_low=2,
            write_time=9,
            handle_path=r"C:\Cases\sample.txt",
            payload=b"hello",
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    expected = create_windows_file_identity(
        canonical_path=r"C:\Cases\sample.txt",
        volume_serial_number=111,
        file_index_high=1,
        file_index_low=2,
        size_bytes=5,
        last_write_time_filetime=9,
    )
    with pytest.raises(IdentityMismatch):
        safe_open_for_read(
            r"C:\Cases\sample.txt",
            expected_identity=expected,
        )


def test_child_file_sha256_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello"
    auth = _sample_authorization()
    auth["expected_size_bytes"] = len(payload)
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32(
            volume=auth["volume_serial_number"],
            size_bytes=len(payload),
            index_high=auth["file_index_high"],
            index_low=auth["file_index_low"],
            write_time=auth["baseline_last_write_time_filetime"],
            handle_path=auth["canonical_path"],
            payload=payload,
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    result = run_file_sha256({"file_authorization": auth}, {})
    assert result["sha256_hex"] == hashlib.sha256(payload).hexdigest()
    assert result["filename_only"] == "sample.txt"
    assert "canonical_path" not in result
    assert auth["canonical_path"] not in json.dumps(result)


def test_child_compare_sha256_match_and_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"hello"
    digest = hashlib.sha256(payload).hexdigest()
    auth = _sample_authorization()
    auth["expected_size_bytes"] = len(payload)
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32(
            volume=auth["volume_serial_number"],
            size_bytes=len(payload),
            index_high=auth["file_index_high"],
            index_low=auth["file_index_low"],
            write_time=auth["baseline_last_write_time_filetime"],
            handle_path=auth["canonical_path"],
            payload=payload,
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    matched = run_compare_sha256(
        {"file_authorization": auth, "expected_sha256": digest},
        {},
    )
    assert matched["matches"] is True
    mismatched = run_compare_sha256(
        {"file_authorization": auth, "expected_sha256": "b" * 64},
        {},
    )
    assert mismatched["matches"] is False


def test_safe_open_unavailable_fails_before_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    monkeypatch.setattr(
        "src.tool_process_adapter.build_isolated_file_tool_parameters",
        lambda **_kwargs: (_ for _ in ()).throw(
            SafeOpenUnavailableError("Windows safe-open requires a Windows platform.")
        ),
    )
    audits: list[Any] = []
    adapter = ToolProcessAdapter(audit_appender=audits.append)
    registry = build_default_tool_registry()
    definition = registry.require("file-sha256")
    root = tmp_path / "root"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("x", encoding="utf-8")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    )
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="file-sha256",
            normalized_parameters={"path": str(target)},
            scope_id=scope.scope_id,
            justification="unavailable coverage",
            request_status="running",
            dry_run_completed=True,
        )
    )
    result = adapter.execute(
        definition=definition,
        request=request,
        scope=scope,
    )
    assert result.outcome == "failed"
    assert result.error_class == "SafeOpenUnavailableError"
    assert result.structured_data.get("file_safe_open_unavailable") is True


def test_audit_and_result_omit_sentinel_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = r"C:\Cases\SENTINEL_M15_UNIQUE_PATH\secret.bin"
    auth = _sample_authorization(path=sentinel, root=r"C:\Cases\SENTINEL_M15_UNIQUE_PATH")
    auth["expected_size_bytes"] = 5
    payload = b"hello"
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: _fake_win32(
            volume=auth["volume_serial_number"],
            size_bytes=5,
            index_high=auth["file_index_high"],
            index_low=auth["file_index_low"],
            write_time=auth["baseline_last_write_time_filetime"],
            handle_path=sentinel,
            payload=payload,
        ),
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    result = run_file_sha256({"file_authorization": auth}, {})
    rendered = json.dumps(result)
    assert sentinel not in rendered
    assert "SENTINEL_M15_UNIQUE_PATH" not in rendered
    assert "volume_serial_number" not in rendered

    entry = create_tool_audit_entry(
        action="process_execution_completed",
        request_id=str(uuid4()),
        tool_id="file-sha256",
        safe_details={
            "correlation_id": str(uuid4()),
            "implementation_identifier": "impl_file_sha256",
            "outcome": "succeeded",
        },
    )
    assert sentinel not in json.dumps(entry.safe_details)


def test_in_process_behavior_preserved_when_flags_off(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("in-process", encoding="utf-8")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(target)},
        scope_id=scope.scope_id,
        justification="in-process coverage",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="ok",
        approver="tester",
    )
    executor = DefensiveToolExecutor()
    definition = build_default_tool_registry().require("file-sha256")
    result = executor.execute(
        definition=definition,
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded"
    assert result.structured_data["sha256"] == hashlib.sha256(b"in-process").hexdigest()
    assert result.structured_data["filename"] == "note.txt"
    assert result.structured_data.get("process_isolated") is not True


def test_source_streaming_hash_does_not_use_read_bytes_helper() -> None:
    source = Path("src/tool_process_file_tools.py").read_text(encoding="utf-8")
    assert "hash_sha256_from_safe_handle" in source
    assert "read_bytes_from_safe_handle" not in source
    assert "safe_open_for_read" in source


def test_no_second_file_compare_redesign() -> None:
    definition = build_default_tool_registry().require("compare-sha256")
    names = {item.name for item in definition.parameter_schema}
    assert names == {"path", "expected_sha256"}
    assert "path_b" not in names
    assert "other_path" not in names


@pytest.mark.skipif(os.name != "nt", reason="Windows isolated file-hash integration")
def test_windows_isolated_file_sha256_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    target = root / "hash-me.txt"
    content = b"milestone-15-isolated-hash"
    target.write_bytes(content)
    path_value = str(target.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")

    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": path_value},
        scope_id=scope.scope_id,
        justification="windows isolated hash",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="ok",
        approver="tester",
    )
    audits: list[Any] = []
    executor = DefensiveToolExecutor()
    executor._process_adapter = ToolProcessAdapter(  # noqa: SLF001
        scratch_dir=tmp_path / "scratch",
        audit_appender=audits.append,
    )
    definition = build_default_tool_registry().require("file-sha256")
    result = executor.execute(
        definition=definition,
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded"
    assert result.structured_data["process_isolated"] is True
    assert result.structured_data["sha256_hex"] == hashlib.sha256(content).hexdigest()
    assert result.structured_data["filename_only"] == "hash-me.txt"
    assert path_value not in json.dumps(result.structured_data)
    assert path_value not in result.safe_summary
    for entry in audits:
        blob = json.dumps(entry.safe_details)
        assert path_value not in blob
        assert "canonical_path" not in blob
        assert "file_authorization" not in blob


@pytest.mark.skipif(os.name != "nt", reason="Windows isolated compare integration")
def test_windows_isolated_compare_sha256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    target = root / "cmp.txt"
    content = b"compare-me"
    target.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    path_value = str(target.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")

    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["compare-sha256"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="compare-sha256",
        normalized_parameters={
            "path": path_value,
            "expected_sha256": digest,
        },
        scope_id=scope.scope_id,
        justification="windows isolated compare",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="ok",
        approver="tester",
    )
    executor = DefensiveToolExecutor()
    definition = build_default_tool_registry().require("compare-sha256")
    result = executor.execute(
        definition=definition,
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded"
    assert result.structured_data["matches"] is True
    assert result.structured_data["filename_only"] == "cmp.txt"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object file-hash path")
def test_windows_file_hash_through_job_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    monkeypatch.setattr("src.config.PROCESS_RESOURCE_LIMITS_ENABLED", True)
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_RESOURCE_LIMITS_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_job.PROCESS_RESOURCE_LIMITS_ENABLED",
        True,
    )
    root = tmp_path / "root"
    root.mkdir()
    target = root / "job.txt"
    content = b"job-object-hash"
    target.write_bytes(content)
    path_value = str(target.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")

    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": path_value},
        scope_id=scope.scope_id,
        justification="job object file hash",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="ok",
        approver="tester",
    )
    executor = DefensiveToolExecutor()
    definition = build_default_tool_registry().require("file-sha256")
    result = executor.execute(
        definition=definition,
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded", (
        f"{result.error_class}: {result.safe_summary} {result.structured_data}"
    )
    assert result.structured_data.get("resource_limits_enabled") is True
    assert result.structured_data["sha256_hex"] == hashlib.sha256(content).hexdigest()


@pytest.mark.skipif(os.name != "nt", reason="Windows TOCTOU replacement")
def test_windows_toctou_replacement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    original = root / "target.txt"
    replacement = root / "other.txt"
    original.write_text("original", encoding="utf-8")
    replacement.write_text("replacement", encoding="utf-8")
    path_value = str(original.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")

    from src.tool_process_file_auth import capture_parent_file_authorization

    scope = make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    authorization = capture_parent_file_authorization(
        path_value=path_value,
        scope=scope,
    )
    original.unlink()
    replacement.replace(original)

    with pytest.raises((IdentityMismatch, SafeOpenError)):
        opened = safe_open_for_read(
            authorization.canonical_path,
            expected_identity=authorization.to_windows_identity(),
            authorized_root=authorization.authorized_root,
        )
        opened.close()


def test_dry_run_and_denied_do_not_require_file_isolation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("x", encoding="utf-8")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(target)},
        scope_id=scope.scope_id,
        justification="dry-run coverage",
        request_status="drafted",
        dry_run_completed=False,
    )
    executor = DefensiveToolExecutor()
    definition = build_default_tool_registry().require("file-sha256")
    planned = executor.plan_dry_run(
        definition=definition,
        request=request,
        scope=scope,
    )
    assert planned.outcome == "planned"
    assert planned.structured_data.get("process_isolated") is not True
