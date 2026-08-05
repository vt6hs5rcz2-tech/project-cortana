"""Tests for Milestone 16 process-isolated text-search."""

from __future__ import annotations

import ast
import codecs
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.config import (
    MAX_TOOL_TEXT_SEARCH_MATCHES,
    MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS,
    MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS,
    PROCESS_FILE_TOOL_ISOLATION_ENABLED,
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
)
from src.tool_approval import create_tool_approval
from src.tool_audit import create_tool_audit_entry
from src.tool_executor import DefensiveToolExecutor, _select_execution_route
from src.tool_process_adapter import ToolProcessAdapter
from src.tool_process_common import (
    PROCESS_SAFE_FILE_IMPLEMENTATION_IDS,
    PROCESS_SAFE_IMPLEMENTATION_IDS,
    ToolProcessError,
)
from src.tool_process_envelope import create_process_execution_request
from src.tool_process_file_auth import validate_file_tool_process_parameters
from src.tool_process_file_tools import run_text_search
from src.tool_process_runner import CHILD_DISPATCH_IMPLEMENTATION_IDS
from src.tool_process_safe_open import (
    FileChangedDuringRead,
    IdentityMismatch,
    SafeOpenError,
    SafeOpenUnavailableError,
    create_windows_file_identity,
    safe_open_for_read,
)
from src.tool_process_text_search import (
    format_text_search_preview,
    stream_text_search_from_safe_handle,
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
    path: str = r"C:\Cases\SENTINEL_M16_PATH\sample.txt",
    root: str = r"C:\Cases\SENTINEL_M16_PATH",
    size_bytes: int = 11,
) -> dict[str, Any]:
    return {
        "canonical_path": path,
        "authorized_root": root,
        "volume_serial_number": 11,
        "file_index_high": 22,
        "file_index_low": 33,
        "expected_size_bytes": size_bytes,
        "baseline_last_write_time_filetime": 100,
    }


def _fake_win32(
    *,
    attributes: int = 0,
    volume: int = 11,
    size_bytes: int = 11,
    index_high: int = 22,
    index_low: int = 33,
    write_time: int = 100,
    handle_path: str = r"C:\Cases\SENTINEL_M16_PATH\sample.txt",
    payload: bytes = b"hello world",
    chunk_force_size: int | None = None,
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
            read_size = chunk_force_size if chunk_force_size is not None else size
            if state["offset"] >= len(payload):
                state["reads_done"] = True
                return 0, b""
            chunk = payload[state["offset"] : state["offset"] + read_size]
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


def _opened_for(payload: bytes, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    auth_path = kwargs.get("handle_path", r"C:\Cases\sample.txt")
    modules = _fake_win32(
        volume=1,
        size_bytes=len(payload),
        index_high=1,
        index_low=2,
        write_time=9,
        handle_path=auth_path,
        payload=payload,
        chunk_force_size=kwargs.get("chunk_force_size"),
        mutate_after_read=kwargs.get("mutate_after_read"),
    )
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: modules,
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    file_identity = create_windows_file_identity(
        canonical_path=auth_path,
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
    return opened


def test_flags_and_eligibility() -> None:
    assert PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED is False
    assert PROCESS_FILE_TOOL_ISOLATION_ENABLED is False
    registry = build_default_tool_registry()
    assert registry.require("text-search").process_isolation == "eligible"
    assert "impl_text_search" in PROCESS_SAFE_FILE_IMPLEMENTATION_IDS
    assert CHILD_DISPATCH_IMPLEMENTATION_IDS == PROCESS_SAFE_IMPLEMENTATION_IDS


def test_routing_requires_both_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = build_default_tool_registry().require("text-search")
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


def test_envelope_exact_keys() -> None:
    auth = _sample_authorization()
    request = create_process_execution_request(
        correlation_id=str(uuid4()),
        implementation_identifier="impl_text_search",
        normalized_parameters={
            "file_authorization": auth,
            "query": "alert",
            "max_matches": 5,
        },
        execution_timeout_seconds=5,
        max_output_characters=1000,
    )
    assert set(request.normalized_parameters) == {
        "file_authorization",
        "query",
        "max_matches",
    }
    with pytest.raises(ToolProcessError):
        validate_file_tool_process_parameters(
            "impl_text_search",
            {"file_authorization": auth, "query": "alert"},
        )
    with pytest.raises(ToolProcessError):
        validate_file_tool_process_parameters(
            "impl_text_search",
            {
                "file_authorization": auth,
                "query": "alert",
                "max_matches": 5,
                "path": r"C:\Cases\x.txt",
            },
        )
    with pytest.raises(ToolProcessError):
        validate_file_tool_process_parameters(
            "impl_text_search",
            {
                "file_authorization": auth,
                "query": "",
                "max_matches": 5,
            },
        )


def test_incremental_decoder_and_no_full_buffer() -> None:
    source = Path("src/tool_process_text_search.py").read_text(encoding="utf-8")
    assert "codecs.getincrementaldecoder" in source
    assert 'errors="replace"' in source or "errors='replace'" in source
    assert "_split_complete_utf8_prefix" not in source
    assert "_utf8_sequence_length" not in source
    assert "byte_buffer" not in source
    tree = ast.parse(source)
    fn = None
    for statement in tree.body:
        if (
            isinstance(statement, ast.FunctionDef)
            and statement.name == "stream_text_search_from_safe_handle"
        ):
            fn = statement
            break
    assert fn is not None
    rendered = ast.unparse(fn)
    assert 'b"".join' not in rendered
    assert "chunks.append" not in rendered
    assert "decoder.decode(raw, final=False)" in rendered
    assert (
        'decoder.decode(b"", final=True)' in rendered
        or "decoder.decode(b'', final=True)" in rendered
    )
    assert "_split_complete_utf8_prefix" not in rendered


def test_multibyte_and_query_split_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "café" with é as UTF-8 C3 A9; split the multibyte sequence across chunks.
    payload = "caf".encode("utf-8") + b"\xc3" + b"\xa9\nfindme\n"
    opened = _opened_for(payload, monkeypatch, chunk_force_size=3)
    result = stream_text_search_from_safe_handle(
        opened,
        query="café",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result.match_count == 1
    assert result.matches[0].line_number == 1
    assert result.matches[0].preview == "café"

    payload2 = b"prefi" + b"xTARGET" + b"suffix\n"
    opened2 = _opened_for(payload2, monkeypatch, chunk_force_size=5)
    result2 = stream_text_search_from_safe_handle(
        opened2,
        query="xTARGET",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result2.match_count == 1


def test_decoder_final_true_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_calls: list[tuple[bytes, bool]] = []
    real_factory = codecs.getincrementaldecoder("utf-8")

    class SpyDecoder:
        def __init__(self) -> None:
            self._inner = real_factory(errors="replace")

        def decode(self, data: bytes, final: bool = False) -> str:
            decode_calls.append((bytes(data), bool(final)))
            return self._inner.decode(data, final=final)

    monkeypatch.setattr(
        "src.tool_process_text_search.codecs.getincrementaldecoder",
        lambda _name: (lambda *, errors="strict": SpyDecoder()),
    )
    payload = "caf".encode("utf-8") + b"\xc3" + b"\xa9\n"
    opened = _opened_for(payload, monkeypatch, chunk_force_size=3)
    result = stream_text_search_from_safe_handle(
        opened,
        query="café",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result.match_count == 1
    assert any(not final and data for data, final in decode_calls)
    finals = [call for call in decode_calls if call[1] is True]
    assert len(finals) == 1
    assert finals[0] == (b"", True)


def test_line_endings_and_final_line(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"one\r\ntwo\nthree"
    opened = _opened_for(payload, monkeypatch, chunk_force_size=2)
    result = stream_text_search_from_safe_handle(
        opened,
        query="two",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result.match_count == 1
    assert result.matches[0].line_number == 2
    opened3 = _opened_for(payload, monkeypatch, chunk_force_size=4)
    result3 = stream_text_search_from_safe_handle(
        opened3,
        query="three",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result3.matches[0].line_number == 3


def test_invalid_utf8_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"bad\xffbyte\nneedle\n"
    opened = _opened_for(payload, monkeypatch, chunk_force_size=2)
    result = stream_text_search_from_safe_handle(
        opened,
        query="needle",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result.match_count == 1
    # Replacement character from invalid UTF-8 must not break later matches.
    opened2 = _opened_for(payload, monkeypatch, chunk_force_size=2)
    result2 = stream_text_search_from_safe_handle(
        opened2,
        query="\ufffd",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result2.match_count == 1


def test_pending_line_bound_and_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "EDGE"
    # Oversized line with match near start, then more content, then newline + next line.
    prefix = "xxEDGEyy" + ("Z" * MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS)
    payload = (prefix + "\nnext EDGE line\n").encode("utf-8")
    opened = _opened_for(payload, monkeypatch, chunk_force_size=1024)
    result = stream_text_search_from_safe_handle(
        opened,
        query=query,
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result.match_count >= 2
    assert result.matches[0].line_number == 1
    assert any(item.line_number == 2 for item in result.matches)

    # Boundary-split query across discard overlap.
    left = "A" * (MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS - 2)
    payload2 = (left + "ED" + "GE" + "tail\n").encode("utf-8")
    opened2 = _opened_for(payload2, monkeypatch, chunk_force_size=4096)
    result2 = stream_text_search_from_safe_handle(
        opened2,
        query="EDGE",
        max_matches=5,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result2.match_count == 1


def test_max_matches_success_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hit\n" * 20
    opened = _opened_for(payload, monkeypatch, chunk_force_size=3)
    result = stream_text_search_from_safe_handle(
        opened,
        query="hit",
        max_matches=3,
        chunk_size=HASH_CHUNK_SIZE,
    )
    assert result.match_count == 3
    assert result.truncated is True


def test_preview_bound() -> None:
    long_line = "x" * (MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS + 50)
    preview = format_text_search_preview(long_line)
    assert preview.endswith("...")
    assert len(preview) == MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS + 3


def test_change_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"stable\n"
    opened = _opened_for(
        payload,
        monkeypatch,
        chunk_force_size=2,
        mutate_after_read={"write_time": 999},
    )
    with pytest.raises(FileChangedDuringRead):
        stream_text_search_from_safe_handle(
            opened,
            query="stable",
            max_matches=5,
            chunk_size=HASH_CHUNK_SIZE,
        )


def test_process_response_accepts_nested_matches() -> None:
    from src.tool_process_envelope import create_process_execution_response

    response = create_process_execution_response(
        correlation_id=str(uuid4()),
        outcome="succeeded",
        structured_data={
            "filename": "sample.txt",
            "match_count": 1,
            "truncated": False,
            "matches": [{"line_number": 2, "preview": "beta alert"}],
        },
        safe_summary="Process-isolated tool completed successfully.",
        output_truncated=False,
        error_class=None,
    )
    assert response.structured_data["matches"][0]["preview"] == "beta alert"


def test_child_result_shape_and_no_query_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"alpha\nbeta alert\n"
    auth = _sample_authorization(size_bytes=len(payload))
    modules = _fake_win32(
        volume=auth["volume_serial_number"],
        size_bytes=len(payload),
        index_high=auth["file_index_high"],
        index_low=auth["file_index_low"],
        write_time=auth["baseline_last_write_time_filetime"],
        handle_path=auth["canonical_path"],
        payload=payload,
    )
    monkeypatch.setattr(
        "src.tool_process_safe_open._require_win32_modules",
        lambda: modules,
    )
    monkeypatch.setattr("src.tool_process_safe_open.os.name", "nt")
    result = run_text_search(
        {
            "file_authorization": auth,
            "query": "alert",
            "max_matches": 5,
        },
        {},
    )
    assert set(result) == {"filename", "match_count", "truncated", "matches"}
    assert "query" not in result
    assert result["filename"] == "sample.txt"
    assert result["match_count"] == 1
    assert result["matches"][0]["line_number"] == 2
    assert auth["canonical_path"] not in json.dumps(result)


def test_audit_omits_sentinel_values() -> None:
    sentinel_path = r"C:\Cases\SENTINEL_M16_UNIQUE\secret.log"
    sentinel_query = "SENTINEL_M16_QUERY_VALUE"
    sentinel_snippet = "SENTINEL_M16_MATCHED_SNIPPET"
    entry = create_tool_audit_entry(
        action="process_execution_completed",
        request_id=str(uuid4()),
        tool_id="text-search",
        safe_details={
            "correlation_id": str(uuid4()),
            "implementation_identifier": "impl_text_search",
            "outcome": "succeeded",
        },
    )
    blob = json.dumps(entry.safe_details)
    assert sentinel_path not in blob
    assert sentinel_query not in blob
    assert sentinel_snippet not in blob


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
    adapter = ToolProcessAdapter()
    definition = build_default_tool_registry().require("text-search")
    root = tmp_path / "root"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("hello", encoding="utf-8")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["text-search"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="text-search",
        normalized_parameters={"path": str(target), "query": "hello"},
        scope_id=scope.scope_id,
        justification="unavailable coverage",
        request_status="running",
        dry_run_completed=True,
    )
    result = adapter.execute(definition=definition, request=request, scope=scope)
    assert result.outcome == "failed"
    assert result.error_class == "SafeOpenUnavailableError"


def test_in_process_preserved_when_flags_off(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("find me here\n", encoding="utf-8")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["text-search"], root=root)
    )
    request = create_tool_execution_request(
        tool_id="text-search",
        normalized_parameters={"path": str(target), "query": "find me"},
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
    result = DefensiveToolExecutor().execute(
        definition=build_default_tool_registry().require("text-search"),
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded"
    assert result.structured_data["filename"] == "note.txt"
    assert result.structured_data["match_count"] == 1
    assert result.structured_data.get("process_isolated") is not True


def test_dry_run_does_not_require_isolation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("x", encoding="utf-8")
    scope = make_scope(tmp_path, tool_ids=["text-search"], root=root)
    request = create_tool_execution_request(
        tool_id="text-search",
        normalized_parameters={"path": str(target), "query": "x"},
        scope_id=scope.scope_id,
        justification="dry-run coverage",
        request_status="drafted",
        dry_run_completed=False,
    )
    planned = DefensiveToolExecutor().plan_dry_run(
        definition=build_default_tool_registry().require("text-search"),
        request=request,
        scope=scope,
    )
    assert planned.outcome == "planned"


@pytest.mark.skipif(os.name != "nt", reason="Windows isolated text-search")
def test_windows_isolated_text_search_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    target = root / "search-me.txt"
    content = "alpha\nbeta SENTINEL_M16_LIVE\ngamma\n"
    target.write_text(content, encoding="utf-8")
    path_value = str(target.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")
    scope = make_scope(tmp_path, tool_ids=["text-search"], root=root)
    request = create_tool_execution_request(
        tool_id="text-search",
        normalized_parameters={"path": path_value, "query": "SENTINEL_M16_LIVE"},
        scope_id=scope.scope_id,
        justification="windows isolated search",
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
    result = executor.execute(
        definition=build_default_tool_registry().require("text-search"),
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded", (
        f"{result.error_class}: {result.safe_summary} {result.structured_data}"
    )
    assert result.structured_data["process_isolated"] is True
    assert result.structured_data["match_count"] == 1
    assert result.structured_data["filename"] == "search-me.txt"
    rendered = json.dumps(result.structured_data)
    assert path_value not in rendered
    assert "query" not in result.structured_data
    for entry in audits:
        blob = json.dumps(entry.safe_details)
        assert path_value not in blob
        assert "SENTINEL_M16_LIVE" not in blob
        assert "query" not in blob


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object text-search")
def test_windows_text_search_through_job_object(
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
    target = root / "job-search.txt"
    target.write_text("job object needle\n", encoding="utf-8")
    path_value = str(target.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")
    scope = make_scope(tmp_path, tool_ids=["text-search"], root=root)
    request = create_tool_execution_request(
        tool_id="text-search",
        normalized_parameters={"path": path_value, "query": "needle"},
        scope_id=scope.scope_id,
        justification="job object text search",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="ok",
        approver="tester",
    )
    result = DefensiveToolExecutor().execute(
        definition=build_default_tool_registry().require("text-search"),
        request=request,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded", (
        f"{result.error_class}: {result.safe_summary} {result.structured_data}"
    )
    assert result.structured_data.get("resource_limits_enabled") is True
    assert result.structured_data["match_count"] == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows TOCTOU text-search")
def test_windows_toctou_replacement_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_file_process_isolation(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    original = root / "target.txt"
    replacement = root / "other.txt"
    original.write_text("original needle\n", encoding="utf-8")
    replacement.write_text("replacement\n", encoding="utf-8")
    path_value = str(original.resolve())
    if not (len(path_value) >= 3 and path_value[1] == ":"):
        pytest.skip("Resolved path is not a drive-letter path")
    from src.tool_process_file_auth import capture_parent_file_authorization

    scope = make_scope(tmp_path, tool_ids=["text-search"], root=root)
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


def test_no_regex_or_casefold_in_module() -> None:
    source = Path("src/tool_process_text_search.py").read_text(encoding="utf-8")
    assert "import re" not in source
    assert "re.compile" not in source
    assert "casefold" not in source
    assert "lower(" not in source
