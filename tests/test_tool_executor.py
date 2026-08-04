"""Tests for dry-run planning and defensive tool execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.security_incident import create_security_incident
from src.tool_approval import create_tool_approval
from src.tool_common import ToolPolicyError
from src.tool_definition import DefensiveToolDefinition
from src.tool_policy import assert_ready_for_execution
from src.tool_registry import ToolRegistry
from src.tool_repository import JsonToolControlRepository
from src.tool_request import (
    ToolExecutionRequest,
    create_tool_execution_request,
    replace_tool_execution_request,
)
from src.tool_result import create_tool_execution_result
from src.tool_scope import AuthorizedScope
from tests.tool_helpers import (
    incident_repository,
    make_scope,
    tool_executor,
    tool_registry,
    tool_repository,
)


def _prepare_file_request(
    tmp_path: Path,
    tool_id: str,
    parameters: dict[str, object],
) -> tuple[
    ToolRegistry,
    JsonToolControlRepository,
    DefensiveToolDefinition,
    AuthorizedScope,
    ToolExecutionRequest,
    Path,
]:
    registry = tool_registry()
    repo = tool_repository(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=[tool_id], root=root))
    definition = registry.require(tool_id)
    request = create_tool_execution_request(
        tool_id=tool_id,
        normalized_parameters=parameters,
        scope_id=scope.scope_id,
        justification="test execution",
        request_status="awaiting-approval",
    )
    request = repo.add_request(request)
    return registry, repo, definition, scope, request, root


def test_dry_run_does_not_read_target_and_redacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "secret-name.txt"
    target.write_text("never-read-if-dry-run", encoding="utf-8")
    path_value = str(target)
    registry = tool_registry()
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["file-sha256"], root=root))
    definition = registry.require("file-sha256")
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="file-sha256",
            normalized_parameters={"path": path_value},
            scope_id=scope.scope_id,
            justification="dry-run only",
            request_status="awaiting-approval",
        )
    )
    target.unlink()
    read_attempts = {"count": 0}

    def guarded_stream(*args: object, **kwargs: object) -> object:
        read_attempts["count"] += 1
        raise AssertionError("Dry run must not read the target file")

    monkeypatch.setattr("src.tool_safe_files.stream_sha256", guarded_stream)
    monkeypatch.setattr("src.tool_safe_files.read_text_lines", guarded_stream)

    result = tool_executor().plan_dry_run(
        definition=definition,
        request=request,
        scope=scope,
    )

    assert read_attempts["count"] == 0
    assert result.dry_run is True
    assert result.outcome == "planned"
    assert result.structured_data["no_action_executed"] is True
    assert result.structured_data["target_filename"] == "secret-name.txt"
    assert path_value not in str(result.structured_data)
    assert "never-read-if-dry-run" not in str(result.structured_data)


def test_dry_run_required_before_run(tmp_path: Path) -> None:
    registry = tool_registry()
    definition = registry.require("file-sha256")
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": "a.txt"},
        scope_id="66666666-6666-6666-6666-666666666666",
        justification="hash",
        request_status="approved",
        dry_run_completed=False,
    )
    with pytest.raises(ToolPolicyError):
        assert_ready_for_execution(definition, request)


def test_each_builtin_tool_success(tmp_path: Path) -> None:
    registry = tool_registry()
    incidents = incident_repository(tmp_path)
    executor = tool_executor(incidents)
    repo = tool_repository(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    sample = root / "sample.txt"
    content = "alpha alert beta alert"
    sample.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    incident = incidents.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary",
            severity="low",
        )
    )

    scope = repo.add_scope(
        make_scope(
            tmp_path,
            tool_ids=[
                "system-summary",
                "file-sha256",
                "text-search",
                "compare-sha256",
                "incident-summary",
                "simulated-log-check",
            ],
            root=root,
        )
    )

    cases = [
        ("system-summary", {}, "drafted"),
        ("file-sha256", {"path": str(sample)}, "awaiting-approval"),
        (
            "text-search",
            {"path": str(sample), "query": "alert"},
            "awaiting-approval",
        ),
        (
            "compare-sha256",
            {"path": str(sample), "expected_sha256": digest},
            "awaiting-approval",
        ),
        (
            "incident-summary",
            {"incident_id": incident.incident_id},
            "drafted",
        ),
        (
            "simulated-log-check",
            {"fixture": "malware-keyword", "needle": "alert"},
            "drafted",
        ),
    ]

    for tool_id, parameters, status in cases:
        definition = registry.require(tool_id)
        request = repo.add_request(
            create_tool_execution_request(
                tool_id=tool_id,
                normalized_parameters=parameters,
                scope_id=scope.scope_id,
                justification=f"run {tool_id}",
                request_status=status,
                dry_run_completed=True,
                incident_id=(
                    incident.incident_id if tool_id == "incident-summary" else None
                ),
            )
        )
        approval = None
        if status == "awaiting-approval":
            approval = create_tool_approval(
                request=request,
                decision="approved",
                reason="ok",
            )
            request = replace_tool_execution_request(
                request,
                request_status="approved",
            )
        running = replace_tool_execution_request(request, request_status="running")
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=approval,
        )
        assert result.outcome == "succeeded", tool_id
        assert result.dry_run is False


def test_symlink_and_non_regular_rejection(tmp_path: Path) -> None:
    registry, repo, definition, scope, _request, root = _prepare_file_request(
        tmp_path,
        "file-sha256",
        {"path": "placeholder"},
    )
    real = tmp_path / "real.txt"
    real.write_text("x", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("Windows denied symlink creation")

    request = repo.add_request(
        create_tool_execution_request(
            tool_id="file-sha256",
            normalized_parameters={"path": str(link)},
            scope_id=scope.scope_id,
            justification="symlink",
            request_status="approved",
            dry_run_completed=True,
        )
    )
    approval = create_tool_approval(request=request, decision="approved", reason="ok")
    running = replace_tool_execution_request(request, request_status="running")
    result = tool_executor().execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=approval,
    )
    assert result.outcome in {"denied", "failed"}

    directory_request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(root)},
        scope_id=scope.scope_id,
        justification="directory",
        request_status="approved",
        dry_run_completed=True,
    )
    directory_request = repo.add_request(directory_request)
    approval = create_tool_approval(
        request=directory_request,
        decision="approved",
        reason="ok",
    )
    running = replace_tool_execution_request(
        directory_request,
        request_status="running",
    )
    result = tool_executor().execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=approval,
    )
    assert result.outcome in {"denied", "failed"}


def test_oversized_file_rejected(tmp_path: Path) -> None:
    from src.config import MAX_TOOL_FILE_BYTES

    registry, repo, definition, scope, _req, root = _prepare_file_request(
        tmp_path,
        "file-sha256",
        {"path": "placeholder"},
    )
    huge = root / "huge.bin"
    huge.write_bytes(b"a" * (MAX_TOOL_FILE_BYTES + 1))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="file-sha256",
            normalized_parameters={"path": str(huge)},
            scope_id=scope.scope_id,
            justification="too big",
            request_status="approved",
            dry_run_completed=True,
        )
    )
    approval = create_tool_approval(request=request, decision="approved", reason="ok")
    running = replace_tool_execution_request(request, request_status="running")
    result = tool_executor().execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=approval,
    )
    assert result.outcome in {"denied", "failed"}


def test_unknown_implementation_fails_closed(tmp_path: Path) -> None:
    from src.tool_definition import create_tool_definition
    from src.tool_executor import DefensiveToolExecutor

    definition = create_tool_definition(
        tool_id="ghost-tool",
        name="Ghost",
        description="Missing implementation",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        implementation_identifier="impl_does_not_exist",
    )
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["ghost-tool"]))
    # Scope creation validates tool IDs only as strings; registry isn't consulted.
    request = create_tool_execution_request(
        tool_id="ghost-tool",
        normalized_parameters={},
        scope_id=scope.scope_id,
        justification="ghost",
        request_status="drafted",
        dry_run_completed=True,
    )
    running = replace_tool_execution_request(request, request_status="running")
    result = DefensiveToolExecutor(implementations={}).execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=None,
    )
    assert result.outcome == "denied"
    assert result.error_class in {"ToolValidationError", "ToolPolicyError"}


def test_output_truncation_and_safe_exception(tmp_path: Path) -> None:
    from src.tool_definition import create_tool_definition
    from src.tool_executor import DefensiveToolExecutor

    def noisy(
        _params: Mapping[str, Any],
        _ctx: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {"blob": "x" * 20_000}

    def boom(
        _params: Mapping[str, Any],
        _ctx: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError("secret-path-C:\\hidden\\file.txt")

    definition = create_tool_definition(
        tool_id="noisy-tool",
        name="Noisy",
        description="Noisy output",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        maximum_output_characters=200,
        implementation_identifier="impl_noisy_tool",
    )
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["noisy-tool"]))
    request = create_tool_execution_request(
        tool_id="noisy-tool",
        normalized_parameters={},
        scope_id=scope.scope_id,
        justification="noise",
        request_status="drafted",
        dry_run_completed=True,
    )
    running = replace_tool_execution_request(request, request_status="running")
    executor = DefensiveToolExecutor(
        implementations={"impl_noisy_tool": noisy},
    )
    result = executor.execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=None,
    )
    assert result.outcome == "succeeded"
    assert result.output_truncated is True

    boom_def = create_tool_definition(
        tool_id="boom-tool",
        name="Boom",
        description="Boom",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        implementation_identifier="impl_boom_tool",
    )
    scope2 = repo.add_scope(make_scope(tmp_path, tool_ids=["boom-tool"], name="Boom"))
    request2 = create_tool_execution_request(
        tool_id="boom-tool",
        normalized_parameters={},
        scope_id=scope2.scope_id,
        justification="boom",
        request_status="drafted",
        dry_run_completed=True,
    )
    running2 = replace_tool_execution_request(request2, request_status="running")
    result2 = DefensiveToolExecutor(
        implementations={"impl_boom_tool": boom},
    ).execute(
        definition=boom_def,
        request=running2,
        scope=scope2,
        approval=None,
    )
    assert result2.outcome == "failed"
    assert result2.error_class == "RuntimeError"
    assert "secret-path" not in result2.safe_summary
    assert "C:\\hidden" not in str(result2.structured_data)


def test_dry_run_result_cannot_claim_success() -> None:
    with pytest.raises(Exception):
        create_tool_execution_result(
            request_id="77777777-7777-7777-7777-777777777777",
            tool_id="system-summary",
            outcome="succeeded",
            safe_summary="nope",
            dry_run=True,
        )


def test_execution_cannot_occur_twice(tmp_path: Path) -> None:
    registry = tool_registry()
    definition = registry.require("system-summary")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="once",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    succeeded = replace_tool_execution_request(running, request_status="succeeded")
    with pytest.raises(ToolPolicyError):
        assert_ready_for_execution(definition, succeeded)
