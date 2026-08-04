"""Focused tests for defensive-tool caller-wait timeout behavior."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.tool_approval import create_tool_approval
from src.tool_common import ToolPolicyError
from src.tool_definition import DefensiveToolDefinition, create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_policy import assert_ready_for_execution
from src.tool_request import (
    create_tool_execution_request,
    replace_tool_execution_request,
)
from tests.tool_helpers import make_scope, tool_registry, tool_repository

SLOW_DURATION_SECONDS = 3.0
TIMEOUT_SECONDS = 1
# Caller must return well before the slow worker's natural completion.
MAX_CALLER_WAIT_SECONDS = 2.0


def _slow_implementation(
    _parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    time.sleep(SLOW_DURATION_SECONDS)
    return {"completed_after_sleep": True, "secret": "should-not-publish"}


def _fast_implementation(
    _parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    return {"ok": True}


def _slow_tool_definition() -> DefensiveToolDefinition:
    return create_tool_definition(
        tool_id="slow-tool",
        name="Slow Tool",
        description="Trusted slow test implementation for timeout coverage.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        timeout_seconds=TIMEOUT_SECONDS,
        requires_approval=False,
        implementation_identifier="impl_slow_tool",
    )


def test_timeout_returns_before_slow_worker_finishes(tmp_path: Path) -> None:
    """Caller wait time must be bounded; late worker completion must not publish."""
    definition = _slow_tool_definition()
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["slow-tool"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="slow-tool",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="timeout coverage",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    repo.update_request(running)

    finished = threading.Event()

    def tracked_slow(
        parameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            return _slow_implementation(parameters, context)
        finally:
            finished.set()

    executor = DefensiveToolExecutor(
        implementations={"impl_slow_tool": tracked_slow},
    )
    try:
        started = time.monotonic()
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
        elapsed = time.monotonic() - started

        assert elapsed < MAX_CALLER_WAIT_SECONDS
        assert elapsed < SLOW_DURATION_SECONDS
        assert result.outcome == "failed"
        assert result.error_class == "TimeoutError"
        assert result.structured_data.get("timeout") is True
        assert result.structured_data.get("caller_stopped_waiting") is True
        assert result.structured_data.get("worker_forcibly_terminated") is False
        assert "traceback" not in result.safe_summary.lower()
        assert "Traceback" not in str(result.structured_data)
        assert "should-not-publish" not in result.safe_summary
        assert "should-not-publish" not in str(result.structured_data)

        saved = repo.add_result(result)
        failed_request = replace_tool_execution_request(
            running,
            request_status="failed",
        )
        repo.update_request(failed_request)

        # Wait until the abandoned worker finishes naturally.
        assert finished.wait(timeout=SLOW_DURATION_SECONDS + 2.0)
        # Extra settle time for the done-callback discard path.
        time.sleep(0.2)

        results = repo.list_results()
        assert len(results) == 1
        assert results[0].result_id == saved.result_id
        assert results[0].outcome == "failed"
        assert all(item.outcome != "succeeded" for item in results)

        failure_audits = [
            entry
            for entry in repo.list_audit_entries()
            if entry.action == "execution-failed"
            and entry.request_id == running.request_id
        ]
        success_audits = [
            entry
            for entry in repo.list_audit_entries()
            if entry.action == "execution-succeeded"
            and entry.request_id == running.request_id
        ]
        assert len(failure_audits) == 1
        assert success_audits == []

        reloaded = repo.get_request(running.request_id)
        assert reloaded is not None
        assert reloaded.request_status == "failed"
        with pytest.raises(ToolPolicyError):
            assert_ready_for_execution(definition, reloaded)
    finally:
        executor.shutdown()


def test_timeout_does_not_affect_fast_tools_dry_runs_or_limits(
    tmp_path: Path,
) -> None:
    """Normal success, dry-run, and output bounding remain intact."""
    fast_definition = create_tool_definition(
        tool_id="fast-tool",
        name="Fast Tool",
        description="Trusted fast test implementation.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        timeout_seconds=TIMEOUT_SECONDS,
        requires_approval=False,
        maximum_output_characters=200,
        implementation_identifier="impl_fast_tool",
    )
    noisy_definition = create_tool_definition(
        tool_id="noisy-timeout-tool",
        name="Noisy",
        description="Bounded output tool.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        timeout_seconds=TIMEOUT_SECONDS,
        requires_approval=False,
        maximum_output_characters=200,
        implementation_identifier="impl_noisy_timeout_tool",
    )

    def noisy(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {"blob": "x" * 20_000}

    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["fast-tool", "noisy-timeout-tool"])
    )
    executor = DefensiveToolExecutor(
        implementations={
            "impl_fast_tool": _fast_implementation,
            "impl_noisy_timeout_tool": noisy,
        },
    )
    try:
        fast_request = create_tool_execution_request(
            tool_id="fast-tool",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="fast path",
            request_status="drafted",
            dry_run_completed=True,
        )
        dry = executor.plan_dry_run(
            definition=fast_definition,
            request=fast_request,
            scope=scope,
        )
        assert dry.outcome == "planned"
        assert dry.dry_run is True

        running = replace_tool_execution_request(
            fast_request,
            request_status="running",
        )
        fast_result = executor.execute(
            definition=fast_definition,
            request=running,
            scope=scope,
            approval=None,
        )
        assert fast_result.outcome == "succeeded"

        noisy_request = create_tool_execution_request(
            tool_id="noisy-timeout-tool",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="noise",
            request_status="running",
            dry_run_completed=True,
        )
        noisy_result = executor.execute(
            definition=noisy_definition,
            request=noisy_request,
            scope=scope,
            approval=None,
        )
        assert noisy_result.outcome == "succeeded"
        assert noisy_result.output_truncated is True
    finally:
        executor.shutdown()


def test_executor_shutdown_returns_promptly_for_finite_slow_worker(
    tmp_path: Path,
) -> None:
    """shutdown(wait=False) returns promptly; suite still waits for finite workers.

    This does not claim the interpreter could exit while the worker remains alive.
    """
    definition = _slow_tool_definition()
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(
        make_scope(tmp_path, tool_ids=["slow-tool"], name="Shutdown scope")
    )
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="slow-tool",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="shutdown coverage",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    repo.update_request(running)

    finished = threading.Event()

    def tracked_slow(
        parameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            return _slow_implementation(parameters, context)
        finally:
            finished.set()

    executor = DefensiveToolExecutor(
        implementations={"impl_slow_tool": tracked_slow},
    )
    try:
        started = time.monotonic()
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
        execute_elapsed = time.monotonic() - started
        assert execute_elapsed < MAX_CALLER_WAIT_SECONDS
        assert result.outcome == "failed"
        assert result.error_class == "TimeoutError"

        shutdown_started = time.monotonic()
        executor.shutdown()
        shutdown_elapsed = time.monotonic() - shutdown_started
        assert shutdown_elapsed < MAX_CALLER_WAIT_SECONDS
        # shutdown returned; the finite worker may still be alive until it finishes.
    finally:
        # Ensure the finite worker completes so the test suite exits cleanly.
        assert finished.wait(timeout=SLOW_DURATION_SECONDS + 2.0)


def test_timeout_preserves_authorization_and_approval_gates(tmp_path: Path) -> None:
    """Timeout changes must not weaken scope or approval enforcement."""
    registry = tool_registry()
    definition = registry.require("file-sha256")
    repo = tool_repository(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(root / "missing.txt")},
        scope_id=scope.scope_id,
        justification="unauthorized tool",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="should still be denied by scope",
    )
    executor = DefensiveToolExecutor()
    try:
        result = executor.execute(
            definition=definition,
            request=replace_tool_execution_request(
                request,
                request_status="running",
            ),
            scope=scope,
            approval=approval,
        )
        assert result.outcome == "denied"
        assert result.error_class == "ToolAuthorizationError"
    finally:
        executor.shutdown()
