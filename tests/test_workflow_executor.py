"""Execution, approval, scope, and policy tests for Milestone 10 workflows."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.tool_approval import create_tool_approval
from collections.abc import Mapping

from src.tool_definition import DefensiveToolDefinition
from src.tool_executor import DefensiveToolExecutor
from src.tool_policy import approval_required
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_request import create_tool_execution_request
from src.tool_scope import disable_authorized_scope
from src.workflow_common import ManualWorkflowClock
from src.workflow_definition import (
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_registry import WorkflowRegistry
from src.workflow_request import create_workflow_run_request
from tests.workflow_helpers import make_executor, make_scope


class CountingToolExecutor(DefensiveToolExecutor):
    """Track plan_dry_run/execute calls without bypassing the real executor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.plan_calls = 0
        self.execute_calls = 0
        self.planned_tool_ids: list[str] = []
        self.executed_tool_ids: list[str] = []
        self.after_first_plan: Any = None

    def plan_dry_run(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.plan_calls += 1
        self.planned_tool_ids.append(kwargs["definition"].tool_id)
        result = super().plan_dry_run(**kwargs)
        if self.after_first_plan is not None and self.plan_calls == 1:
            self.after_first_plan()
        return result

    def execute(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.execute_calls += 1
        self.executed_tool_ids.append(kwargs["definition"].tool_id)
        if self.after_first_plan is not None and self.execute_calls == 1:
            self.after_first_plan()
        return super().execute(**kwargs)


def _two_step_registry(tools: ToolRegistry) -> WorkflowRegistry:
    registry = WorkflowRegistry(tool_registry=tools)
    registry.register(
        create_workflow_definition(
            name="two-step",
            version="1.0.0",
            description="Two step playbook",
            steps=(
                create_workflow_step_definition(
                    step_id="first",
                    tool_id="system-summary",
                    position=0,
                ),
                create_workflow_step_definition(
                    step_id="second",
                    tool_id="simulated-log-check",
                    static_parameters={"fixture": "auth-noise"},
                    position=1,
                ),
            ),
            tool_registry=tools,
        )
    )
    return registry


def test_dry_run_is_sequential_and_never_executes(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    executor._tool_executor = counting  # noqa: SLF001 - intentional test seam
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "completed"
    assert result.dry_run is True
    assert counting.plan_calls == 2
    assert counting.execute_calls == 0
    assert counting.planned_tool_ids == ["system-summary", "simulated-log-check"]
    assert [step.status for step in result.step_results] == ["planned", "planned"]
    assert all(
        step.tool_result is not None and step.tool_result.outcome == "planned"
        for step in result.step_results
    )
    assert runs.get_run(result.run_id) is not None


def test_stop_after_first_failure_no_retries(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )

    def boom(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError("simulated failure")

    from src.tool_implementations import build_implementation_dispatch

    dispatch = build_implementation_dispatch()
    dispatch["impl_system_summary"] = boom
    counting = CountingToolExecutor(implementations=dispatch)
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert result.status == "failed"
    assert counting.execute_calls == 1
    assert counting.plan_calls == 0
    assert len(result.step_results) == 1
    assert result.step_results[0].status == "failed"


def test_scope_disabled_after_first_step_denies_next(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    def disable_after_first() -> None:
        current = repo.get_scope(scope.scope_id)
        assert current is not None
        repo.update_scope(disable_authorized_scope(current))

    counting.after_first_plan = disable_after_first
    executor._tool_executor = counting  # noqa: SLF001

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "denied"
    assert counting.plan_calls == 1
    assert counting.execute_calls == 0
    assert len(result.step_results) == 2
    assert result.step_results[0].status == "planned"
    assert result.step_results[1].status == "denied"


def test_preflight_denies_when_scope_missing_a_tool(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(tool_ids=["system-summary"])  # missing simulated-log-check
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "preflight_failed"
    assert counting.plan_calls == 0
    assert counting.execute_calls == 0
    assert result.step_results == ()


def test_runtime_budget_enforced_live(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    clock = ManualWorkflowClock()
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
        clock=clock,
        max_runtime_seconds=1,
    )
    counting = CountingToolExecutor()

    def advance_clock() -> None:
        clock.advance(2)

    counting.after_first_plan = advance_clock
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "timed_out"
    assert counting.plan_calls == 1
    assert counting.execute_calls == 0


def test_explicit_execute_uses_execute_after_validation(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert result.status == "completed"
    assert result.dry_run is False
    assert counting.execute_calls == 2
    assert counting.plan_calls == 0
    assert counting.executed_tool_ids == [
        "system-summary",
        "simulated-log-check",
    ]
    assert all(
        step.tool_result is not None and step.tool_result.outcome == "succeeded"
        for step in result.step_results
    )


def test_approval_required_step_blocked_without_record(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = WorkflowRegistry(tool_registry=tools)
    sample_root = tmp_path / "root"
    sample_root.mkdir()
    sample = sample_root / "sample.txt"
    sample.write_text("abc", encoding="utf-8")
    workflows.register(
        create_workflow_definition(
            name="hash-once",
            version="1.0.0",
            description="Hash one file",
            steps=(
                create_workflow_step_definition(
                    step_id="hash",
                    tool_id="file-sha256",
                    static_parameters={"path": str(sample)},
                    position=0,
                ),
            ),
            tool_registry=tools,
        )
    )
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(
        tool_ids=["file-sha256"],
        root=sample_root,
    )
    repo.add_scope(scope)
    assert approval_required(tools.require("file-sha256"))

    result = executor.run(
        create_workflow_run_request(
            playbook_name="hash-once",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert result.status == "denied"
    assert counting.execute_calls == 0


def test_approval_fingerprint_and_expiry_and_denied(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = WorkflowRegistry(tool_registry=tools)
    sample_root = tmp_path / "root"
    sample_root.mkdir()
    sample = sample_root / "sample.txt"
    sample.write_text("abc", encoding="utf-8")
    workflows.register(
        create_workflow_definition(
            name="hash-once",
            version="1.0.0",
            description="Hash one file",
            steps=(
                create_workflow_step_definition(
                    step_id="hash",
                    tool_id="file-sha256",
                    static_parameters={"path": str(sample)},
                    position=0,
                ),
            ),
            tool_registry=tools,
        )
    )
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    scope = make_scope(tool_ids=["file-sha256"], root=sample_root)
    repo.add_scope(scope)

    good_request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(sample)},
        scope_id=scope.scope_id,
        justification="Workflow 'hash-once' step 'hash'",
        request_status="approved",
        dry_run_completed=True,
    )
    mismatched = create_tool_approval(
        request=good_request,
        decision="approved",
        reason="ok",
    )
    mismatched = replace(
        mismatched,
        request_fingerprint="0" * 64,
    )
    result = executor.run(
        create_workflow_run_request(
            playbook_name="hash-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"hash": good_request},
            step_approvals={"hash": mismatched},
        )
    )
    assert result.status == "denied"

    expired_request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(sample)},
        scope_id=scope.scope_id,
        justification="Workflow 'hash-once' step 'hash'",
        request_status="approved",
        dry_run_completed=True,
    )
    expired = create_tool_approval(
        request=expired_request,
        decision="approved",
        reason="ok",
        expires_at=(
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat().replace("+00:00", "Z"),
    )
    result = executor.run(
        create_workflow_run_request(
            playbook_name="hash-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"hash": expired_request},
            step_approvals={"hash": expired},
        )
    )
    assert result.status == "denied"

    rejected_request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(sample)},
        scope_id=scope.scope_id,
        justification="Workflow 'hash-once' step 'hash'",
        request_status="approved",
        dry_run_completed=True,
    )
    rejected = create_tool_approval(
        request=rejected_request,
        decision="rejected",
        reason="no",
    )
    result = executor.run(
        create_workflow_run_request(
            playbook_name="hash-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"hash": rejected_request},
            step_approvals={"hash": rejected},
        )
    )
    assert result.status == "denied"


def test_valid_step_approval_allows_execute(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = WorkflowRegistry(tool_registry=tools)
    sample_root = tmp_path / "root"
    sample_root.mkdir()
    sample = sample_root / "sample.txt"
    sample.write_text("abc", encoding="utf-8")
    workflows.register(
        create_workflow_definition(
            name="hash-once",
            version="1.0.0",
            description="Hash one file",
            steps=(
                create_workflow_step_definition(
                    step_id="hash",
                    tool_id="file-sha256",
                    static_parameters={"path": str(sample)},
                    position=0,
                ),
            ),
            tool_registry=tools,
        )
    )
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(tool_ids=["file-sha256"], root=sample_root)
    repo.add_scope(scope)

    tool_request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": str(sample)},
        scope_id=scope.scope_id,
        justification="Workflow 'hash-once' step 'hash'",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=tool_request,
        decision="approved",
        reason="approved for integrity check",
    )
    result = executor.run(
        create_workflow_run_request(
            playbook_name="hash-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"hash": tool_request},
            step_approvals={"hash": approval},
        )
    )
    assert result.status == "completed"
    assert counting.execute_calls == 1
    assert counting.plan_calls == 0


def test_tool_disabled_after_first_step_denies_next(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    original_require = tools.require

    def require_with_disable(tool_id: str) -> DefensiveToolDefinition:
        definition = original_require(tool_id)
        if counting.plan_calls >= 1 and tool_id == "simulated-log-check":
            return replace(definition, enabled=False)
        return definition

    tools.require = require_with_disable  # type: ignore[method-assign]
    executor._tool_executor = counting  # noqa: SLF001

    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "denied"
    assert counting.plan_calls == 1
    assert result.step_results[-1].status == "denied"


def test_cancellation_before_next_step(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = _two_step_registry(tools)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
    )
    counting = CountingToolExecutor()
    executor._tool_executor = counting  # noqa: SLF001
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)
    cancelled = {"value": False}

    def after_first() -> None:
        cancelled["value"] = True

    counting.after_first_plan = after_first
    result = executor.run(
        create_workflow_run_request(
            playbook_name="two-step",
            scope_id=scope.scope_id,
            dry_run=True,
        ),
        cancellation_check=lambda: cancelled["value"],
    )
    assert result.status == "cancelled"
    assert counting.plan_calls == 1
    assert counting.execute_calls == 0
