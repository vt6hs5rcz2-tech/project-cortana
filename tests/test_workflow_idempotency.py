"""Side-effecting workflow idempotency and explicit re-execution tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.tool_approval import create_tool_approval
from src.tool_common import ToolValidationError
from src.tool_definition import create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_policy import approval_required, dry_run_required, tool_is_side_effecting
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_request import create_tool_execution_request
from src.workflow_common import (
    SIDE_EFFECT_CLAIM_FAILED_ERROR_CODE,
    SIDE_EFFECT_REEXECUTION_ERROR_CODE,
    WorkflowStorageError,
    compute_workflow_side_effect_key,
)
from src.workflow_definition import (
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import JsonWorkflowRunRepository
from src.workflow_request import create_workflow_run_request
from tests.tool_helpers import make_gated_test_tool, make_scope
from tests.workflow_helpers import make_executor


def _mutating_impl(calls: list[int]):
    def impl(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append(1)
        return {"ok": True}

    return impl


def _mutating_playbook(tmp_path: Path, calls: list[int], *, persist: bool = False):
    tools = ToolRegistry()
    tools.register(make_gated_test_tool(capability_class="internal-mutating"))
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="side-effect-once",
            version="1.0.0",
            description="Single side-effecting step for idempotency tests.",
            steps=(
                create_workflow_step_definition(
                    step_id="mutate",
                    tool_id="future-side-effect",
                    position=0,
                ),
            ),
            tool_registry=tools,
        )
    )
    run_repo = (
        JsonWorkflowRunRepository(tmp_path / "workflow_runs.json") if persist else None
    )
    executor, repo, _tools, _workflows, runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
        runs=run_repo,
    )
    executor._tool_executor = DefensiveToolExecutor(
        implementations={"impl_future_side_effect": _mutating_impl(calls)},
    )
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["future-side-effect"]))
    return executor, repo, scope, runs


def _approved_request(scope_id: str):
    tool_request = create_tool_execution_request(
        tool_id="future-side-effect",
        normalized_parameters={},
        scope_id=scope_id,
        justification="Workflow 'side-effect-once' step 'mutate'",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=tool_request,
        decision="approved",
        reason="operator approved mutating workflow step",
    )
    return tool_request, approval


def test_internal_mutating_requires_approval_at_definition() -> None:
    with pytest.raises(ToolValidationError, match="approval"):
        create_tool_definition(
            tool_id="future-side-effect",
            name="Future Side Effect",
            description="Must require approval.",
            category="diagnostics",
            version="1.0.0",
            risk_level="informational",
            execution_mode="internal-python",
            supported_objective_types=("inspect",),
            supported_target_types=("none",),
            parameter_schema=(),
            requires_approval=False,
            implementation_identifier="impl_future_side_effect",
            capability_class="internal-mutating",
        )


def test_internal_mutating_is_side_effecting_and_gated_for_approval() -> None:
    definition = make_gated_test_tool(capability_class="internal-mutating")
    assert tool_is_side_effecting(definition) is True
    assert approval_required(definition) is True
    assert dry_run_required(definition) is True
    readonly = build_default_tool_registry().require("system-summary")
    assert tool_is_side_effecting(readonly) is False


def test_first_approved_side_effecting_run_executes_once(tmp_path: Path) -> None:
    calls: list[int] = []
    executor, _repo, scope, _runs = _mutating_playbook(tmp_path, calls)
    tool_request, approval = _approved_request(scope.scope_id)
    result = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    assert result.status == "completed"
    assert calls == [1]


def test_accidental_rerun_does_not_execute_again(tmp_path: Path) -> None:
    calls: list[int] = []
    executor, _repo, scope, _runs = _mutating_playbook(tmp_path, calls)
    tool_request, approval = _approved_request(scope.scope_id)
    first = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    second = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    assert first.status == "completed"
    assert second.status == "denied"
    assert second.step_results[0].error_code == SIDE_EFFECT_REEXECUTION_ERROR_CODE
    assert calls == [1]


def test_intentional_second_operation_executes_with_new_operation_id(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    executor, _repo, scope, _runs = _mutating_playbook(tmp_path, calls)
    tool_request, approval = _approved_request(scope.scope_id)
    first = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    second = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
            operation_id=str(uuid4()),
        )
    )
    assert first.status == "completed"
    assert second.status == "completed"
    assert calls == [1, 1]


def test_side_effecting_run_still_requires_approval(tmp_path: Path) -> None:
    calls: list[int] = []
    executor, _repo, scope, _runs = _mutating_playbook(tmp_path, calls)
    result = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert result.status == "denied"
    assert calls == []


def test_side_effecting_dry_run_does_not_execute_or_claim(tmp_path: Path) -> None:
    calls: list[int] = []
    executor, _repo, scope, runs = _mutating_playbook(tmp_path, calls)
    first = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    second = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert first.status == "completed"
    assert second.status == "completed"
    assert calls == []
    assert runs.list_side_effect_markers() == []


def test_conversational_yes_is_not_workflow_approval(tmp_path: Path) -> None:
    calls: list[int] = []
    executor, _repo, scope, _runs = _mutating_playbook(tmp_path, calls)
    guidance = ConversationIntelligence().interpret("yes", ConversationState())
    assert guidance.authorizes_privileged_action is False
    result = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert result.status == "denied"
    assert calls == []


def test_kill_switch_still_blocks_gated_mutating_workflow(tmp_path: Path) -> None:
    calls: list[int] = []
    tools = ToolRegistry()
    tools.register(make_gated_test_tool(capability_class="autonomous-remediation"))
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="side-effect-once",
            version="1.0.0",
            description="Gated mutating step.",
            steps=(
                create_workflow_step_definition(
                    step_id="mutate",
                    tool_id="future-side-effect",
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
    executor._tool_executor = DefensiveToolExecutor(
        implementations={"impl_future_side_effect": _mutating_impl(calls)},
    )
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["future-side-effect"]))
    tool_request, approval = _approved_request(scope.scope_id)
    result = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    assert result.status in {"denied", "preflight_failed"}
    assert calls == []


def test_readonly_playbook_rerun_still_executes(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="readonly-once",
            version="1.0.0",
            description="Read-only replay-safe playbook.",
            steps=(
                create_workflow_step_definition(
                    step_id="summary",
                    tool_id="system-summary",
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
    counting = DefensiveToolExecutor()
    execute_calls = {"count": 0}
    original_execute = counting.execute

    def wrapped_execute(**kwargs: Any):  # type: ignore[no-untyped-def]
        execute_calls["count"] += 1
        return original_execute(**kwargs)

    counting.execute = wrapped_execute  # type: ignore[method-assign]
    executor._tool_executor = counting
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    first = executor.run(
        create_workflow_run_request(
            playbook_name="readonly-once",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    second = executor.run(
        create_workflow_run_request(
            playbook_name="readonly-once",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert first.status == "completed"
    assert second.status == "completed"
    assert execute_calls["count"] == 2


def test_completion_persist_failure_does_not_replay_side_effect(tmp_path: Path) -> None:
    calls: list[int] = []
    path = tmp_path / "workflow_runs.json"
    tools = ToolRegistry()
    tools.register(make_gated_test_tool(capability_class="internal-mutating"))
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="side-effect-once",
            version="1.0.0",
            description="Crash-after-effect playbook.",
            steps=(
                create_workflow_step_definition(
                    step_id="mutate",
                    tool_id="future-side-effect",
                    position=0,
                ),
            ),
            tool_registry=tools,
        )
    )

    class CompletionFailingRepository(JsonWorkflowRunRepository):
        def save_run(self, run):  # type: ignore[no-untyped-def]
            if run.status == "completed":
                raise WorkflowStorageError(
                    "Simulated completion persist failure.",
                    user_message="Cortana: Workflow repository data could not be updated safely.",
                )
            return super().save_run(run)

    failing = CompletionFailingRepository(path)
    executor, repo, _tools, _workflows, _runs, incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
        runs=failing,
    )
    executor._tool_executor = DefensiveToolExecutor(
        implementations={"impl_future_side_effect": _mutating_impl(calls)},
    )
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["future-side-effect"]))
    tool_request, approval = _approved_request(scope.scope_id)
    with pytest.raises(WorkflowStorageError):
        executor.run(
            create_workflow_run_request(
                playbook_name="side-effect-once",
                scope_id=scope.scope_id,
                dry_run=False,
                step_tool_requests={"mutate": tool_request},
                step_approvals={"mutate": approval},
            )
        )
    assert calls == [1]

    reconstructed = JsonWorkflowRunRepository(path)
    assert reconstructed.list_side_effect_markers()
    retry_executor = WorkflowExecutor(
        workflow_registry=workflows,
        tool_registry=tools,
        tool_executor=DefensiveToolExecutor(
            implementations={"impl_future_side_effect": _mutating_impl(calls)},
        ),
        tool_repository=repo,
        run_repository=reconstructed,
        incident_repository=incidents,
    )
    retry = retry_executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    assert retry.status == "denied"
    assert retry.step_results[0].error_code == SIDE_EFFECT_REEXECUTION_ERROR_CODE
    assert calls == [1]


def test_claim_persist_failure_does_not_execute(tmp_path: Path) -> None:
    calls: list[int] = []
    path = tmp_path / "workflow_runs.json"
    tools = ToolRegistry()
    tools.register(make_gated_test_tool(capability_class="internal-mutating"))
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="side-effect-once",
            version="1.0.0",
            description="Claim-failure playbook.",
            steps=(
                create_workflow_step_definition(
                    step_id="mutate",
                    tool_id="future-side-effect",
                    position=0,
                ),
            ),
            tool_registry=tools,
        )
    )

    class ClaimFailingRepository(JsonWorkflowRunRepository):
        def claim_side_effect_key(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            raise WorkflowStorageError(
                "Simulated claim persist failure.",
                user_message="Cortana: Workflow repository data could not be updated safely.",
            )

    failing = ClaimFailingRepository(path)
    executor, repo, _tools, _workflows, _runs, _incidents = make_executor(
        tmp_path,
        tools=tools,
        workflows=workflows,
        runs=failing,
    )
    executor._tool_executor = DefensiveToolExecutor(
        implementations={"impl_future_side_effect": _mutating_impl(calls)},
    )
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["future-side-effect"]))
    tool_request, approval = _approved_request(scope.scope_id)
    result = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    assert result.status == "denied"
    assert result.step_results[0].error_code == SIDE_EFFECT_CLAIM_FAILED_ERROR_CODE
    assert calls == []


def test_side_effect_key_is_stable_for_default_operation() -> None:
    first = compute_workflow_side_effect_key(
        playbook_name="side-effect-once",
        playbook_version="1.0.0",
        step_id="mutate",
        tool_id="future-side-effect",
        static_parameters={},
        operation_id=None,
    )
    second = compute_workflow_side_effect_key(
        playbook_name="side-effect-once",
        playbook_version="1.0.0",
        step_id="mutate",
        tool_id="future-side-effect",
        static_parameters={},
        operation_id=None,
    )
    other = compute_workflow_side_effect_key(
        playbook_name="side-effect-once",
        playbook_version="1.0.0",
        step_id="mutate",
        tool_id="future-side-effect",
        static_parameters={},
        operation_id=str(uuid4()),
    )
    assert first == second
    assert first != other
    assert len(first) == 64
