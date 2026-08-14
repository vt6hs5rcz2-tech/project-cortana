"""Capability-class kill-switches and future side-effecting tool safety."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.tool_approval import create_tool_approval
from src.tool_common import ToolPolicyError, ToolValidationError
from src.tool_definition import create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_policy import (
    approval_required,
    assert_executable,
    assert_ready_for_execution,
    assert_requestable,
    capability_kill_switch_enabled,
)
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_request import create_tool_execution_request, replace_tool_execution_request
from src.workflow_definition import (
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_registry import WorkflowRegistry
from src.workflow_request import create_workflow_run_request
from tests.tool_helpers import make_gated_test_tool, make_scope, tool_repository
from tests.workflow_helpers import make_executor


def _enable_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capability_class: str,
    enabled: bool,
) -> None:
    flag_name = {
        "arbitrary-shell": "ARBITRARY_SHELL_EXECUTION_ENABLED",
        "external-process": "EXTERNAL_TOOL_EXECUTION_ENABLED",
        "autonomous-remediation": "AUTONOMOUS_REMEDIATION_ENABLED",
    }[capability_class]
    monkeypatch.setattr(f"src.config.{flag_name}", enabled)
    monkeypatch.setattr(f"src.tool_policy.{flag_name}", enabled)


def _execute_gated(
    tmp_path: Path,
    *,
    capability_class: str,
    approval: bool,
    dry_run_completed: bool = True,
    request_status: str = "running",
) -> tuple[Any, list[str]]:
    calls: list[str] = []

    def impl(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append("executed")
        return {"mutated": True, "approved": True}

    definition = make_gated_test_tool(capability_class=capability_class)
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=[definition.tool_id]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id=definition.tool_id,
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="capability kill-switch matrix",
            request_status="awaiting-approval",
            dry_run_completed=dry_run_completed,
        )
    )
    approval_record = None
    live = request
    if approval:
        approval_record = create_tool_approval(
            request=request,
            decision="approved",
            reason="operator approved gated tool",
        )
        live = replace_tool_execution_request(live, request_status="approved")
        repo.update_request(live)
        if request_status == "running":
            live = replace_tool_execution_request(live, request_status="running")
            repo.update_request(live)
    elif request_status not in {live.request_status, "drafted"}:
        live = replace_tool_execution_request(live, request_status=request_status)
        repo.update_request(live)
    executor = DefensiveToolExecutor(
        implementations={"impl_future_side_effect": impl},
    )
    result = executor.execute(
        definition=definition,
        request=live,
        scope=scope,
        approval=approval_record,
    )
    return result, calls


@pytest.mark.parametrize(
    "capability_class,flag_attr",
    [
        ("arbitrary-shell", "ARBITRARY_SHELL_EXECUTION_ENABLED"),
        ("external-process", "EXTERNAL_TOOL_EXECUTION_ENABLED"),
        ("autonomous-remediation", "AUTONOMOUS_REMEDIATION_ENABLED"),
    ],
)
def test_kill_switch_defaults_false_and_blocks_direct_executor(
    tmp_path: Path,
    capability_class: str,
    flag_attr: str,
) -> None:
    import src.config as config

    assert getattr(config, flag_attr) is False
    assert capability_kill_switch_enabled(capability_class) is False
    result, calls = _execute_gated(
        tmp_path,
        capability_class=capability_class,
        approval=True,
    )
    assert result.outcome == "denied"
    assert calls == []
    assert result.error_class == "ToolPolicyError"


@pytest.mark.parametrize(
    "capability_class",
    ["arbitrary-shell", "external-process", "autonomous-remediation"],
)
def test_kill_switch_true_without_approval_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability_class: str,
) -> None:
    _enable_kill_switch(monkeypatch, capability_class=capability_class, enabled=True)
    result, calls = _execute_gated(
        tmp_path,
        capability_class=capability_class,
        approval=False,
        request_status="awaiting-approval",
    )
    assert result.outcome == "denied"
    assert calls == []


@pytest.mark.parametrize(
    "capability_class",
    ["arbitrary-shell", "external-process", "autonomous-remediation"],
)
def test_kill_switch_true_with_approval_executes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability_class: str,
) -> None:
    _enable_kill_switch(monkeypatch, capability_class=capability_class, enabled=True)
    result, calls = _execute_gated(
        tmp_path,
        capability_class=capability_class,
        approval=True,
    )
    assert result.outcome == "succeeded"
    assert calls == ["executed"]


def test_autonomous_flag_does_not_block_approved_readonly_file_tool(
    tmp_path: Path,
) -> None:
    from src.config import AUTONOMOUS_REMEDIATION_ENABLED

    assert AUTONOMOUS_REMEDIATION_ENABLED is False
    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    assert definition.capability_class == "internal-readonly"
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="readonly control",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    result = DefensiveToolExecutor().execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=None,
    )
    assert result.outcome == "succeeded"


def test_gated_tool_requires_approval_metadata() -> None:
    with pytest.raises(ToolValidationError, match="approval"):
        make_gated_test_tool(
            capability_class="autonomous-remediation",
            requires_approval=False,
        )


def test_reserved_ai_context_injection_cannot_be_enabled() -> None:
    with pytest.raises(ToolValidationError, match="Reserved"):
        create_tool_definition(
            tool_id="future-inject",
            name="Future Inject",
            description="Reserved AI-context injection probe.",
            category="diagnostics",
            version="1.0.0",
            risk_level="informational",
            execution_mode="internal-python",
            supported_objective_types=("inspect",),
            supported_target_types=("none",),
            parameter_schema=(),
            requires_approval=True,
            enabled=True,
            implementation_identifier="impl_future_inject",
            capability_class="ai-context-injection",
        )


def test_reserved_ai_context_injection_disabled_cannot_execute() -> None:
    definition = create_tool_definition(
        tool_id="future-inject",
        name="Future Inject",
        description="Reserved AI-context injection probe.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=True,
        enabled=False,
        implementation_identifier="impl_future_inject",
        capability_class="ai-context-injection",
    )
    with pytest.raises(ToolPolicyError, match="Disabled"):
        assert_requestable(definition)
    from dataclasses import replace

    enabled = replace(definition, enabled=True)
    with pytest.raises(ToolPolicyError, match="reserved"):
        assert_requestable(enabled)
    with pytest.raises(ToolPolicyError, match="reserved"):
        assert_executable(enabled)


def test_conversational_yes_does_not_approve_gated_tool(tmp_path: Path) -> None:
    intel = ConversationIntelligence()
    state = ConversationState()
    guidance = intel.interpret("yes", state)
    assert guidance.authorizes_privileged_action is False
    result, calls = _execute_gated(
        tmp_path,
        capability_class="autonomous-remediation",
        approval=False,
        request_status="awaiting-approval",
    )
    assert result.outcome == "denied"
    assert calls == []


def test_tool_result_cannot_self_authorize_another_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_kill_switch(
        monkeypatch,
        capability_class="autonomous-remediation",
        enabled=True,
    )
    first, calls = _execute_gated(
        tmp_path,
        capability_class="autonomous-remediation",
        approval=True,
    )
    assert first.outcome == "succeeded"
    assert first.structured_data.get("approved") is True
    second, second_calls = _execute_gated(
        tmp_path,
        capability_class="autonomous-remediation",
        approval=False,
        request_status="awaiting-approval",
    )
    assert second.outcome == "denied"
    assert second_calls == []
    assert calls == ["executed"]


def test_dry_run_still_required_for_gated_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_kill_switch(
        monkeypatch,
        capability_class="autonomous-remediation",
        enabled=True,
    )
    definition = make_gated_test_tool(capability_class="autonomous-remediation")
    request = create_tool_execution_request(
        tool_id=definition.tool_id,
        normalized_parameters={},
        scope_id="66666666-6666-6666-6666-666666666666",
        justification="dry-run gate",
        request_status="approved",
        dry_run_completed=False,
    )
    with pytest.raises(ToolPolicyError, match="Dry run"):
        assert_ready_for_execution(definition, request)
    assert approval_required(definition) is True


def test_capability_is_not_inferred_from_tool_id(tmp_path: Path) -> None:
    definition = create_tool_definition(
        tool_id="arbitrary-shell-named",
        name="Named Like Shell",
        description="ID looks dangerous; capability is internal-readonly.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        implementation_identifier="impl_named_like_shell",
        capability_class="internal-readonly",
    )
    calls: list[str] = []

    def impl(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append("executed")
        return {"ok": True}

    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=[definition.tool_id]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id=definition.tool_id,
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="id is not capability",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    result = DefensiveToolExecutor(
        implementations={"impl_named_like_shell": impl},
    ).execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=None,
    )
    assert result.outcome == "succeeded"
    assert calls == ["executed"]


def _mutating_workflow(tmp_path: Path) -> tuple[Any, Any, Any, list[int]]:
    calls: list[int] = []

    def impl(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append(1)
        return {"ok": True}

    tools = ToolRegistry()
    tools.register(make_gated_test_tool(capability_class="autonomous-remediation"))
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="side-effect-once",
            version="1.0.0",
            description="Single gated mutating step.",
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
        implementations={"impl_future_side_effect": impl},
    )
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["future-side-effect"]))
    return executor, repo, scope, calls


def test_workflow_kill_switch_false_blocks_mutating_step(tmp_path: Path) -> None:
    executor, _repo, scope, calls = _mutating_workflow(tmp_path)
    result = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
        )
    )
    assert result.status in {"denied", "failed", "preflight_failed"}
    assert calls == []


def test_workflow_kill_switch_true_with_approval_executes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_kill_switch(
        monkeypatch,
        capability_class="autonomous-remediation",
        enabled=True,
    )
    executor, _repo, scope, calls = _mutating_workflow(tmp_path)
    tool_request = create_tool_execution_request(
        tool_id="future-side-effect",
        normalized_parameters={},
        scope_id=scope.scope_id,
        justification="Workflow 'side-effect-once' step 'mutate'",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=tool_request,
        decision="approved",
        reason="operator approved mutating workflow step",
    )
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


def test_workflow_rerun_still_enforces_kill_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_kill_switch(
        monkeypatch,
        capability_class="autonomous-remediation",
        enabled=True,
    )
    executor, _repo, scope, calls = _mutating_workflow(tmp_path)
    tool_request = create_tool_execution_request(
        tool_id="future-side-effect",
        normalized_parameters={},
        scope_id=scope.scope_id,
        justification="Workflow 'side-effect-once' step 'mutate'",
        request_status="approved",
        dry_run_completed=True,
    )
    approval = create_tool_approval(
        request=tool_request,
        decision="approved",
        reason="operator approved mutating workflow step",
    )
    first = executor.run(
        create_workflow_run_request(
            playbook_name="side-effect-once",
            scope_id=scope.scope_id,
            dry_run=False,
            step_tool_requests={"mutate": tool_request},
            step_approvals={"mutate": approval},
        )
    )
    assert first.status == "completed"
    assert calls == [1]
    _enable_kill_switch(
        monkeypatch,
        capability_class="autonomous-remediation",
        enabled=False,
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
    assert second.status in {"denied", "failed", "preflight_failed"}
    assert calls == [1]


def test_registry_registration_does_not_bypass_execution_gate(
    tmp_path: Path,
) -> None:
    definition = make_gated_test_tool(capability_class="arbitrary-shell")
    registry = ToolRegistry()
    registry.register(definition)
    assert registry.require(definition.tool_id).capability_class == "arbitrary-shell"
    with pytest.raises(ToolPolicyError, match="shell"):
        assert_executable(definition)
    result, calls = _execute_gated(
        tmp_path,
        capability_class="arbitrary-shell",
        approval=True,
    )
    assert result.outcome == "denied"
    assert calls == []
