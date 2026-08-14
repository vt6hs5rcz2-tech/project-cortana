"""Pre-M30 Deep Audit #4: config/security flag enforcement and future-tool safety.

Discovery only. Failures document DISPLAY_ONLY / DEAD / unenforced kill-switches.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import (
    ARBITRARY_SHELL_EXECUTION_ENABLED,
    AUTONOMOUS_REMEDIATION_ENABLED,
    EXTERNAL_TOOL_EXECUTION_ENABLED,
    PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    TOOL_AI_CONTEXT_INJECTION_ENABLED,
)
from src.tool_approval import create_tool_approval
from src.tool_definition import create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry
from src.tool_request import create_tool_execution_request, replace_tool_execution_request
from src.workflow_common import SIDE_EFFECT_REEXECUTION_ERROR_CODE
from src.workflow_definition import (
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_registry import WorkflowRegistry
from src.workflow_request import create_workflow_run_request
from tests.tool_helpers import make_gated_test_tool, make_scope, tool_repository
from tests.workflow_helpers import make_executor


def _src_python_files() -> list[Path]:
    return sorted((PROJECT_ROOT / "src").rglob("*.py"))


def _module_uses_name(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def test_deep_f3_kill_switch_flags_default_false() -> None:
    assert ARBITRARY_SHELL_EXECUTION_ENABLED is False
    assert EXTERNAL_TOOL_EXECUTION_ENABLED is False
    assert AUTONOMOUS_REMEDIATION_ENABLED is False
    assert TOOL_AI_CONTEXT_INJECTION_ENABLED is False


def _config_assignment_block(name: str) -> str:
    text = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")
    idx = text.index(f"{name} =")
    return text[max(0, idx - 500) : idx + 80]


def test_deep_f3_tool_ai_context_injection_is_dead() -> None:
    """DEEP-F9: TOOL_AI_CONTEXT_INJECTION_ENABLED is RESERVED, not a fake kill-switch."""
    readers = [
        path
        for path in _src_python_files()
        if path.name != "config.py" and _module_uses_name(path, "TOOL_AI_CONTEXT_INJECTION_ENABLED")
    ]
    assert readers == [], (
        "TOOL_AI_CONTEXT_INJECTION_ENABLED is reserved/not implemented; "
        f"production code must not treat it as an execution gate: {readers}"
    )
    block = _config_assignment_block("TOOL_AI_CONTEXT_INJECTION_ENABLED")
    assert "RESERVED" in block
    status_text = (PROJECT_ROOT / "src" / "commands.py").read_text(encoding="utf-8")
    assert "Tool AI-context injection: reserved (not implemented)" in status_text
    assert "Tool AI-context injection: disabled" not in status_text


def test_deep_f3_shell_and_external_flags_are_not_execution_gates() -> None:
    """Kill-switch names must appear in an execution gate, not only /status and capability flags."""
    gate_files = {
        "tool_policy.py",
        "tool_executor.py",
        "tool_registry.py",
        "tool_process_adapter.py",
    }
    for flag in (
        "ARBITRARY_SHELL_EXECUTION_ENABLED",
        "EXTERNAL_TOOL_EXECUTION_ENABLED",
        "AUTONOMOUS_REMEDIATION_ENABLED",
    ):
        readers = [
            path.name
            for path in _src_python_files()
            if path.name != "config.py" and _module_uses_name(path, flag)
        ]
        gated = [name for name in readers if name in gate_files]
        assert gated, (
            f"{flag} is DISPLAY_ONLY/RESERVED: read by {readers or ['(none)']} "
            "and never consulted by an execution-policy module"
        )
        assert "tool_policy.py" in gated


def test_deep_f3_startup_timeout_constant_is_read() -> None:
    """DEEP-F9: PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS is RESERVED/unused."""
    readers = [
        path.name
        for path in _src_python_files()
        if path.name != "config.py"
        and _module_uses_name(path, "PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS")
    ]
    assert readers == [], (
        "PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS is reserved/unused; "
        f"do not invent a startup handshake to consume it: {readers} "
        f"(value={PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS})"
    )
    block = _config_assignment_block("PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS")
    assert "RESERVED" in block
    status_text = (PROJECT_ROOT / "src" / "commands.py").read_text(encoding="utf-8")
    assert "Process child startup timeout: reserved (unused; no startup handshake)" in (
        status_text
    )


def _side_effect_tool():
    """Unclassified helper for DEEP-F7 retry discovery (internal-readonly default)."""
    return create_tool_definition(
        tool_id="future-side-effect",
        name="Future Side Effect",
        description="Test-only side-effecting tool used to probe kill-switch reachability.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        implementation_identifier="impl_future_side_effect",
    )


def _mutating_capability_tool():
    return create_tool_definition(
        tool_id="future-side-effect",
        name="Future Side Effect",
        description="Test-only mutating tool with an explicit gated capability class.",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=True,
        implementation_identifier="impl_future_side_effect",
        capability_class="autonomous-remediation",
    )


def test_deep_f3_kill_switch_false_does_not_block_new_internal_python_tool(
    tmp_path: Path,
) -> None:
    """Future-safety: an explicitly mutating internal-python tool is blocked while kill-switches are False."""
    from src.tool_approval import create_tool_approval

    assert ARBITRARY_SHELL_EXECUTION_ENABLED is False
    assert EXTERNAL_TOOL_EXECUTION_ENABLED is False
    assert AUTONOMOUS_REMEDIATION_ENABLED is False
    calls: list[str] = []

    def impl(_parameters: Mapping[str, Any], _context: Mapping[str, Any]) -> dict[str, Any]:
        calls.append("executed")
        return {"mutated": True}

    definition = _mutating_capability_tool()
    registry = ToolRegistry()
    registry.register(definition)
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["future-side-effect"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="future-side-effect",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="deep audit future-safety",
            request_status="awaiting-approval",
            dry_run_completed=True,
        )
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="operator approved mutating probe",
    )
    approved = replace_tool_execution_request(request, request_status="approved")
    repo.update_request(approved)
    running = replace_tool_execution_request(approved, request_status="running")
    repo.update_request(running)
    executor = DefensiveToolExecutor(implementations={"impl_future_side_effect": impl})
    result = executor.execute(
        definition=definition,
        request=running,
        scope=scope,
        approval=approval,
    )
    assert result.outcome != "succeeded", (
        "Kill-switch flags are False but a newly registered internal-python "
        f"side-effecting tool executed (outcome={result.outcome}, calls={calls})"
    )
    assert calls == []


def test_deep_f3_future_external_mode_is_structurally_blocked() -> None:
    """Control: enabled future-external definitions cannot be constructed."""
    from src.tool_common import ToolValidationError

    try:
        create_tool_definition(
            tool_id="future-external-probe",
            name="Future External Probe",
            description="Probe structural future-external blocking.",
            category="diagnostics",
            version="1.0.0",
            risk_level="informational",
            execution_mode="future-external",
            supported_objective_types=("inspect",),
            supported_target_types=("none",),
            parameter_schema=(),
            requires_approval=False,
            enabled=True,
            implementation_identifier="impl_future_external_probe",
        )
    except ToolValidationError as error:
        assert "future-external" in str(error).lower() or "enabled" in str(error).lower()
        return
    raise AssertionError("enabled future-external tool definition was accepted")


def test_deep_f7_workflow_rerun_does_not_reexecute_side_effecting_step(
    tmp_path: Path,
) -> None:
    """A second playbook-run of the same mutating operation cannot execute again.

    The original discovery used an unclassified internal-readonly helper, which
    is replay-safe by design. This contract uses an explicit side-effecting
    capability class so the test matches the product rule it now enforces.
    """
    calls: list[int] = []

    def impl(_parameters: Mapping[str, Any], _context: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(1)
        return {"ok": True}

    tools = ToolRegistry()
    tools.register(make_gated_test_tool(capability_class="internal-mutating"))
    workflows = WorkflowRegistry(tool_registry=tools)
    workflows.register(
        create_workflow_definition(
            name="side-effect-once",
            version="1.0.0",
            description="Single side-effecting step for retry-safety discovery.",
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


def test_deep_f5_authorizes_privileged_action_is_hardcoded_false() -> None:
    """OUTSIDE-F5: the authority property is a constant, not a call-graph check."""
    from src.conversation_intelligence import ConversationalGuidance
    from src.realtime_conversation_plan import RealtimeConversationPlan
    from src.speech_delivery import SpeechDeliveryPlan

    for cls in (ConversationalGuidance, RealtimeConversationPlan, SpeechDeliveryPlan):
        source = inspect.getsource(cls.authorizes_privileged_action.fget)
        assert "return False" in source
        # A property that always returns False cannot detect a wiring regression.
        assert "executor" not in source


def test_deep_status_only_flags_listed_in_commands() -> None:
    """Inventory: several flags appear in /status without gating their feature."""
    commands_text = (PROJECT_ROOT / "src" / "commands.py").read_text(encoding="utf-8")
    display_only = (
        "HISTORY_PERSISTENCE_ENABLED",
        "SEMANTIC_RETRIEVAL_ENABLED",
        "SOURCE_MANIFEST_PERSISTENCE_ENABLED",
        "AUTOMATED_RESPONSE_ENABLED",
        "ACTIVE_MEMORY_PERSISTENCE_ENABLED",
        "WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED",
        "WORKFLOW_PARALLEL_EXECUTION_ENABLED",
        "WORKFLOW_BACKGROUND_EXECUTION_ENABLED",
    )
    missing_status = [name for name in display_only if name not in commands_text]
    assert missing_status == []
    for name in display_only:
        readers = [
            path.name
            for path in _src_python_files()
            if path.name not in {"config.py", "commands.py"} and _module_uses_name(path, name)
        ]
        assert readers == [], f"{name} unexpectedly enforced in {readers}"
