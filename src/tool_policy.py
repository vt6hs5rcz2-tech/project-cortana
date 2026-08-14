"""Centralized approval, dry-run, and capability kill-switch policy."""

from __future__ import annotations

from src.config import (
    ARBITRARY_SHELL_EXECUTION_ENABLED,
    AUTONOMOUS_REMEDIATION_ENABLED,
    EXTERNAL_TOOL_EXECUTION_ENABLED,
    TOOL_DRY_RUN_ENFORCEMENT_ENABLED,
    TOOL_HUMAN_APPROVAL_ENABLED,
)
from src.tool_common import (
    GATED_TOOL_CAPABILITY_CLASSES,
    READ_ONLY_TOOL_CAPABILITY_CLASSES,
    RESERVED_TOOL_CAPABILITY_CLASSES,
    ToolPolicyError,
)
from src.tool_definition import DefensiveToolDefinition
from src.tool_request import ToolExecutionRequest

# Maps a gated capability class to the kill-switch that must be True to execute.
# Read at assertion time so tests can patch these module globals.
_CAPABILITY_KILL_SWITCH_MESSAGES: dict[str, str] = {
    "arbitrary-shell": "Arbitrary shell execution is disabled [ENFORCED].",
    "external-process": "External tool execution is disabled [ENFORCED].",
    "autonomous-remediation": (
        "Autonomous remediation is disabled [ENFORCED]. "
        "This kill-switch does not disable user-explicit approved "
        "bounded read-only tools."
    ),
}


def capability_kill_switch_enabled(capability_class: str) -> bool:
    """Return whether the kill-switch for a gated capability class is on."""
    if capability_class == "arbitrary-shell":
        return ARBITRARY_SHELL_EXECUTION_ENABLED
    if capability_class == "external-process":
        return EXTERNAL_TOOL_EXECUTION_ENABLED
    if capability_class == "autonomous-remediation":
        return AUTONOMOUS_REMEDIATION_ENABLED
    return True


def approval_required(definition: DefensiveToolDefinition) -> bool:
    """Return whether explicit approval is required before execution.

    Gated capability classes (shell, external, autonomous remediation) always
    require a stored approval record. Conversational text cannot satisfy this.
    Enabling a kill-switch does not authorize unattended invocation.

    Bounded ``internal-readonly`` tools keep the existing risk-level policy.
    User-explicit approved file tools are not autonomous remediation.
    """
    if definition.capability_class in GATED_TOOL_CAPABILITY_CLASSES:
        return True
    if definition.capability_class == "internal-mutating":
        return True
    if not TOOL_HUMAN_APPROVAL_ENABLED:
        return False
    if definition.risk_level == "prohibited":
        return True
    if definition.risk_level in {"moderate", "high"}:
        return True
    if definition.risk_level == "low":
        # File-access tools require approval; definition flag remains authoritative.
        return bool(definition.requires_approval)
    return bool(definition.requires_approval)


def dry_run_required(definition: DefensiveToolDefinition) -> bool:
    """Return whether a completed dry run is required before execution."""
    if not TOOL_DRY_RUN_ENFORCEMENT_ENABLED:
        return False
    if definition.capability_class in GATED_TOOL_CAPABILITY_CLASSES:
        return True
    if definition.capability_class == "internal-mutating":
        return True
    return definition.risk_level in {
        "informational",
        "low",
        "moderate",
        "high",
    }


def tool_is_side_effecting(definition: DefensiveToolDefinition) -> bool:
    """Return whether a live workflow step must use explicit re-execution.

    Built-in ``internal-readonly`` tools remain replay-safe. Every other
    capability class is treated as side-effecting so a later mutating tool
    cannot be silently re-run from step 0.
    """
    return definition.capability_class not in READ_ONLY_TOOL_CAPABILITY_CLASSES


def assert_capability_allowed(definition: DefensiveToolDefinition) -> None:
    """Reject tools whose capability class is reserved or kill-switch disabled.

    This is the single execution-time capability gate. Registry registration
    does not bypass it; ``DefensiveToolExecutor.execute`` always calls
    ``assert_executable`` / ``assert_ready_for_execution``.
    """
    if definition.capability_class in RESERVED_TOOL_CAPABILITY_CLASSES:
        raise ToolPolicyError(
            "Tool AI-context injection is reserved and not implemented "
            "[DISABLED_BY_ARCHITECTURE]."
        )
    if definition.capability_class not in GATED_TOOL_CAPABILITY_CLASSES:
        return
    if capability_kill_switch_enabled(definition.capability_class):
        return
    raise ToolPolicyError(
        _CAPABILITY_KILL_SWITCH_MESSAGES[definition.capability_class]
    )


def assert_requestable(definition: DefensiveToolDefinition) -> None:
    """Reject tools that cannot be requested in this milestone."""
    if definition.risk_level == "prohibited":
        raise ToolPolicyError("Prohibited tools cannot be requested.")
    if not definition.enabled:
        raise ToolPolicyError("Disabled tools cannot be requested.")
    if definition.execution_mode == "future-external":
        raise ToolPolicyError(
            "Future-external tools cannot be requested in this milestone."
        )
    if definition.capability_class in RESERVED_TOOL_CAPABILITY_CLASSES:
        raise ToolPolicyError(
            "Tool AI-context injection is reserved and not implemented "
            "[DISABLED_BY_ARCHITECTURE]."
        )


def assert_executable(definition: DefensiveToolDefinition) -> None:
    """Reject tools that cannot execute in this milestone."""
    assert_requestable(definition)
    if definition.execution_mode == "future-external":
        raise ToolPolicyError(
            "Future-external tools cannot execute in this milestone."
        )
    if definition.risk_level == "high" and definition.execution_mode != "simulated":
        raise ToolPolicyError(
            "High-risk tools are simulation-only in this milestone."
        )
    if definition.execution_mode not in {"internal-python", "simulated"}:
        raise ToolPolicyError("Unsupported execution mode.")
    assert_capability_allowed(definition)


def assert_ready_for_execution(
    definition: DefensiveToolDefinition,
    request: ToolExecutionRequest,
) -> None:
    """Enforce dry-run, capability, and status prerequisites before execution."""
    assert_executable(definition)

    if request.request_status in {"rejected", "cancelled", "expired"}:
        raise ToolPolicyError(
            f"Request status '{request.request_status}' cannot execute."
        )
    if request.request_status in {"succeeded", "failed"}:
        raise ToolPolicyError("A completed request cannot run again.")

    if dry_run_required(definition) and not request.dry_run_completed:
        raise ToolPolicyError(
            "Dry run is required before execution for this tool."
        )

    if approval_required(definition):
        if request.request_status not in {"approved", "running"}:
            raise ToolPolicyError(
                "Explicit approval is required before execution."
            )
    elif request.request_status not in {"drafted", "approved", "running"}:
        raise ToolPolicyError(
            f"Request status '{request.request_status}' cannot execute."
        )


def initial_request_status(definition: DefensiveToolDefinition) -> str:
    """Return the initial request status after creation."""
    if approval_required(definition):
        return "awaiting-approval"
    return "drafted"
