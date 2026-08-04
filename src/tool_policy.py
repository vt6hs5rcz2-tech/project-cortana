"""Centralized approval and dry-run policy for defensive tools."""

from __future__ import annotations

from src.config import TOOL_DRY_RUN_ENFORCEMENT_ENABLED, TOOL_HUMAN_APPROVAL_ENABLED
from src.tool_common import ToolPolicyError
from src.tool_definition import DefensiveToolDefinition
from src.tool_request import ToolExecutionRequest


def approval_required(definition: DefensiveToolDefinition) -> bool:
    """Return whether explicit approval is required before execution."""
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
    return definition.risk_level in {
        "informational",
        "low",
        "moderate",
        "high",
    }


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


def assert_ready_for_execution(
    definition: DefensiveToolDefinition,
    request: ToolExecutionRequest,
) -> None:
    """Enforce dry-run and status prerequisites before execution."""
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
