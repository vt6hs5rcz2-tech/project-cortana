"""Shared types and helpers for Milestone 10 defensive workflow orchestration."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Mapping, Protocol

from src.config import (
    MAX_WORKFLOW_NAME_LENGTH,
    MAX_WORKFLOW_VERSION_LENGTH,
)
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolIdError,
    ToolAuthorizationError,
    ToolPolicyError,
    ToolValidationError,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
)

WorkflowRunStatus = Literal[
    "pending",
    "preflight_failed",
    "running",
    "completed",
    "failed",
    "denied",
    "timed_out",
    "cancelled",
    "abandoned",
]
WorkflowStepStatus = Literal[
    "pending",
    "running",
    "planned",
    "completed",
    "failed",
    "denied",
    "timed_out",
    "cancelled",
    "skipped",
]
WorkflowAuditAction = Literal[
    "workflow-run-created",
    "workflow-preflight-accepted",
    "workflow-preflight-denied",
    "workflow-approval-requested",
    "workflow-approval-decision",
    "workflow-step-attempt",
    "workflow-step-dry-run",
    "workflow-step-result",
    "workflow-step-denied",
    "workflow-step-failed",
    "workflow-step-timed-out",
    "workflow-cancelled",
    "workflow-completed",
    "workflow-failed",
]

WORKFLOW_RUN_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "preflight_failed",
        "running",
        "completed",
        "failed",
        "denied",
        "timed_out",
        "cancelled",
        "abandoned",
    }
)
WORKFLOW_STEP_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "running",
        "planned",
        "completed",
        "failed",
        "denied",
        "timed_out",
        "cancelled",
        "skipped",
    }
)
WORKFLOW_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "workflow-run-created",
        "workflow-preflight-accepted",
        "workflow-preflight-denied",
        "workflow-approval-requested",
        "workflow-approval-decision",
        "workflow-step-attempt",
        "workflow-step-dry-run",
        "workflow-step-result",
        "workflow-step-denied",
        "workflow-step-failed",
        "workflow-step-timed-out",
        "workflow-cancelled",
        "workflow-completed",
        "workflow-failed",
    }
)
WORKFLOW_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "preflight_failed",
        "completed",
        "failed",
        "denied",
        "timed_out",
        "cancelled",
        "abandoned",
    }
)

ABANDONED_AFTER_RESTART_ERROR_CODE = "AbandonedAfterRestart"
ABANDONED_AFTER_RESTART_MESSAGE = (
    "Workflow run was abandoned after process restart and cannot be resumed."
)
DEFAULT_WORKFLOW_OPERATION_INSTANCE = "default"
SIDE_EFFECT_REEXECUTION_ERROR_CODE = "SideEffectReexecutionRequired"
SIDE_EFFECT_REEXECUTION_MESSAGE = (
    "This side-effecting workflow step was already attempted. "
    "A new explicit operation is required."
)
SIDE_EFFECT_CLAIM_FAILED_ERROR_CODE = "SideEffectClaimFailed"
SIDE_EFFECT_CLAIM_FAILED_MESSAGE = (
    "The side-effecting workflow step could not be recorded safely "
    "and was not executed."
)

PLAYBOOK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
PLAYBOOK_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
STEP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}$")


class WorkflowValidationError(ToolValidationError):
    """Raised when a workflow field or definition fails validation."""


class WorkflowAuthorizationError(ToolAuthorizationError):
    """Raised when workflow scope or approval checks deny an action."""


class WorkflowPolicyError(ToolPolicyError):
    """Raised when a workflow policy rule denies an action."""


class WorkflowStorageError(Exception):
    """Raised when in-memory workflow run storage cannot complete an operation."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or (
            "Cortana: Workflow run storage is unavailable."
        )


class WorkflowClock(Protocol):
    """Injectable clock for deterministic runtime-budget testing."""

    def monotonic(self) -> float:
        """Return a monotonic time value in seconds."""

    def utc_now_iso(self) -> str:
        """Return the current UTC timestamp in ISO 8601 form."""


class SystemWorkflowClock:
    """Production clock backed by the system monotonic and UTC clocks."""

    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now_iso(self) -> str:
        return utc_timestamp()


class ManualWorkflowClock:
    """Test clock with explicitly advanced monotonic time."""

    def __init__(self, *, start: float = 0.0) -> None:
        self._monotonic = float(start)
        self._utc = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self._monotonic

    def utc_now_iso(self) -> str:
        return (
            self._utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
        )

    def advance(self, seconds: float) -> None:
        """Advance the monotonic clock by the given number of seconds."""
        self._monotonic += float(seconds)


def compute_workflow_side_effect_key(
    *,
    playbook_name: str,
    playbook_version: str,
    step_id: str,
    tool_id: str,
    static_parameters: Mapping[str, Any],
    operation_id: str | None,
) -> str:
    """Return a stable SHA-256 key for one side-effecting workflow operation."""
    payload = {
        "playbook_name": playbook_name,
        "playbook_version": playbook_version,
        "step_id": step_id,
        "tool_id": tool_id,
        "static_parameters": dict(static_parameters),
        "operation_id": operation_id or DEFAULT_WORKFLOW_OPERATION_INSTANCE,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(blob.encode("utf-8")).hexdigest()


def validate_playbook_name(value: str) -> str:
    """Validate a stable trusted playbook name."""
    cleaned = require_non_blank_text(
        value,
        field_name="Playbook name",
        max_length=MAX_WORKFLOW_NAME_LENGTH,
    ).lower()
    if not PLAYBOOK_NAME_PATTERN.fullmatch(cleaned):
        raise InvalidToolIdError(
            "Playbook name must be a lowercase hyphenated identifier."
        )
    return cleaned


def validate_playbook_version(value: str) -> str:
    """Validate an explicit playbook version string."""
    cleaned = require_non_blank_text(
        value,
        field_name="Playbook version",
        max_length=MAX_WORKFLOW_VERSION_LENGTH,
    )
    if not PLAYBOOK_VERSION_PATTERN.fullmatch(cleaned):
        raise WorkflowValidationError(
            "Playbook version must use major.minor.patch numeric form."
        )
    return cleaned


def validate_step_id(value: str) -> str:
    """Validate a stable workflow step identifier."""
    cleaned = require_non_blank_text(
        value,
        field_name="Step ID",
        max_length=MAX_WORKFLOW_NAME_LENGTH,
    ).lower()
    if not STEP_ID_PATTERN.fullmatch(cleaned):
        raise InvalidToolIdError(
            "Step ID must be a lowercase hyphenated identifier."
        )
    return cleaned


def validate_workflow_run_status(value: str) -> str:
    """Validate a workflow run status value."""
    return validate_controlled_value(
        value,
        field_name="workflow run status",
        allowed=WORKFLOW_RUN_STATUSES,
    )


def validate_workflow_step_status(value: str) -> str:
    """Validate a workflow step status value."""
    return validate_controlled_value(
        value,
        field_name="workflow step status",
        allowed=WORKFLOW_STEP_STATUSES,
    )


def validate_workflow_audit_action(value: str) -> str:
    """Validate a workflow audit action value."""
    return validate_controlled_value(
        value,
        field_name="workflow audit action",
        allowed=WORKFLOW_AUDIT_ACTIONS,
    )


def normalize_safe_error_message(message: str | None, *, max_length: int) -> str | None:
    """Return a bounded safe error message without raw exception text."""
    if message is None:
        return None
    if not isinstance(message, str):
        raise WorkflowValidationError("Error message must be a string.")
    cleaned = " ".join(message.split())
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        return f"{cleaned[:max_length]}..."
    return cleaned


# Re-export commonly needed tool errors for workflow modules.
__all__ = [
    "BlankToolFieldError",
    "InvalidToolIdError",
    "ManualWorkflowClock",
    "SystemWorkflowClock",
    "WORKFLOW_AUDIT_ACTIONS",
    "WORKFLOW_RUN_STATUSES",
    "WORKFLOW_STEP_STATUSES",
    "WORKFLOW_TERMINAL_STATUSES",
    "WorkflowAuditAction",
    "WorkflowAuthorizationError",
    "WorkflowClock",
    "WorkflowPolicyError",
    "WorkflowRunStatus",
    "WorkflowStepStatus",
    "WorkflowStorageError",
    "WorkflowValidationError",
    "compute_workflow_side_effect_key",
    "normalize_safe_error_message",
    "validate_playbook_name",
    "validate_playbook_version",
    "validate_step_id",
    "validate_workflow_audit_action",
    "validate_workflow_run_status",
    "validate_workflow_step_status",
]
