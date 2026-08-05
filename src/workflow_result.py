"""Immutable workflow run and step result models."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import MAX_WORKFLOW_ERROR_MESSAGE_LENGTH
from src.tool_common import (
    utc_timestamp,
    validate_optional_utc_timestamp,
    validate_optional_uuid,
    validate_tool_id,
    validate_utc_timestamp,
    validate_uuid,
)
from src.tool_result import ToolExecutionResult, validate_tool_execution_result
from src.workflow_common import (
    WORKFLOW_TERMINAL_STATUSES,
    WorkflowPolicyError,
    WorkflowRunStatus,
    WorkflowStepStatus,
    WorkflowValidationError,
    normalize_safe_error_message,
    validate_playbook_name,
    validate_playbook_version,
    validate_step_id,
    validate_workflow_run_status,
    validate_workflow_step_status,
)


@dataclass(frozen=True)
class WorkflowStepResult:
    """Immutable result for one workflow step."""

    step_id: str
    tool_id: str
    position: int
    status: WorkflowStepStatus
    tool_result: ToolExecutionResult | None
    tool_request_id: str | None
    started_timestamp: str | None
    completed_timestamp: str | None
    error_code: str | None
    error_message: str | None
    dry_run: bool


@dataclass(frozen=True)
class WorkflowRunResult:
    """Immutable workflow run record and terminal/current result."""

    run_id: str
    playbook_name: str
    playbook_version: str
    dry_run: bool
    status: WorkflowRunStatus
    scope_id: str
    incident_id: str | None
    step_results: tuple[WorkflowStepResult, ...]
    created_timestamp: str
    started_timestamp: str | None
    completed_timestamp: str | None
    error_code: str | None
    error_message: str | None
    cancellation_requested: bool


_ALLOWED_RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset(
        {
            "preflight_failed",
            "running",
            "cancelled",
            "denied",
            "timed_out",
            "failed",
            "abandoned",
        }
    ),
    "preflight_failed": frozenset(),
    "running": frozenset(
        {
            "completed",
            "failed",
            "denied",
            "timed_out",
            "cancelled",
            "abandoned",
        }
    ),
    "completed": frozenset(),
    "failed": frozenset(),
    "denied": frozenset(),
    "timed_out": frozenset(),
    "cancelled": frozenset(),
    "abandoned": frozenset(),
}


def create_workflow_step_result(
    *,
    step_id: str,
    tool_id: str,
    position: int,
    status: str,
    tool_result: ToolExecutionResult | None = None,
    tool_request_id: str | None = None,
    started_timestamp: str | None = None,
    completed_timestamp: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    dry_run: bool = True,
) -> WorkflowStepResult:
    """Create one validated immutable workflow step result."""
    return validate_workflow_step_result(
        WorkflowStepResult(
            step_id=step_id,
            tool_id=tool_id,
            position=position,
            status=status,  # type: ignore[arg-type]
            tool_result=tool_result,
            tool_request_id=tool_request_id,
            started_timestamp=started_timestamp,
            completed_timestamp=completed_timestamp,
            error_code=error_code,
            error_message=error_message,
            dry_run=bool(dry_run),
        )
    )


def validate_workflow_step_result(result: WorkflowStepResult) -> WorkflowStepResult:
    """Validate one workflow step result."""
    if not isinstance(result.position, int) or isinstance(result.position, bool):
        raise WorkflowValidationError("Step position must be an integer.")
    if result.position < 0:
        raise WorkflowValidationError("Step position cannot be negative.")

    status = validate_workflow_step_status(result.status)
    tool_result = None
    if result.tool_result is not None:
        tool_result = validate_tool_execution_result(result.tool_result)
        if result.dry_run and tool_result.outcome == "succeeded":
            raise WorkflowValidationError(
                "Dry-run step results cannot report successful execution."
            )
        if result.dry_run and not tool_result.dry_run:
            raise WorkflowValidationError(
                "Dry-run step results must reference dry-run tool results."
            )

    request_id = None
    if result.tool_request_id is not None and result.tool_request_id.strip():
        request_id = validate_uuid(result.tool_request_id, field_name="Tool request ID")

    return WorkflowStepResult(
        step_id=validate_step_id(result.step_id),
        tool_id=validate_tool_id(result.tool_id),
        position=result.position,
        status=status,  # type: ignore[arg-type]
        tool_result=tool_result,
        tool_request_id=request_id,
        started_timestamp=validate_optional_utc_timestamp(
            result.started_timestamp,
            field_name="Step started timestamp",
        ),
        completed_timestamp=validate_optional_utc_timestamp(
            result.completed_timestamp,
            field_name="Step completed timestamp",
        ),
        error_code=normalize_safe_error_message(
            result.error_code,
            max_length=100,
        ),
        error_message=normalize_safe_error_message(
            result.error_message,
            max_length=MAX_WORKFLOW_ERROR_MESSAGE_LENGTH,
        ),
        dry_run=bool(result.dry_run),
    )


def create_workflow_run_result(
    *,
    run_id: str,
    playbook_name: str,
    playbook_version: str,
    dry_run: bool,
    status: str,
    scope_id: str,
    incident_id: str | None = None,
    step_results: list[WorkflowStepResult] | tuple[WorkflowStepResult, ...] = (),
    created_timestamp: str | None = None,
    started_timestamp: str | None = None,
    completed_timestamp: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    cancellation_requested: bool = False,
) -> WorkflowRunResult:
    """Create one validated immutable workflow run result."""
    return validate_workflow_run_result(
        WorkflowRunResult(
            run_id=run_id,
            playbook_name=playbook_name,
            playbook_version=playbook_version,
            dry_run=bool(dry_run),
            status=status,  # type: ignore[arg-type]
            scope_id=scope_id,
            incident_id=incident_id,
            step_results=tuple(step_results),
            created_timestamp=created_timestamp or utc_timestamp(),
            started_timestamp=started_timestamp,
            completed_timestamp=completed_timestamp,
            error_code=error_code,
            error_message=error_message,
            cancellation_requested=bool(cancellation_requested),
        )
    )


def validate_workflow_run_result(result: WorkflowRunResult) -> WorkflowRunResult:
    """Validate one workflow run result."""
    validated_steps = tuple(
        validate_workflow_step_result(step) for step in result.step_results
    )
    status = validate_workflow_run_status(result.status)
    if result.dry_run and status == "completed":
        for step in validated_steps:
            if step.tool_result is not None and step.tool_result.outcome == "succeeded":
                raise WorkflowValidationError(
                    "Dry-run workflow results cannot include successful tool execution."
                )

    return WorkflowRunResult(
        run_id=validate_uuid(result.run_id, field_name="Run ID"),
        playbook_name=validate_playbook_name(result.playbook_name),
        playbook_version=validate_playbook_version(result.playbook_version),
        dry_run=bool(result.dry_run),
        status=status,  # type: ignore[arg-type]
        scope_id=validate_uuid(result.scope_id, field_name="Scope ID"),
        incident_id=validate_optional_uuid(
            result.incident_id,
            field_name="Incident ID",
        ),
        step_results=validated_steps,
        created_timestamp=validate_utc_timestamp(
            result.created_timestamp,
            field_name="Run created timestamp",
        ),
        started_timestamp=validate_optional_utc_timestamp(
            result.started_timestamp,
            field_name="Run started timestamp",
        ),
        completed_timestamp=validate_optional_utc_timestamp(
            result.completed_timestamp,
            field_name="Run completed timestamp",
        ),
        error_code=normalize_safe_error_message(result.error_code, max_length=100),
        error_message=normalize_safe_error_message(
            result.error_message,
            max_length=MAX_WORKFLOW_ERROR_MESSAGE_LENGTH,
        ),
        cancellation_requested=bool(result.cancellation_requested),
    )


def transition_workflow_run_result(
    result: WorkflowRunResult,
    *,
    status: str,
    step_results: list[WorkflowStepResult] | tuple[WorkflowStepResult, ...] | None = None,
    started_timestamp: str | None = None,
    completed_timestamp: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    cancellation_requested: bool | None = None,
) -> WorkflowRunResult:
    """Return a replacement run result with a monotonic status transition."""
    next_status = validate_workflow_run_status(status)
    if next_status != result.status:
        allowed = _ALLOWED_RUN_TRANSITIONS.get(result.status, frozenset())
        if next_status not in allowed:
            raise WorkflowPolicyError(
                "Workflow run status cannot transition from "
                f"'{result.status}' to '{next_status}'."
            )
    if result.status in WORKFLOW_TERMINAL_STATUSES and next_status != result.status:
        raise WorkflowPolicyError("Terminal workflow runs cannot change status.")

    next_steps = (
        result.step_results if step_results is None else tuple(step_results)
    )
    return validate_workflow_run_result(
        WorkflowRunResult(
            run_id=result.run_id,
            playbook_name=result.playbook_name,
            playbook_version=result.playbook_version,
            dry_run=result.dry_run,
            status=next_status,  # type: ignore[arg-type]
            scope_id=result.scope_id,
            incident_id=result.incident_id,
            step_results=next_steps,
            created_timestamp=result.created_timestamp,
            started_timestamp=(
                result.started_timestamp
                if started_timestamp is None
                else started_timestamp
            ),
            completed_timestamp=(
                result.completed_timestamp
                if completed_timestamp is None
                else completed_timestamp
            ),
            error_code=result.error_code if error_code is None else error_code,
            error_message=(
                result.error_message if error_message is None else error_message
            ),
            cancellation_requested=(
                result.cancellation_requested
                if cancellation_requested is None
                else bool(cancellation_requested)
            ),
        )
    )


def append_step_result(
    result: WorkflowRunResult,
    step_result: WorkflowStepResult,
) -> WorkflowRunResult:
    """Append one immutable step result without rewriting prior history."""
    validated_step = validate_workflow_step_result(step_result)
    for existing in result.step_results:
        if existing.step_id == validated_step.step_id:
            raise WorkflowValidationError(
                f"Step result for '{validated_step.step_id}' already recorded."
            )
        if existing.position == validated_step.position:
            raise WorkflowValidationError(
                f"Step result position {validated_step.position} already recorded."
            )
    return validate_workflow_run_result(
        WorkflowRunResult(
            run_id=result.run_id,
            playbook_name=result.playbook_name,
            playbook_version=result.playbook_version,
            dry_run=result.dry_run,
            status=result.status,
            scope_id=result.scope_id,
            incident_id=result.incident_id,
            step_results=(*result.step_results, validated_step),
            created_timestamp=result.created_timestamp,
            started_timestamp=result.started_timestamp,
            completed_timestamp=result.completed_timestamp,
            error_code=result.error_code,
            error_message=result.error_message,
            cancellation_requested=result.cancellation_requested,
        )
    )
