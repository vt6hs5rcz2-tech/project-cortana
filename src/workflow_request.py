"""Immutable workflow run request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import uuid4

from src.tool_approval import ToolApprovalRecord, validate_tool_approval
from src.tool_common import (
    require_non_blank_text,
    utc_timestamp,
    validate_uuid,
)
from src.tool_request import ToolExecutionRequest, validate_tool_execution_request
from src.workflow_common import (
    WorkflowValidationError,
    validate_playbook_name,
    validate_playbook_version,
)


@dataclass(frozen=True)
class WorkflowRunRequest:
    """Immutable request to dry-run or execute one trusted playbook."""

    run_id: str
    playbook_name: str
    playbook_version: str | None
    scope_id: str
    requested_by: str
    created_timestamp: str
    dry_run: bool
    step_tool_requests: dict[str, ToolExecutionRequest]
    step_approvals: dict[str, ToolApprovalRecord]
    cancellation_requested: bool


def create_workflow_run_request(
    *,
    playbook_name: str,
    scope_id: str,
    playbook_version: str | None = None,
    requested_by: str = "local-user",
    dry_run: bool = True,
    step_tool_requests: Mapping[str, ToolExecutionRequest] | None = None,
    step_approvals: Mapping[str, ToolApprovalRecord] | None = None,
    cancellation_requested: bool = False,
    run_id: str | None = None,
) -> WorkflowRunRequest:
    """Create a validated immutable workflow run request.

    Dry-run is the default. Real execution requires ``dry_run=False``.
    """
    cleaned_run_id = validate_uuid(run_id or str(uuid4()), field_name="Run ID")
    cleaned_name = validate_playbook_name(playbook_name)
    cleaned_version = None
    if playbook_version is not None and playbook_version.strip():
        cleaned_version = validate_playbook_version(playbook_version)
    cleaned_scope = validate_uuid(scope_id, field_name="Scope ID")
    cleaned_by = require_non_blank_text(
        requested_by,
        field_name="Requested by",
        max_length=200,
    )

    requests: dict[str, ToolExecutionRequest] = {}
    for step_id, request in dict(step_tool_requests or {}).items():
        if not isinstance(step_id, str) or not step_id.strip():
            raise WorkflowValidationError("Step tool request keys must be step IDs.")
        if not isinstance(request, ToolExecutionRequest):
            raise WorkflowValidationError(
                "Prepared step tool requests must be ToolExecutionRequest objects."
            )
        requests[step_id.strip().lower()] = validate_tool_execution_request(request)

    approvals: dict[str, ToolApprovalRecord] = {}
    for step_id, approval in dict(step_approvals or {}).items():
        if not isinstance(step_id, str) or not step_id.strip():
            raise WorkflowValidationError("Step approval keys must be step IDs.")
        if not isinstance(approval, ToolApprovalRecord):
            raise WorkflowValidationError(
                "Step approvals must be ToolApprovalRecord objects."
            )
        approvals[step_id.strip().lower()] = validate_tool_approval(approval)

    return WorkflowRunRequest(
        run_id=cleaned_run_id,
        playbook_name=cleaned_name,
        playbook_version=cleaned_version,
        scope_id=cleaned_scope,
        requested_by=cleaned_by,
        created_timestamp=utc_timestamp(),
        dry_run=bool(dry_run),
        step_tool_requests=requests,
        step_approvals=approvals,
        cancellation_requested=bool(cancellation_requested),
    )
