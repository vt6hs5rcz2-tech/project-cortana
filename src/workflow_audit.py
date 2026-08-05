"""Append-only audit entries for defensive workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from src.config import MAX_WORKFLOW_AUDIT_DETAILS_LENGTH
from src.tool_audit import assert_audit_details_are_safe
from src.tool_common import (
    json_safe_value,
    require_non_blank_text,
    utc_timestamp,
    validate_optional_uuid,
    validate_tool_id,
    validate_utc_timestamp,
    validate_uuid,
)
from src.workflow_common import (
    WorkflowAuditAction,
    WorkflowValidationError,
    validate_playbook_name,
    validate_step_id,
    validate_workflow_audit_action,
)


@dataclass(frozen=True)
class WorkflowAuditEntry:
    """Immutable append-only workflow audit entry with safe details only."""

    audit_id: str
    timestamp: str
    action: WorkflowAuditAction
    actor: str
    run_id: str | None
    playbook_name: str | None
    step_id: str | None
    tool_id: str | None
    scope_id: str | None
    approval_id: str | None
    result_id: str | None
    safe_details: dict[str, Any]


def create_workflow_audit_entry(
    *,
    action: str,
    actor: str = "local-user",
    run_id: str | None = None,
    playbook_name: str | None = None,
    step_id: str | None = None,
    tool_id: str | None = None,
    scope_id: str | None = None,
    approval_id: str | None = None,
    result_id: str | None = None,
    safe_details: dict[str, Any] | None = None,
) -> WorkflowAuditEntry:
    """Create a validated immutable workflow audit entry."""
    return validate_workflow_audit_entry(
        WorkflowAuditEntry(
            audit_id=str(uuid4()),
            timestamp=utc_timestamp(),
            action=action,  # type: ignore[arg-type]
            actor=actor,
            run_id=run_id,
            playbook_name=playbook_name,
            step_id=step_id,
            tool_id=tool_id,
            scope_id=scope_id,
            approval_id=approval_id,
            result_id=result_id,
            safe_details=dict(safe_details or {}),
        )
    )


def validate_workflow_audit_entry(entry: WorkflowAuditEntry) -> WorkflowAuditEntry:
    """Validate one workflow audit entry using Milestone 9 forbidden-key rules."""
    audit_id = validate_uuid(entry.audit_id, field_name="Audit ID")
    timestamp = validate_utc_timestamp(entry.timestamp, field_name="Audit timestamp")
    action = validate_workflow_audit_action(entry.action)
    actor = require_non_blank_text(entry.actor, field_name="Actor", max_length=200)
    run_id = validate_optional_uuid(entry.run_id, field_name="Run ID")
    scope_id = validate_optional_uuid(entry.scope_id, field_name="Scope ID")
    approval_id = validate_optional_uuid(entry.approval_id, field_name="Approval ID")
    result_id = validate_optional_uuid(entry.result_id, field_name="Result ID")

    playbook_name = None
    if entry.playbook_name is not None and entry.playbook_name.strip():
        playbook_name = validate_playbook_name(entry.playbook_name)

    step_id = None
    if entry.step_id is not None and entry.step_id.strip():
        step_id = validate_step_id(entry.step_id)

    tool_id = None
    if entry.tool_id is not None and entry.tool_id.strip():
        tool_id = validate_tool_id(entry.tool_id)

    details = json_safe_value(entry.safe_details)
    if not isinstance(details, dict):
        raise WorkflowValidationError("Audit safe_details must be an object.")
    assert_audit_details_are_safe(
        details,
        max_value_length=MAX_WORKFLOW_AUDIT_DETAILS_LENGTH,
    )
    serialized_length = len(str(details))
    if serialized_length > MAX_WORKFLOW_AUDIT_DETAILS_LENGTH * 4:
        raise WorkflowValidationError("Audit safe_details exceed the allowed size.")

    return WorkflowAuditEntry(
        audit_id=audit_id,
        timestamp=timestamp,
        action=action,  # type: ignore[arg-type]
        actor=actor,
        run_id=run_id,
        playbook_name=playbook_name,
        step_id=step_id,
        tool_id=tool_id,
        scope_id=scope_id,
        approval_id=approval_id,
        result_id=result_id,
        safe_details=details,
    )
