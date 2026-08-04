"""Immutable tool execution request model and status transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from src.config import MAX_TOOL_JUSTIFICATION_LENGTH
from src.tool_common import (
    TOOL_REQUEST_STATUSES,
    ToolPolicyError,
    ToolRequestStatus,
    ToolValidationError,
    compute_request_fingerprint,
    json_safe_value,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_justification,
    validate_optional_uuid,
    validate_tool_id,
    validate_utc_timestamp,
    validate_uuid,
)


@dataclass(frozen=True)
class ToolExecutionRequest:
    """Immutable defensive tool execution request."""

    request_id: str
    tool_id: str
    normalized_parameters: dict[str, Any]
    scope_id: str
    requested_by: str
    requested_timestamp: str
    dry_run_requested: bool
    incident_id: str | None
    justification: str
    request_status: ToolRequestStatus
    fingerprint: str
    dry_run_completed: bool


_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "drafted": frozenset(
        {"awaiting-approval", "approved", "running", "cancelled", "expired"}
    ),
    "awaiting-approval": frozenset(
        {"approved", "rejected", "cancelled", "expired"}
    ),
    "approved": frozenset({"running", "cancelled", "expired"}),
    "rejected": frozenset(),
    "running": frozenset({"succeeded", "failed"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "expired": frozenset(),
    "cancelled": frozenset(),
}


def create_tool_execution_request(
    *,
    tool_id: str,
    normalized_parameters: dict[str, Any],
    scope_id: str,
    requested_by: str = "local-user",
    dry_run_requested: bool = False,
    incident_id: str | None = None,
    justification: str,
    request_status: str = "drafted",
    dry_run_completed: bool = False,
) -> ToolExecutionRequest:
    """Create a validated immutable tool execution request."""
    request_id = str(uuid4())
    cleaned_tool_id = validate_tool_id(tool_id)
    cleaned_scope_id = validate_uuid(scope_id, field_name="Scope ID")
    cleaned_incident = validate_optional_uuid(incident_id, field_name="Incident ID")
    cleaned_justification = validate_justification(justification)
    cleaned_by = require_non_blank_text(
        requested_by,
        field_name="Requested by",
        max_length=200,
    )
    safe_parameters = json_safe_value(normalized_parameters)
    if not isinstance(safe_parameters, dict):
        raise ToolValidationError("Normalized parameters must be an object.")

    fingerprint = compute_request_fingerprint(
        request_id=request_id,
        tool_id=cleaned_tool_id,
        normalized_parameters=safe_parameters,
        scope_id=cleaned_scope_id,
        incident_id=cleaned_incident,
        dry_run_requested=bool(dry_run_requested),
        justification=cleaned_justification,
    )
    return validate_tool_execution_request(
        ToolExecutionRequest(
            request_id=request_id,
            tool_id=cleaned_tool_id,
            normalized_parameters=safe_parameters,
            scope_id=cleaned_scope_id,
            requested_by=cleaned_by,
            requested_timestamp=utc_timestamp(),
            dry_run_requested=bool(dry_run_requested),
            incident_id=cleaned_incident,
            justification=cleaned_justification,
            request_status=request_status,  # type: ignore[arg-type]
            fingerprint=fingerprint,
            dry_run_completed=bool(dry_run_completed),
        )
    )


def replace_tool_execution_request(
    request: ToolExecutionRequest,
    *,
    request_status: str | None = None,
    dry_run_completed: bool | None = None,
) -> ToolExecutionRequest:
    """Return a validated replacement request with an allowed status transition."""
    next_status = (
        request.request_status if request_status is None else request_status
    )
    if next_status != request.request_status:
        _assert_transition_allowed(request.request_status, next_status)

    return validate_tool_execution_request(
        ToolExecutionRequest(
            request_id=request.request_id,
            tool_id=request.tool_id,
            normalized_parameters=dict(request.normalized_parameters),
            scope_id=request.scope_id,
            requested_by=request.requested_by,
            requested_timestamp=request.requested_timestamp,
            dry_run_requested=request.dry_run_requested,
            incident_id=request.incident_id,
            justification=request.justification,
            request_status=next_status,  # type: ignore[arg-type]
            fingerprint=request.fingerprint,
            dry_run_completed=(
                request.dry_run_completed
                if dry_run_completed is None
                else bool(dry_run_completed)
            ),
        )
    )


def validate_tool_execution_request(
    request: ToolExecutionRequest,
) -> ToolExecutionRequest:
    """Validate a tool execution request record."""
    request_id = validate_uuid(request.request_id, field_name="Request ID")
    tool_id = validate_tool_id(request.tool_id)
    scope_id = validate_uuid(request.scope_id, field_name="Scope ID")
    incident_id = validate_optional_uuid(
        request.incident_id,
        field_name="Incident ID",
    )
    justification = require_non_blank_text(
        request.justification,
        field_name="Justification",
        max_length=MAX_TOOL_JUSTIFICATION_LENGTH,
    )
    requested_by = require_non_blank_text(
        request.requested_by,
        field_name="Requested by",
        max_length=200,
    )
    requested_timestamp = validate_utc_timestamp(
        request.requested_timestamp,
        field_name="Requested timestamp",
    )
    status = validate_controlled_value(
        request.request_status,
        field_name="request status",
        allowed=TOOL_REQUEST_STATUSES,
    )
    safe_parameters = json_safe_value(request.normalized_parameters)
    if not isinstance(safe_parameters, dict):
        raise ToolValidationError("Normalized parameters must be an object.")

    expected_fingerprint = compute_request_fingerprint(
        request_id=request_id,
        tool_id=tool_id,
        normalized_parameters=safe_parameters,
        scope_id=scope_id,
        incident_id=incident_id,
        dry_run_requested=bool(request.dry_run_requested),
        justification=justification,
    )
    if request.fingerprint != expected_fingerprint:
        raise ToolValidationError(
            "Request fingerprint does not match bound request fields."
        )

    return ToolExecutionRequest(
        request_id=request_id,
        tool_id=tool_id,
        normalized_parameters=safe_parameters,
        scope_id=scope_id,
        requested_by=requested_by,
        requested_timestamp=requested_timestamp,
        dry_run_requested=bool(request.dry_run_requested),
        incident_id=incident_id,
        justification=justification,
        request_status=status,  # type: ignore[arg-type]
        fingerprint=expected_fingerprint,
        dry_run_completed=bool(request.dry_run_completed),
    )


def _assert_transition_allowed(current: str, next_status: str) -> None:
    """Reject invalid request status transitions."""
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if next_status not in allowed:
        raise ToolPolicyError(
            f"Request status cannot transition from '{current}' to '{next_status}'."
        )
