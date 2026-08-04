"""Immutable tool approval records with fingerprint binding."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.config import MAX_TOOL_APPROVAL_REASON_LENGTH
from src.tool_common import (
    TOOL_APPROVAL_DECISIONS,
    ToolApprovalDecision,
    ToolAuthorizationError,
    ToolValidationError,
    is_expired,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_utc_timestamp,
    validate_sha256_digest,
    validate_utc_timestamp,
    validate_uuid,
)
from src.tool_request import ToolExecutionRequest


@dataclass(frozen=True)
class ToolApprovalRecord:
    """Immutable approval or rejection bound to a request fingerprint."""

    approval_id: str
    request_id: str
    decision: ToolApprovalDecision
    approver: str
    decision_timestamp: str
    reason: str
    request_fingerprint: str
    expires_at: str | None


def create_tool_approval(
    *,
    request: ToolExecutionRequest,
    decision: str,
    reason: str,
    approver: str = "local-user",
    expires_at: str | None = None,
) -> ToolApprovalRecord:
    """Create a validated approval record bound to the request fingerprint."""
    cleaned_decision = validate_controlled_value(
        decision,
        field_name="approval decision",
        allowed=TOOL_APPROVAL_DECISIONS,
    )
    cleaned_reason = require_non_blank_text(
        reason,
        field_name="Approval reason",
        max_length=MAX_TOOL_APPROVAL_REASON_LENGTH,
    )
    cleaned_approver = require_non_blank_text(
        approver,
        field_name="Approver",
        max_length=200,
    )
    return validate_tool_approval(
        ToolApprovalRecord(
            approval_id=str(uuid4()),
            request_id=request.request_id,
            decision=cleaned_decision,  # type: ignore[arg-type]
            approver=cleaned_approver,
            decision_timestamp=utc_timestamp(),
            reason=cleaned_reason,
            request_fingerprint=request.fingerprint,
            expires_at=expires_at,
        )
    )


def validate_tool_approval(approval: ToolApprovalRecord) -> ToolApprovalRecord:
    """Validate one tool approval record."""
    approval_id = validate_uuid(approval.approval_id, field_name="Approval ID")
    request_id = validate_uuid(approval.request_id, field_name="Request ID")
    decision = validate_controlled_value(
        approval.decision,
        field_name="approval decision",
        allowed=TOOL_APPROVAL_DECISIONS,
    )
    approver = require_non_blank_text(
        approval.approver,
        field_name="Approver",
        max_length=200,
    )
    reason = require_non_blank_text(
        approval.reason,
        field_name="Approval reason",
        max_length=MAX_TOOL_APPROVAL_REASON_LENGTH,
    )
    decision_timestamp = validate_utc_timestamp(
        approval.decision_timestamp,
        field_name="Decision timestamp",
    )
    fingerprint = validate_sha256_digest(
        approval.request_fingerprint,
        field_name="Request fingerprint",
    )
    expires_at = validate_optional_utc_timestamp(
        approval.expires_at,
        field_name="Approval expiry",
    )
    return ToolApprovalRecord(
        approval_id=approval_id,
        request_id=request_id,
        decision=decision,  # type: ignore[arg-type]
        approver=approver,
        decision_timestamp=decision_timestamp,
        reason=reason,
        request_fingerprint=fingerprint,
        expires_at=expires_at,
    )


def assert_approval_allows_execution(
    approval: ToolApprovalRecord | None,
    request: ToolExecutionRequest,
) -> ToolApprovalRecord:
    """Ensure an approval exists, matches the fingerprint, and is still valid."""
    if approval is None:
        raise ToolAuthorizationError("Explicit approval is required.")
    if approval.request_id != request.request_id:
        raise ToolAuthorizationError(
            "Approval is not bound to the requested execution request."
        )
    if approval.request_fingerprint != request.fingerprint:
        raise ToolAuthorizationError(
            "Approval fingerprint does not match the current request."
        )
    if approval.decision != "approved":
        raise ToolAuthorizationError("Rejected requests cannot execute.")
    if is_expired(approval.expires_at):
        raise ToolAuthorizationError("Approval has expired.")
    if request.request_status not in {"approved", "running"}:
        raise ToolValidationError(
            "Request is not in an executable approval state."
        )
    return approval
