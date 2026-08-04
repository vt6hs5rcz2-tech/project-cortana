"""Append-only audit entries for the defensive tool framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from src.config import MAX_TOOL_AUDIT_DETAILS_LENGTH
from src.tool_common import (
    TOOL_AUDIT_ACTIONS,
    ToolAuditAction,
    ToolValidationError,
    json_safe_value,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_uuid,
    validate_tool_id,
    validate_utc_timestamp,
    validate_uuid,
)


@dataclass(frozen=True)
class ToolAuditEntry:
    """Immutable append-only tool-control audit entry."""

    audit_id: str
    timestamp: str
    action: ToolAuditAction
    actor: str
    request_id: str | None
    tool_id: str | None
    scope_id: str | None
    approval_id: str | None
    result_id: str | None
    safe_details: dict[str, Any]


def create_tool_audit_entry(
    *,
    action: str,
    actor: str = "local-user",
    request_id: str | None = None,
    tool_id: str | None = None,
    scope_id: str | None = None,
    approval_id: str | None = None,
    result_id: str | None = None,
    safe_details: dict[str, Any] | None = None,
) -> ToolAuditEntry:
    """Create a validated immutable audit entry with safe details only."""
    return validate_tool_audit_entry(
        ToolAuditEntry(
            audit_id=str(uuid4()),
            timestamp=utc_timestamp(),
            action=action,  # type: ignore[arg-type]
            actor=actor,
            request_id=request_id,
            tool_id=tool_id,
            scope_id=scope_id,
            approval_id=approval_id,
            result_id=result_id,
            safe_details=dict(safe_details or {}),
        )
    )


def validate_tool_audit_entry(entry: ToolAuditEntry) -> ToolAuditEntry:
    """Validate one tool audit entry."""
    audit_id = validate_uuid(entry.audit_id, field_name="Audit ID")
    timestamp = validate_utc_timestamp(entry.timestamp, field_name="Audit timestamp")
    action = validate_controlled_value(
        entry.action,
        field_name="audit action",
        allowed=TOOL_AUDIT_ACTIONS,
    )
    actor = require_non_blank_text(
        entry.actor,
        field_name="Actor",
        max_length=200,
    )
    request_id = validate_optional_uuid(entry.request_id, field_name="Request ID")
    scope_id = validate_optional_uuid(entry.scope_id, field_name="Scope ID")
    approval_id = validate_optional_uuid(entry.approval_id, field_name="Approval ID")
    result_id = validate_optional_uuid(entry.result_id, field_name="Result ID")
    tool_id = None
    if entry.tool_id is not None and entry.tool_id.strip():
        tool_id = validate_tool_id(entry.tool_id)

    details = json_safe_value(entry.safe_details)
    if not isinstance(details, dict):
        raise ToolValidationError("Audit safe_details must be an object.")
    _assert_details_are_safe(details)

    # Bound serialized detail size without logging sensitive content.
    serialized_length = len(str(details))
    if serialized_length > MAX_TOOL_AUDIT_DETAILS_LENGTH * 4:
        raise ToolValidationError("Audit safe_details exceed the allowed size.")

    return ToolAuditEntry(
        audit_id=audit_id,
        timestamp=timestamp,
        action=action,  # type: ignore[arg-type]
        actor=actor,
        request_id=request_id,
        tool_id=tool_id,
        scope_id=scope_id,
        approval_id=approval_id,
        result_id=result_id,
        safe_details=details,
    )


_FORBIDDEN_DETAIL_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "full_path",
        "absolute_path",
        "filename",
        "content",
        "file_content",
        "query",
        "searched_text",
        "justification",
        "notes",
        "scope_notes",
        "parameters",
        "parameter_json",
        "raw_parameters",
        "secret",
        "password",
        "token",
        "api_key",
        "environment",
        "traceback",
        "exception",
    }
)


def _assert_details_are_safe(details: dict[str, Any]) -> None:
    """Reject obviously sensitive audit detail keys."""
    for key, value in details.items():
        lowered = str(key).strip().lower()
        if lowered in _FORBIDDEN_DETAIL_KEYS:
            raise ToolValidationError(
                f"Audit detail key '{lowered}' is not allowed."
            )
        if isinstance(value, str) and len(value) > MAX_TOOL_AUDIT_DETAILS_LENGTH:
            raise ToolValidationError(
                "Audit detail value exceeds the maximum length."
            )
