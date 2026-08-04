"""Immutable tool execution result model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from src.config import MAX_TOOL_RESULT_SUMMARY_LENGTH
from src.tool_common import (
    TOOL_EXECUTION_OUTCOMES,
    ToolExecutionOutcome,
    ToolValidationError,
    json_safe_value,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_text,
    validate_optional_uuid,
    validate_tool_id,
    validate_utc_timestamp,
    validate_uuid,
)


@dataclass(frozen=True)
class ToolExecutionResult:
    """Immutable defensive tool execution or planning result."""

    result_id: str
    request_id: str
    tool_id: str
    started_timestamp: str
    completed_timestamp: str
    outcome: ToolExecutionOutcome
    safe_summary: str
    structured_data: dict[str, Any]
    output_truncated: bool
    error_class: str | None
    dry_run: bool
    incident_id: str | None


def create_tool_execution_result(
    *,
    request_id: str,
    tool_id: str,
    started_timestamp: str | None = None,
    completed_timestamp: str | None = None,
    outcome: str,
    safe_summary: str,
    structured_data: dict[str, Any] | None = None,
    output_truncated: bool = False,
    error_class: str | None = None,
    dry_run: bool = False,
    incident_id: str | None = None,
) -> ToolExecutionResult:
    """Create a validated immutable tool execution result."""
    started = started_timestamp or utc_timestamp()
    completed = completed_timestamp or utc_timestamp()
    return validate_tool_execution_result(
        ToolExecutionResult(
            result_id=str(uuid4()),
            request_id=request_id,
            tool_id=tool_id,
            started_timestamp=started,
            completed_timestamp=completed,
            outcome=outcome,  # type: ignore[arg-type]
            safe_summary=safe_summary,
            structured_data=dict(structured_data or {}),
            output_truncated=bool(output_truncated),
            error_class=error_class,
            dry_run=bool(dry_run),
            incident_id=incident_id,
        )
    )


def validate_tool_execution_result(
    result: ToolExecutionResult,
) -> ToolExecutionResult:
    """Validate one tool execution result."""
    result_id = validate_uuid(result.result_id, field_name="Result ID")
    request_id = validate_uuid(result.request_id, field_name="Request ID")
    tool_id = validate_tool_id(result.tool_id)
    started = validate_utc_timestamp(
        result.started_timestamp,
        field_name="Started timestamp",
    )
    completed = validate_utc_timestamp(
        result.completed_timestamp,
        field_name="Completed timestamp",
    )
    outcome = validate_controlled_value(
        result.outcome,
        field_name="execution outcome",
        allowed=TOOL_EXECUTION_OUTCOMES,
    )
    summary = require_non_blank_text(
        result.safe_summary,
        field_name="Safe summary",
        max_length=MAX_TOOL_RESULT_SUMMARY_LENGTH,
    )
    error_class = validate_optional_text(
        result.error_class,
        field_name="Error class",
        max_length=200,
    )
    incident_id = validate_optional_uuid(
        result.incident_id,
        field_name="Incident ID",
    )
    structured = json_safe_value(result.structured_data)
    if not isinstance(structured, dict):
        raise ToolValidationError("Structured data must be an object.")

    if result.dry_run and outcome == "succeeded":
        raise ToolValidationError(
            "Dry-run results cannot claim execution success."
        )
    if result.dry_run and outcome not in {"planned", "denied", "expired", "cancelled"}:
        raise ToolValidationError(
            "Dry-run results must use a planning or denial outcome."
        )

    return ToolExecutionResult(
        result_id=result_id,
        request_id=request_id,
        tool_id=tool_id,
        started_timestamp=started,
        completed_timestamp=completed,
        outcome=outcome,  # type: ignore[arg-type]
        safe_summary=summary,
        structured_data=structured,
        output_truncated=bool(result.output_truncated),
        error_class=error_class,
        dry_run=bool(result.dry_run),
        incident_id=incident_id,
    )
