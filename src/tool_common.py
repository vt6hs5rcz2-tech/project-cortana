"""Shared validation helpers for the Milestone 9 defensive tool framework."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID

from src.config import (
    FINGERPRINT_DISPLAY_PREFIX_CHARS,
    MAX_TOOL_JUSTIFICATION_LENGTH,
    MAX_TOOL_NAME_LENGTH,
    MAX_TOOL_PARAMETER_STRING_LENGTH,
)

ToolCategory = Literal[
    "diagnostics",
    "file-integrity",
    "log-analysis",
    "evidence-support",
    "system-information",
    "network-information",
    "incident-support",
    "simulation",
]
ToolRiskLevel = Literal[
    "informational",
    "low",
    "moderate",
    "high",
    "prohibited",
]
ToolExecutionMode = Literal[
    "internal-python",
    "simulated",
    "future-external",
]
ToolObjectiveType = Literal[
    "inspect",
    "hash",
    "compare",
    "summarize",
    "search",
    "simulate",
]
ToolTargetType = Literal[
    "none",
    "local-file",
    "incident",
    "mock-log",
    "system-summary",
]
ToolParameterType = Literal[
    "string",
    "integer",
    "boolean",
    "choice",
    "file-path",
    "sha256",
    "record-id",
]
ToolRequestStatus = Literal[
    "drafted",
    "awaiting-approval",
    "approved",
    "rejected",
    "running",
    "succeeded",
    "failed",
    "expired",
    "cancelled",
]
ToolApprovalDecision = Literal["approved", "rejected"]
ToolExecutionOutcome = Literal[
    "planned",
    "succeeded",
    "failed",
    "denied",
    "expired",
    "cancelled",
    "timed_out_terminated",
    "termination_unconfirmed",
    "resource_limit_exceeded",
]
ToolProcessIsolation = Literal[
    "prohibited",
    "eligible",
    "required",
]
ToolCapabilityClass = Literal[
    "internal-readonly",
    "internal-mutating",
    "arbitrary-shell",
    "external-process",
    "autonomous-remediation",
    "ai-context-injection",
]
ToolAuditAction = Literal[
    "scope-created",
    "request-created",
    "dry-run-generated",
    "approval-recorded",
    "execution-started",
    "execution-succeeded",
    "execution-failed",
    "execution-denied",
    "request-cancelled",
    "process_execution_started",
    "process_execution_completed",
    "process_launch_failed",
    "process_timeout_observed",
    "process_timeout_terminated",
    "process_cancellation_requested",
    "process_cancellation_terminated",
    "process_termination_unconfirmed",
    "process_result_rejected",
    "process_job_object_created",
    "process_job_object_configured",
    "process_job_object_assigned",
    "process_job_object_setup_failed",
    "process_resource_limit_exceeded",
    "process_tree_termination_requested",
    "process_tree_terminated",
    "process_tree_termination_unconfirmed",
]

TOOL_CATEGORIES: frozenset[str] = frozenset(
    {
        "diagnostics",
        "file-integrity",
        "log-analysis",
        "evidence-support",
        "system-information",
        "network-information",
        "incident-support",
        "simulation",
    }
)
TOOL_RISK_LEVELS: frozenset[str] = frozenset(
    {"informational", "low", "moderate", "high", "prohibited"}
)
TOOL_EXECUTION_MODES: frozenset[str] = frozenset(
    {"internal-python", "simulated", "future-external"}
)
TOOL_OBJECTIVE_TYPES: frozenset[str] = frozenset(
    {"inspect", "hash", "compare", "summarize", "search", "simulate"}
)
TOOL_TARGET_TYPES: frozenset[str] = frozenset(
    {"none", "local-file", "incident", "mock-log", "system-summary"}
)
TOOL_PARAMETER_TYPES: frozenset[str] = frozenset(
    {
        "string",
        "integer",
        "boolean",
        "choice",
        "file-path",
        "sha256",
        "record-id",
    }
)
TOOL_REQUEST_STATUSES: frozenset[str] = frozenset(
    {
        "drafted",
        "awaiting-approval",
        "approved",
        "rejected",
        "running",
        "succeeded",
        "failed",
        "expired",
        "cancelled",
    }
)
TOOL_APPROVAL_DECISIONS: frozenset[str] = frozenset({"approved", "rejected"})
TOOL_EXECUTION_OUTCOMES: frozenset[str] = frozenset(
    {
        "planned",
        "succeeded",
        "failed",
        "denied",
        "expired",
        "cancelled",
        "timed_out_terminated",
        "termination_unconfirmed",
        "resource_limit_exceeded",
    }
)
TOOL_PROCESS_ISOLATION_VALUES: frozenset[str] = frozenset(
    {"prohibited", "eligible", "required"}
)
TOOL_CAPABILITY_CLASSES: frozenset[str] = frozenset(
    {
        "internal-readonly",
        "internal-mutating",
        "arbitrary-shell",
        "external-process",
        "autonomous-remediation",
        "ai-context-injection",
    }
)
# Replay-safe tools may re-run from step 0 without a new operation instance.
READ_ONLY_TOOL_CAPABILITY_CLASSES: frozenset[str] = frozenset({"internal-readonly"})
# Side-effecting classes require a persisted claim plus a new operation
# instance before the same step can execute again.
SIDE_EFFECTING_TOOL_CAPABILITY_CLASSES: frozenset[str] = (
    TOOL_CAPABILITY_CLASSES - READ_ONLY_TOOL_CAPABILITY_CLASSES
)
# Capability classes that require a live kill-switch at execution time.
GATED_TOOL_CAPABILITY_CLASSES: frozenset[str] = frozenset(
    {
        "arbitrary-shell",
        "external-process",
        "autonomous-remediation",
    }
)
# Declared capabilities with no production implementation.
RESERVED_TOOL_CAPABILITY_CLASSES: frozenset[str] = frozenset(
    {"ai-context-injection"}
)
TOOL_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "scope-created",
        "request-created",
        "dry-run-generated",
        "approval-recorded",
        "execution-started",
        "execution-succeeded",
        "execution-failed",
        "execution-denied",
        "request-cancelled",
        "process_execution_started",
        "process_execution_completed",
        "process_launch_failed",
        "process_timeout_observed",
        "process_timeout_terminated",
        "process_cancellation_requested",
        "process_cancellation_terminated",
        "process_termination_unconfirmed",
        "process_result_rejected",
        "process_job_object_created",
        "process_job_object_configured",
        "process_job_object_assigned",
        "process_job_object_setup_failed",
        "process_resource_limit_exceeded",
        "process_tree_termination_requested",
        "process_tree_terminated",
        "process_tree_termination_unconfirmed",
    }
)

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
IMPLEMENTATION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,62}$")


class ToolValidationError(ValueError):
    """Raised when a tool-framework field fails local validation."""


class BlankToolFieldError(ToolValidationError):
    """Raised when a required tool field is blank after trimming."""


class ToolFieldTooLongError(ToolValidationError):
    """Raised when a tool field exceeds a centralized maximum length."""


class InvalidToolIdError(ToolValidationError):
    """Raised when a tool-related ID is invalid."""


class UnknownToolIdError(ToolValidationError):
    """Raised when a syntactically valid tool ID is not registered."""


class InvalidToolEnumError(ToolValidationError):
    """Raised when a controlled vocabulary value is not allowed."""


class InvalidToolTimestampError(ToolValidationError):
    """Raised when a timestamp is not valid UTC ISO 8601."""


class ToolParameterError(ToolValidationError):
    """Raised when tool parameters fail schema validation."""


class ToolAuthorizationError(ToolValidationError):
    """Raised when a scope or approval check denies an action."""


class ToolPolicyError(ToolValidationError):
    """Raised when a tool policy rule denies an action."""


def utc_timestamp() -> str:
    """Return the current UTC time in unambiguous ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_utc_timestamp(value: str) -> datetime:
    """Parse a validated UTC ISO 8601 timestamp into an aware datetime."""
    cleaned = validate_utc_timestamp(value, field_name="Timestamp")
    return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))


def validate_utc_timestamp(value: str, *, field_name: str) -> str:
    """Validate and return a UTC ISO 8601 timestamp string."""
    cleaned = value.strip()
    if not cleaned:
        raise BlankToolFieldError(f"{field_name} cannot be blank.")

    try:
        normalized = cleaned.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise InvalidToolTimestampError(
            f"{field_name} must be valid UTC ISO 8601."
        ) from error

    if parsed.tzinfo is None:
        raise InvalidToolTimestampError(
            f"{field_name} must include a UTC timezone."
        )

    utc_offset = parsed.utcoffset()
    if utc_offset is None or utc_offset.total_seconds() != 0:
        raise InvalidToolTimestampError(f"{field_name} must be in UTC.")

    return cleaned


def validate_optional_utc_timestamp(
    value: str | None,
    *,
    field_name: str,
) -> str | None:
    """Validate an optional UTC timestamp, returning None when absent."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return validate_utc_timestamp(cleaned, field_name=field_name)


def validate_uuid(value: str, *, field_name: str) -> str:
    """Validate and return a canonical UUID string."""
    cleaned = value.strip()
    if not cleaned:
        raise BlankToolFieldError(f"{field_name} cannot be blank.")
    try:
        return str(UUID(cleaned))
    except ValueError as error:
        raise InvalidToolIdError(f"{field_name} must be a valid UUID.") from error


def validate_optional_uuid(value: str | None, *, field_name: str) -> str | None:
    """Validate an optional UUID string, returning None when absent."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return validate_uuid(cleaned, field_name=field_name)


def require_non_blank_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Trim, reject blank values, and enforce a maximum length."""
    if not isinstance(value, str):
        raise ToolValidationError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise BlankToolFieldError(f"{field_name} cannot be blank.")
    if len(cleaned) > max_length:
        raise ToolFieldTooLongError(
            f"{field_name} exceeds the maximum length of {max_length} characters."
        )
    return cleaned


def validate_optional_text(
    value: str | None,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    """Trim optional text; blank optional text becomes None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolValidationError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise ToolFieldTooLongError(
            f"{field_name} exceeds the maximum length of {max_length} characters."
        )
    return cleaned


def validate_controlled_value(
    value: str,
    *,
    field_name: str,
    allowed: frozenset[str],
) -> str:
    """Validate a controlled vocabulary value after trimming and lowercasing."""
    cleaned = value.strip().lower()
    if not cleaned:
        raise BlankToolFieldError(f"{field_name} cannot be blank.")
    if cleaned not in allowed:
        raise InvalidToolEnumError(f"Invalid {field_name}: {cleaned}.")
    return cleaned


def validate_tool_id(value: str) -> str:
    """Validate a stable deterministic tool ID."""
    cleaned = require_non_blank_text(
        value,
        field_name="Tool ID",
        max_length=MAX_TOOL_NAME_LENGTH,
    ).lower()
    if not TOOL_ID_PATTERN.fullmatch(cleaned):
        raise InvalidToolIdError(
            "Tool ID must be a lowercase hyphenated identifier."
        )
    return cleaned


def validate_implementation_id(value: str) -> str:
    """Validate a trusted internal implementation identifier."""
    cleaned = require_non_blank_text(
        value,
        field_name="Implementation ID",
        max_length=MAX_TOOL_NAME_LENGTH,
    ).lower()
    if not IMPLEMENTATION_ID_PATTERN.fullmatch(cleaned):
        raise InvalidToolIdError(
            "Implementation ID must be a lowercase underscore identifier."
        )
    return cleaned


def validate_sha256_digest(value: str, *, field_name: str = "SHA-256") -> str:
    """Validate and return a lowercase SHA-256 hex digest."""
    cleaned = require_non_blank_text(
        value,
        field_name=field_name,
        max_length=64,
    )
    if not SHA256_PATTERN.fullmatch(cleaned):
        raise ToolParameterError(f"{field_name} must be a 64-character hex digest.")
    return cleaned.lower()


def validate_justification(value: str) -> str:
    """Validate a required tool-request justification."""
    return require_non_blank_text(
        value,
        field_name="Justification",
        max_length=MAX_TOOL_JUSTIFICATION_LENGTH,
    )


def normalize_string_tuple(
    values: list[str] | tuple[str, ...] | None,
    *,
    field_name: str,
    allowed: frozenset[str] | None = None,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    """Normalize a list of controlled or free-form strings."""
    if values is None:
        values = ()
    if not isinstance(values, (list, tuple)):
        raise ToolValidationError(f"{field_name} must be a list or tuple.")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ToolValidationError(f"Each {field_name} entry must be a string.")
        if allowed is not None:
            item = validate_controlled_value(
                value,
                field_name=field_name,
                allowed=allowed,
            )
        else:
            item = require_non_blank_text(
                value,
                field_name=field_name,
                max_length=MAX_TOOL_PARAMETER_STRING_LENGTH,
            )
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)

    if not allow_empty and not normalized:
        raise ToolValidationError(f"{field_name} cannot be empty.")
    return tuple(normalized)


def canonical_json(value: Any) -> str:
    """Serialize a JSON-safe value with deterministic key ordering."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    """Convert tuples to lists for deterministic JSON serialization."""
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compute_request_fingerprint(
    *,
    request_id: str,
    tool_id: str,
    normalized_parameters: dict[str, Any],
    scope_id: str,
    incident_id: str | None,
    dry_run_requested: bool,
    justification: str,
) -> str:
    """Return a deterministic SHA-256 fingerprint for an immutable request."""
    payload = {
        "request_id": request_id,
        "tool_id": tool_id,
        "normalized_parameters": normalized_parameters,
        "scope_id": scope_id,
        "incident_id": incident_id,
        "dry_run_requested": dry_run_requested,
        "justification": justification,
    }
    digest = sha256(canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def fingerprint_display_prefix(fingerprint: str) -> str:
    """Return a short fingerprint prefix safe for user display."""
    return fingerprint[:FINGERPRINT_DISPLAY_PREFIX_CHARS]


def is_expired(expires_at: str | None, *, now: datetime | None = None) -> bool:
    """Return True when an optional expiry timestamp is in the past."""
    if expires_at is None:
        return False
    current = now or datetime.now(timezone.utc)
    return parse_utc_timestamp(expires_at) <= current


def json_safe_value(value: Any) -> Any:
    """Return a JSON-safe representation of structured tool data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): json_safe_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise ToolValidationError(
        f"Structured data contains unsupported type: {type(value).__name__}."
    )
