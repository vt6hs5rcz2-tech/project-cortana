"""Shared validation helpers for Milestone 8 security records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from src.config import MAX_TAG_LENGTH, MAX_TAGS_PER_RECORD

EventSeverity = Literal["informational", "low", "medium", "high", "critical"]
EventStatus = Literal[
    "new",
    "investigating",
    "contained",
    "resolved",
    "false-positive",
]
IncidentSeverity = EventSeverity
IncidentStatus = Literal[
    "open",
    "triage",
    "investigating",
    "contained",
    "monitoring",
    "resolved",
    "closed",
]
IndicatorType = Literal[
    "ipv4",
    "ipv6",
    "domain",
    "url",
    "email",
    "sha256",
    "sha1",
    "md5",
    "filename",
    "process",
    "registry-key",
    "generic",
]
EvidenceType = Literal[
    "file",
    "log",
    "screenshot",
    "document",
    "network-record",
    "analyst-note",
    "external-reference",
    "generic",
]
EvidenceStorageStatus = Literal["metadata-only", "copied"]
CustodyAction = Literal[
    "registered",
    "copied",
    "verified",
    "accessed",
    "transferred",
    "exported",
    "deleted",
]
NoteType = Literal[
    "observation",
    "hypothesis",
    "action",
    "decision",
    "handoff",
    "summary",
]
TimelineEntryType = Literal["event", "note", "custody"]

EVENT_SEVERITIES: frozenset[str] = frozenset(
    {"informational", "low", "medium", "high", "critical"}
)
EVENT_STATUSES: frozenset[str] = frozenset(
    {"new", "investigating", "contained", "resolved", "false-positive"}
)
INCIDENT_STATUSES: frozenset[str] = frozenset(
    {
        "open",
        "triage",
        "investigating",
        "contained",
        "monitoring",
        "resolved",
        "closed",
    }
)
INDICATOR_TYPES: frozenset[str] = frozenset(
    {
        "ipv4",
        "ipv6",
        "domain",
        "url",
        "email",
        "sha256",
        "sha1",
        "md5",
        "filename",
        "process",
        "registry-key",
        "generic",
    }
)
EVIDENCE_TYPES: frozenset[str] = frozenset(
    {
        "file",
        "log",
        "screenshot",
        "document",
        "network-record",
        "analyst-note",
        "external-reference",
        "generic",
    }
)
EVIDENCE_STORAGE_STATUSES: frozenset[str] = frozenset({"metadata-only", "copied"})
CUSTODY_ACTIONS: frozenset[str] = frozenset(
    {
        "registered",
        "copied",
        "verified",
        "accessed",
        "transferred",
        "exported",
        "deleted",
    }
)
NOTE_TYPES: frozenset[str] = frozenset(
    {"observation", "hypothesis", "action", "decision", "handoff", "summary"}
)

CLOSED_INCIDENT_STATUSES: frozenset[str] = frozenset({"resolved", "closed"})


class SecurityValidationError(ValueError):
    """Raised when a security record field fails local validation."""


class BlankSecurityFieldError(SecurityValidationError):
    """Raised when a required security field is blank after trimming."""


class SecurityFieldTooLongError(SecurityValidationError):
    """Raised when a security field exceeds a centralized maximum length."""


class InvalidSecurityIdError(SecurityValidationError):
    """Raised when a security record ID is not a valid UUID string."""


class InvalidSecurityTimestampError(SecurityValidationError):
    """Raised when a timestamp is not valid UTC ISO 8601."""


class InvalidSecurityEnumError(SecurityValidationError):
    """Raised when a controlled vocabulary value is not allowed."""


def utc_timestamp() -> str:
    """Return the current UTC time in unambiguous ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def validate_security_id(value: str, *, field_name: str) -> str:
    """Validate and return a canonical UUID string."""
    cleaned = value.strip()
    if not cleaned:
        raise BlankSecurityFieldError(f"{field_name} cannot be blank.")
    try:
        return str(UUID(cleaned))
    except ValueError as error:
        raise InvalidSecurityIdError(f"{field_name} must be a valid UUID.") from error


def validate_optional_security_id(
    value: str | None,
    *,
    field_name: str,
) -> str | None:
    """Validate an optional UUID string, returning None when absent."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return validate_security_id(cleaned, field_name=field_name)


def validate_utc_timestamp(value: str, *, field_name: str) -> str:
    """Validate and return a UTC ISO 8601 timestamp string."""
    cleaned = value.strip()
    if not cleaned:
        raise BlankSecurityFieldError(f"{field_name} cannot be blank.")

    try:
        normalized = cleaned.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise InvalidSecurityTimestampError(
            f"{field_name} must be valid UTC ISO 8601."
        ) from error

    if parsed.tzinfo is None:
        raise InvalidSecurityTimestampError(
            f"{field_name} must include a UTC timezone."
        )

    utc_offset = parsed.utcoffset()
    if utc_offset is None or utc_offset.total_seconds() != 0:
        raise InvalidSecurityTimestampError(f"{field_name} must be in UTC.")

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


def require_non_blank_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Trim, reject blank values, and enforce a maximum length."""
    if not isinstance(value, str):
        raise SecurityValidationError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise BlankSecurityFieldError(f"{field_name} cannot be blank.")
    if len(cleaned) > max_length:
        raise SecurityFieldTooLongError(
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
        raise SecurityValidationError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise SecurityFieldTooLongError(
            f"{field_name} exceeds the maximum length of {max_length} characters."
        )
    return cleaned


def normalize_tags(tags: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Trim tags, reject blanks, preserve order, and remove duplicates."""
    if tags is None:
        return ()
    if not isinstance(tags, (list, tuple)):
        raise SecurityValidationError("Tags must be a list or tuple of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            raise SecurityValidationError("Each tag must be a string.")
        cleaned = tag.strip()
        if not cleaned:
            raise BlankSecurityFieldError("Tags cannot contain blank values.")
        if len(cleaned) > MAX_TAG_LENGTH:
            raise SecurityFieldTooLongError(
                f"Tag exceeds the maximum length of {MAX_TAG_LENGTH} characters."
            )
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)

    if len(normalized) > MAX_TAGS_PER_RECORD:
        raise SecurityFieldTooLongError(
            f"A maximum of {MAX_TAGS_PER_RECORD} tags is allowed."
        )
    return tuple(normalized)


def normalize_id_list(
    values: list[str] | tuple[str, ...] | None,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Validate UUID IDs, preserve order, and remove duplicates."""
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise SecurityValidationError(f"{field_name} must be a list or tuple.")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise SecurityValidationError(f"Each {field_name} entry must be a string.")
        item_id = validate_security_id(value, field_name=field_name)
        if item_id in seen:
            continue
        seen.add(item_id)
        normalized.append(item_id)
    return tuple(normalized)


def validate_controlled_value(
    value: str,
    *,
    field_name: str,
    allowed: frozenset[str],
) -> str:
    """Validate a controlled vocabulary value after trimming and lowercasing."""
    cleaned = value.strip().lower()
    if not cleaned:
        raise BlankSecurityFieldError(f"{field_name} cannot be blank.")
    if cleaned not in allowed:
        raise InvalidSecurityEnumError(f"Invalid {field_name}: {cleaned}.")
    return cleaned
