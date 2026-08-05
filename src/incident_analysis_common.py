"""Shared types and helpers for Milestone 12 incident AI analysis."""

from __future__ import annotations

import re
import secrets
from typing import Literal

from src.config import ANALYSIS_TRUNCATION_MARKER

_BOUNDARY_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_BOUNDARY_TOKEN_BYTES = 16

AnalysisKind = Literal[
    "summary",
    "gaps",
    "investigation_questions",
    "report_draft",
    "remediation_checklist",
    "outcome_explanation",
]

ANALYSIS_KINDS: frozenset[str] = frozenset(
    {
        "summary",
        "gaps",
        "investigation_questions",
        "report_draft",
        "remediation_checklist",
        "outcome_explanation",
    }
)

AnalysisStatus = Literal["prepared", "completed", "failed"]

ANALYSIS_NONE_SELECTION = "-"


class IncidentAnalysisError(ValueError):
    """Base error for controlled incident analysis operations."""


class IncidentAnalysisDisabledError(IncidentAnalysisError):
    """Raised when incident AI analysis is disabled by configuration."""


class IncidentAnalysisNoteSaveDisabledError(IncidentAnalysisError):
    """Raised when analysis note saving is disabled by configuration."""


class IncidentAnalysisAuthorizationError(IncidentAnalysisError):
    """Raised when incident/scope authorization checks fail."""


class IncidentAnalysisValidationError(IncidentAnalysisError):
    """Raised when analysis inputs or packets fail validation."""


class IncidentAnalysisLimitError(IncidentAnalysisError):
    """Raised when an analysis retention or size limit would be exceeded."""


class IncidentAnalysisConflictError(IncidentAnalysisError):
    """Raised when an immutable analysis identity conflict occurs."""


class InvalidIncidentAnalysisBoundaryTokenError(IncidentAnalysisError):
    """Raised when an analysis boundary token is unsafe or malformed."""


def generate_incident_analysis_boundary_token() -> str:
    """Generate an unpredictable boundary token for untrusted incident packets."""
    return secrets.token_hex(_BOUNDARY_TOKEN_BYTES)


def validate_incident_analysis_boundary_token(token: str) -> str:
    """Validate and return a boundary token safe for text markers."""
    if not _BOUNDARY_TOKEN_PATTERN.fullmatch(token):
        raise InvalidIncidentAnalysisBoundaryTokenError(
            "Incident analysis boundary token must be 16-64 characters of "
            "letters, digits, underscore, or hyphen."
        )
    return token


def bound_analysis_text(value: str, *, max_length: int) -> str:
    """Return whitespace-normalized text bounded with an explicit truncation marker."""
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_length:
        return cleaned
    keep = max_length - len(ANALYSIS_TRUNCATION_MARKER)
    if keep < 1:
        return ANALYSIS_TRUNCATION_MARKER.strip()
    return f"{cleaned[:keep]}{ANALYSIS_TRUNCATION_MARKER}"


def validate_analysis_kind(value: str) -> AnalysisKind:
    """Validate and return one supported analysis kind."""
    cleaned = value.strip().lower()
    if cleaned not in ANALYSIS_KINDS:
        raise IncidentAnalysisValidationError(
            f"Unsupported analysis kind: {value.strip()!r}."
        )
    return cleaned  # type: ignore[return-value]
