"""In-memory safe audit trail for Milestone 12 incident analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from src.config import MAX_INCIDENT_ANALYSIS_AUDIT_ENTRIES
from src.incident_analysis_common import (
    AnalysisKind,
    IncidentAnalysisValidationError,
)
from src.tool_common import utc_timestamp, validate_uuid

AnalysisAuditAction = Literal[
    "analysis_prepared",
    "analysis_confirmed",
    "analysis_completed",
    "analysis_failed",
    "analysis_note_saved",
    "analysis_note_save_denied",
]

FORBIDDEN_AUDIT_DETAIL_KEYS: frozenset[str] = frozenset(
    {
        "packet",
        "prompt",
        "full_response",
        "output_text",
        "advisory_output",
        "note_text",
        "event_description",
        "indicator_value",
        "note_content",
        "workflow_summary",
        "tool_summary",
        "exception",
        "exception_text",
        "secret",
        "credential",
    }
)


class IncidentAnalysisAuditSafetyError(IncidentAnalysisValidationError):
    """Raised when audit content includes forbidden fields or nested keys."""


@dataclass(frozen=True)
class IncidentAnalysisAuditEntry:
    """Safe metadata-only audit entry for one analysis action."""

    analysis_id: str
    incident_id: str
    scope_id: str
    analysis_kind: AnalysisKind
    action: AnalysisAuditAction
    selected_event_count: int
    selected_indicator_count: int
    selected_note_count: int
    selected_workflow_count: int
    packet_character_count: int
    output_character_count: int
    status: str
    error_code: str | None
    saved_note_id: str | None
    created_at: str


ALLOWED_AUDIT_FIELD_KEYS: frozenset[str] = frozenset(
    field.name for field in fields(IncidentAnalysisAuditEntry)
)


def assert_incident_analysis_audit_payload_safe(payload: object) -> None:
    """Reject forbidden audit keys anywhere in a nested audit payload."""
    _reject_forbidden_audit_keys(payload)


def _reject_forbidden_audit_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_AUDIT_DETAIL_KEYS:
                raise IncidentAnalysisAuditSafetyError(
                    "Incident analysis audit entry contained a forbidden field."
                )
            _reject_forbidden_audit_keys(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_audit_keys(item)


class InMemoryIncidentAnalysisAuditLog:
    """Bounded in-memory audit log with metadata-only entries."""

    def __init__(
        self,
        *,
        max_entries: int = MAX_INCIDENT_ANALYSIS_AUDIT_ENTRIES,
    ) -> None:
        if max_entries < 1:
            raise IncidentAnalysisValidationError(
                "max_entries must be at least 1."
            )
        self._max_entries = max_entries
        self._entries: list[IncidentAnalysisAuditEntry] = []

    def append(
        self,
        *,
        analysis_id: str,
        incident_id: str,
        scope_id: str,
        analysis_kind: AnalysisKind,
        action: AnalysisAuditAction,
        selected_event_count: int = 0,
        selected_indicator_count: int = 0,
        selected_note_count: int = 0,
        selected_workflow_count: int = 0,
        packet_character_count: int = 0,
        output_character_count: int = 0,
        status: str,
        error_code: str | None = None,
        saved_note_id: str | None = None,
    ) -> IncidentAnalysisAuditEntry:
        """Append one validated safe audit entry."""
        entry = IncidentAnalysisAuditEntry(
            analysis_id=validate_uuid(analysis_id, field_name="Analysis ID"),
            incident_id=validate_uuid(incident_id, field_name="Incident ID"),
            scope_id=validate_uuid(scope_id, field_name="Scope ID"),
            analysis_kind=analysis_kind,
            action=action,
            selected_event_count=int(selected_event_count),
            selected_indicator_count=int(selected_indicator_count),
            selected_note_count=int(selected_note_count),
            selected_workflow_count=int(selected_workflow_count),
            packet_character_count=int(packet_character_count),
            output_character_count=int(output_character_count),
            status=status,
            error_code=error_code,
            saved_note_id=(
                None
                if saved_note_id is None
                else validate_uuid(saved_note_id, field_name="Note ID")
            ),
            created_at=utc_timestamp(),
        )
        self._assert_safe(entry)
        self._entries.append(entry)
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)
        return entry

    def list_entries(self) -> list[IncidentAnalysisAuditEntry]:
        """Return audit entries in append order."""
        return list(self._entries)

    def _assert_safe(self, entry: IncidentAnalysisAuditEntry) -> None:
        """Reject forbidden fields or nested keys in one typed audit entry."""
        payload: dict[str, Any] = asdict(entry)
        if frozenset(payload.keys()) != ALLOWED_AUDIT_FIELD_KEYS:
            raise IncidentAnalysisAuditSafetyError(
                "Incident analysis audit entry keys failed allowlist validation."
            )
        assert_incident_analysis_audit_payload_safe(payload)
