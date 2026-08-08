"""Sanitized allowlisted single-incident packets for AI analysis.

This module must remain structurally isolated from evidence stores, custody
APIs, tool executors, workflow executors, and raw structured tool payloads.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config import (
    MAX_ANALYSIS_EVENTS,
    MAX_ANALYSIS_FIELD_CHARS,
    MAX_ANALYSIS_INDICATORS,
    MAX_ANALYSIS_NOTES,
    MAX_ANALYSIS_SAFE_SUMMARY_CHARS,
    MAX_ANALYSIS_SOURCE_LABEL_CHARS,
    MAX_ANALYSIS_TAG_CHARS,
    MAX_ANALYSIS_TOOL_SUMMARIES,
    MAX_ANALYSIS_WORKFLOW_SUMMARIES,
    MAX_INCIDENT_ANALYSIS_PACKET_CHARS,
    WORKFLOW_AI_CONTEXT_INJECTION_ENABLED,
)
from src.incident_analysis_common import (
    IncidentAnalysisAuthorizationError,
    IncidentAnalysisLimitError,
    IncidentAnalysisValidationError,
    bound_analysis_text,
)
from src.incident_analysis_models import IncidentAnalysisPacket
from src.incident_repository import IncidentRepository
from src.security_event import SecurityEvent
from src.security_incident import SecurityIncident
from src.security_indicator import SecurityIndicator
from src.security_note import IncidentNote
from src.workflow_repository import WorkflowRunRepository
from src.workflow_result import WorkflowRunResult, WorkflowStepResult

PACKET_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "incident",
        "events",
        "indicators",
        "notes",
        "workflow_summaries",
        "tool_summaries",
        "selected_event_count",
        "selected_indicator_count",
        "selected_note_count",
        "selected_workflow_count",
        "selected_tool_summary_count",
        "omitted_evidence",
        "omitted_custody",
        "omitted_structured_tool_data",
    }
)
PACKET_INCIDENT_KEYS: frozenset[str] = frozenset(
    {
        "incident_id",
        "title",
        "summary",
        "severity",
        "status",
        "created_at",
        "updated_at",
    }
)
PACKET_EVENT_KEYS: frozenset[str] = frozenset(
    {
        "event_id",
        "event_type",
        "title",
        "description",
        "severity",
        "status",
        "observed_at",
        "source",
    }
)
PACKET_INDICATOR_KEYS: frozenset[str] = frozenset(
    {
        "indicator_id",
        "indicator_type",
        "value",
        "confidence",
    }
)
PACKET_NOTE_KEYS: frozenset[str] = frozenset(
    {
        "note_id",
        "text",
        "note_type",
        "author",
        "tags",
        "created_at",
    }
)
PACKET_WORKFLOW_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "playbook_name",
        "playbook_version",
        "status",
        "error_code",
        "step_count",
        "created_timestamp",
        "completed_timestamp",
    }
)
PACKET_TOOL_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "step_id",
        "tool_id",
        "outcome",
        "dry_run",
        "safe_summary",
        "output_truncated",
    }
)

FORBIDDEN_PACKET_KEYS: frozenset[str] = frozenset(
    {
        "evidence",
        "evidence_ids",
        "custody",
        "custody_entries",
        "structured_data",
        "parameters",
        "path",
        "full_path",
        "filename",
        "content",
        "file_content",
        "secret",
        "password",
        "api_key",
        "approval",
        "approvals",
        "fingerprint",
        "allowed_local_path_roots",
    }
)


def build_bounded_incident_context_for_display(
    *,
    incident_repository: IncidentRepository,
    workflow_run_repository: WorkflowRunRepository,
    incident_id: str,
    selected_event_ids: tuple[str, ...] = (),
    selected_indicator_ids: tuple[str, ...] = (),
    selected_note_ids: tuple[str, ...] = (),
    selected_workflow_run_ids: tuple[str, ...] = (),
) -> IncidentAnalysisPacket:
    """Assemble the existing M12 allowlisted packet for AI-off operator review.

    Delegates to :func:`build_incident_analysis_packet`. Does not call AI,
    does not create analysis repository state, and does not widen packet
    allowlists.
    """
    return build_incident_analysis_packet(
        incident_repository=incident_repository,
        workflow_run_repository=workflow_run_repository,
        incident_id=incident_id,
        selected_event_ids=selected_event_ids,
        selected_indicator_ids=selected_indicator_ids,
        selected_note_ids=selected_note_ids,
        selected_workflow_run_ids=selected_workflow_run_ids,
    )


def build_incident_analysis_packet(
    *,
    incident_repository: IncidentRepository,
    workflow_run_repository: WorkflowRunRepository,
    incident_id: str,
    selected_event_ids: tuple[str, ...],
    selected_indicator_ids: tuple[str, ...],
    selected_note_ids: tuple[str, ...],
    selected_workflow_run_ids: tuple[str, ...],
) -> IncidentAnalysisPacket:
    """Build one sanitized allowlisted packet for explicitly selected records."""
    if len(selected_event_ids) > MAX_ANALYSIS_EVENTS:
        raise IncidentAnalysisLimitError("Selected event count exceeds the limit.")
    if len(selected_indicator_ids) > MAX_ANALYSIS_INDICATORS:
        raise IncidentAnalysisLimitError(
            "Selected indicator count exceeds the limit."
        )
    if len(selected_note_ids) > MAX_ANALYSIS_NOTES:
        raise IncidentAnalysisLimitError("Selected note count exceeds the limit.")
    if len(selected_workflow_run_ids) > MAX_ANALYSIS_WORKFLOW_SUMMARIES:
        raise IncidentAnalysisLimitError(
            "Selected workflow summary count exceeds the limit."
        )

    incident = incident_repository.get_incident(incident_id)
    if incident is None:
        raise IncidentAnalysisAuthorizationError(
            "Linked incident was not found or is no longer authorized."
        )

    events = _resolve_events(
        incident_repository,
        incident,
        selected_event_ids,
    )
    indicators = _resolve_indicators(
        incident_repository,
        incident,
        selected_indicator_ids,
    )
    notes = _resolve_notes(
        incident_repository,
        incident,
        selected_note_ids,
    )
    workflows, tool_summaries = _resolve_workflows(
        workflow_run_repository,
        incident,
        selected_workflow_run_ids,
    )

    payload: dict[str, Any] = {
        "incident": _serialize_incident(incident),
        "events": [_serialize_event(event) for event in events],
        "indicators": [
            _serialize_indicator(indicator) for indicator in indicators
        ],
        "notes": [_serialize_note(note) for note in notes],
        "workflow_summaries": [
            _serialize_workflow(run) for run in workflows
        ],
        "tool_summaries": tool_summaries,
        "selected_event_count": len(events),
        "selected_indicator_count": len(indicators),
        "selected_note_count": len(notes),
        "selected_workflow_count": len(workflows),
        "selected_tool_summary_count": len(tool_summaries),
        "omitted_evidence": True,
        "omitted_custody": True,
        "omitted_structured_tool_data": True,
    }
    _assert_packet_allowlist(payload)
    serialized = _serialize_packet_text(payload)
    if len(serialized) > MAX_INCIDENT_ANALYSIS_PACKET_CHARS:
        raise IncidentAnalysisLimitError(
            "Incident analysis packet exceeds the configured size limit."
        )
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return IncidentAnalysisPacket(
        incident_id=incident.incident_id,
        payload=payload,
        serialized_text=serialized,
        character_count=len(serialized),
        packet_fingerprint=fingerprint,
        selected_event_count=len(events),
        selected_indicator_count=len(indicators),
        selected_note_count=len(notes),
        selected_workflow_count=len(workflows),
        selected_tool_summary_count=len(tool_summaries),
    )


def _resolve_events(
    repository: IncidentRepository,
    incident: SecurityIncident,
    selected_event_ids: tuple[str, ...],
) -> list[SecurityEvent]:
    resolved: list[SecurityEvent] = []
    for event_id in selected_event_ids:
        event = repository.get_event(event_id)
        if event is None:
            raise IncidentAnalysisValidationError(
                f"Selected event was not found: {event_id}."
            )
        belongs = (
            event_id in incident.event_ids
            or event.related_incident_id == incident.incident_id
        )
        if not belongs:
            raise IncidentAnalysisAuthorizationError(
                "Selected event does not belong to the authorized incident."
            )
        resolved.append(event)
    return resolved


def _resolve_indicators(
    repository: IncidentRepository,
    incident: SecurityIncident,
    selected_indicator_ids: tuple[str, ...],
) -> list[SecurityIndicator]:
    resolved: list[SecurityIndicator] = []
    for indicator_id in selected_indicator_ids:
        indicator = repository.get_indicator(indicator_id)
        if indicator is None:
            raise IncidentAnalysisValidationError(
                f"Selected indicator was not found: {indicator_id}."
            )
        belongs = (
            indicator_id in incident.indicator_ids
            or incident.incident_id in indicator.related_incident_ids
        )
        if not belongs:
            raise IncidentAnalysisAuthorizationError(
                "Selected indicator does not belong to the authorized incident."
            )
        resolved.append(indicator)
    return resolved


def _resolve_notes(
    repository: IncidentRepository,
    incident: SecurityIncident,
    selected_note_ids: tuple[str, ...],
) -> list[IncidentNote]:
    notes_by_id = {
        note.note_id: note
        for note in repository.list_notes(incident.incident_id)
    }
    resolved: list[IncidentNote] = []
    for note_id in selected_note_ids:
        note = notes_by_id.get(note_id)
        if note is None:
            # Distinguish missing vs cross-incident.
            raise IncidentAnalysisAuthorizationError(
                "Selected note does not belong to the authorized incident."
            )
        resolved.append(note)
    return resolved


def _resolve_workflows(
    workflow_run_repository: WorkflowRunRepository,
    incident: SecurityIncident,
    selected_workflow_run_ids: tuple[str, ...],
) -> tuple[list[WorkflowRunResult], list[dict[str, Any]]]:
    if not selected_workflow_run_ids:
        return [], []
    if not WORKFLOW_AI_CONTEXT_INJECTION_ENABLED:
        raise IncidentAnalysisValidationError(
            "Workflow and tool summaries are disabled. "
            "Set WORKFLOW_AI_CONTEXT_INJECTION_ENABLED to include them."
        )

    resolved: list[WorkflowRunResult] = []
    tool_summaries: list[dict[str, Any]] = []
    for run_id in selected_workflow_run_ids:
        run = workflow_run_repository.get_run(run_id)
        if run is None:
            raise IncidentAnalysisValidationError(
                f"Selected workflow run was not found: {run_id}."
            )
        if run.incident_id != incident.incident_id:
            raise IncidentAnalysisAuthorizationError(
                "Selected workflow run is not linked to the authorized incident."
            )
        resolved.append(run)
        for step in run.step_results:
            summary = _serialize_tool_summary(step)
            if summary is not None:
                tool_summaries.append(summary)
                if len(tool_summaries) > MAX_ANALYSIS_TOOL_SUMMARIES:
                    raise IncidentAnalysisLimitError(
                        "Selected tool summary count exceeds the limit."
                    )
    return resolved, tool_summaries


def _serialize_incident(incident: SecurityIncident) -> dict[str, Any]:
    return {
        "incident_id": incident.incident_id,
        "title": bound_analysis_text(
            incident.title,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "summary": bound_analysis_text(
            incident.summary,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "severity": incident.severity,
        "status": incident.status,
        "created_at": incident.created_at,
        "updated_at": incident.updated_at,
    }


def _serialize_event(event: SecurityEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "title": bound_analysis_text(
            event.title,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "description": bound_analysis_text(
            event.description,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "severity": event.severity,
        "status": event.status,
        "observed_at": event.observed_at,
        "source": bound_analysis_text(
            event.source,
            max_length=MAX_ANALYSIS_SOURCE_LABEL_CHARS,
        ),
    }


def _serialize_indicator(indicator: SecurityIndicator) -> dict[str, Any]:
    # Indicator free-text notes are excluded by default.
    return {
        "indicator_id": indicator.indicator_id,
        "indicator_type": indicator.indicator_type,
        "value": bound_analysis_text(
            indicator.normalized_value,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "confidence": indicator.confidence,
    }


def _serialize_note(note: IncidentNote) -> dict[str, Any]:
    tags = [
        bound_analysis_text(tag, max_length=MAX_ANALYSIS_TAG_CHARS)
        for tag in note.tags
    ]
    return {
        "note_id": note.note_id,
        "text": bound_analysis_text(
            note.text,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "note_type": note.note_type,
        "author": bound_analysis_text(
            note.author,
            max_length=MAX_ANALYSIS_FIELD_CHARS,
        ),
        "tags": tags,
        "created_at": note.created_at,
    }


def _serialize_workflow(run: WorkflowRunResult) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "playbook_name": run.playbook_name,
        "playbook_version": run.playbook_version,
        "status": run.status,
        "error_code": run.error_code,
        "step_count": len(run.step_results),
        "created_timestamp": run.created_timestamp,
        "completed_timestamp": run.completed_timestamp,
    }


def _serialize_tool_summary(step: WorkflowStepResult) -> dict[str, Any] | None:
    tool_result = step.tool_result
    if tool_result is None:
        return None
    return {
        "step_id": step.step_id,
        "tool_id": step.tool_id,
        "outcome": tool_result.outcome,
        "dry_run": bool(tool_result.dry_run),
        "safe_summary": bound_analysis_text(
            tool_result.safe_summary,
            max_length=MAX_ANALYSIS_SAFE_SUMMARY_CHARS,
        ),
        "output_truncated": bool(tool_result.output_truncated),
    }


def _assert_packet_allowlist(payload: dict[str, Any]) -> None:
    if frozenset(payload.keys()) != PACKET_ROOT_KEYS:
        raise IncidentAnalysisValidationError(
            "Incident analysis packet root keys failed allowlist validation."
        )
    if frozenset(payload["incident"].keys()) != PACKET_INCIDENT_KEYS:
        raise IncidentAnalysisValidationError(
            "Incident analysis packet incident keys failed allowlist validation."
        )
    for event in payload["events"]:
        if frozenset(event.keys()) != PACKET_EVENT_KEYS:
            raise IncidentAnalysisValidationError(
                "Incident analysis packet event keys failed allowlist validation."
            )
    for indicator in payload["indicators"]:
        if frozenset(indicator.keys()) != PACKET_INDICATOR_KEYS:
            raise IncidentAnalysisValidationError(
                "Incident analysis packet indicator keys failed allowlist validation."
            )
        if "notes" in indicator:
            raise IncidentAnalysisValidationError(
                "Incident analysis packet included forbidden indicator notes."
            )
    for note in payload["notes"]:
        if frozenset(note.keys()) != PACKET_NOTE_KEYS:
            raise IncidentAnalysisValidationError(
                "Incident analysis packet note keys failed allowlist validation."
            )
    for workflow in payload["workflow_summaries"]:
        if frozenset(workflow.keys()) != PACKET_WORKFLOW_KEYS:
            raise IncidentAnalysisValidationError(
                "Incident analysis packet workflow keys failed allowlist validation."
            )
    for tool_summary in payload["tool_summaries"]:
        if frozenset(tool_summary.keys()) != PACKET_TOOL_SUMMARY_KEYS:
            raise IncidentAnalysisValidationError(
                "Incident analysis packet tool summary keys failed allowlist validation."
            )
    _assert_no_forbidden_structured_keys(payload)


def _assert_no_forbidden_structured_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_PACKET_KEYS:
                raise IncidentAnalysisValidationError(
                    "Incident analysis packet contained a forbidden field."
                )
            _assert_no_forbidden_structured_keys(nested)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_structured_keys(item)


def _serialize_packet_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
