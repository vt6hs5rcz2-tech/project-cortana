"""Immutable security incident model for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.config import MAX_INCIDENT_SUMMARY_LENGTH, MAX_INCIDENT_TITLE_LENGTH
from src.security_common import (
    CLOSED_INCIDENT_STATUSES,
    EVENT_SEVERITIES,
    INCIDENT_STATUSES,
    IncidentSeverity,
    IncidentStatus,
    SecurityValidationError,
    normalize_id_list,
    normalize_tags,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_utc_timestamp,
    validate_security_id,
    validate_utc_timestamp,
)


@dataclass(frozen=True)
class SecurityIncident:
    """Immutable cybersecurity incident recorded by an explicit user command."""

    incident_id: str
    title: str
    summary: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: str
    updated_at: str
    opened_at: str
    closed_at: str | None
    event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    indicator_ids: tuple[str, ...]
    note_ids: tuple[str, ...]
    tags: tuple[str, ...]


def create_security_incident(
    *,
    title: str,
    summary: str,
    severity: str,
    status: str = "open",
    opened_at: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    event_ids: list[str] | tuple[str, ...] | None = None,
    evidence_ids: list[str] | tuple[str, ...] | None = None,
    indicator_ids: list[str] | tuple[str, ...] | None = None,
    note_ids: list[str] | tuple[str, ...] | None = None,
) -> SecurityIncident:
    """Create a validated immutable security incident."""
    now = utc_timestamp()
    return validate_security_incident(
        _build_incident_for_validation(
            incident_id=str(uuid4()),
            title=title,
            summary=summary,
            severity=severity,
            status=status,
            created_at=now,
            updated_at=now,
            opened_at=opened_at or now,
            closed_at=None,
            event_ids=tuple(event_ids or ()),
            evidence_ids=tuple(evidence_ids or ()),
            indicator_ids=tuple(indicator_ids or ()),
            note_ids=tuple(note_ids or ()),
            tags=tuple(tags or ()),
        )
    )


def update_security_incident_status(
    incident: SecurityIncident,
    status: str,
) -> SecurityIncident:
    """Return a validated replacement incident with updated status/closed rules."""
    normalized_status = validate_controlled_value(
        status,
        field_name="incident status",
        allowed=INCIDENT_STATUSES,
    )
    closed_at = incident.closed_at
    if normalized_status in CLOSED_INCIDENT_STATUSES:
        if closed_at is None:
            closed_at = utc_timestamp()
    else:
        closed_at = None

    return validate_security_incident(
        _build_incident_for_validation(
            incident_id=incident.incident_id,
            title=incident.title,
            summary=incident.summary,
            severity=incident.severity,
            status=normalized_status,
            created_at=incident.created_at,
            updated_at=utc_timestamp(),
            opened_at=incident.opened_at,
            closed_at=closed_at,
            event_ids=incident.event_ids,
            evidence_ids=incident.evidence_ids,
            indicator_ids=incident.indicator_ids,
            note_ids=incident.note_ids,
            tags=incident.tags,
        )
    )


def replace_security_incident(
    incident: SecurityIncident,
    *,
    title: str | None = None,
    summary: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    opened_at: str | None = None,
    closed_at: str | None = None,
    clear_closed_at: bool = False,
    event_ids: list[str] | tuple[str, ...] | None = None,
    evidence_ids: list[str] | tuple[str, ...] | None = None,
    indicator_ids: list[str] | tuple[str, ...] | None = None,
    note_ids: list[str] | tuple[str, ...] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> SecurityIncident:
    """Return a validated replacement incident, preserving identity and created_at."""
    next_status = incident.status if status is None else status
    next_closed = incident.closed_at
    if clear_closed_at:
        next_closed = None
    elif closed_at is not None:
        next_closed = closed_at

    if status is not None:
        normalized = validate_controlled_value(
            status,
            field_name="incident status",
            allowed=INCIDENT_STATUSES,
        )
        if normalized in CLOSED_INCIDENT_STATUSES:
            if next_closed is None:
                next_closed = utc_timestamp()
        else:
            next_closed = None
        next_status = normalized

    return validate_security_incident(
        _build_incident_for_validation(
            incident_id=incident.incident_id,
            title=incident.title if title is None else title,
            summary=incident.summary if summary is None else summary,
            severity=incident.severity if severity is None else severity,
            status=next_status,
            created_at=incident.created_at,
            updated_at=utc_timestamp(),
            opened_at=incident.opened_at if opened_at is None else opened_at,
            closed_at=next_closed,
            event_ids=incident.event_ids if event_ids is None else tuple(event_ids),
            evidence_ids=(
                incident.evidence_ids if evidence_ids is None else tuple(evidence_ids)
            ),
            indicator_ids=(
                incident.indicator_ids
                if indicator_ids is None
                else tuple(indicator_ids)
            ),
            note_ids=incident.note_ids if note_ids is None else tuple(note_ids),
            tags=incident.tags if tags is None else tuple(tags),
        )
    )


def _build_incident_for_validation(
    *,
    incident_id: str,
    title: str,
    summary: str,
    severity: str,
    status: str,
    created_at: str,
    updated_at: str,
    opened_at: str,
    closed_at: str | None,
    event_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    indicator_ids: tuple[str, ...],
    note_ids: tuple[str, ...],
    tags: tuple[str, ...],
) -> SecurityIncident:
    """Build an incident shell that validate_security_incident will normalize."""
    normalized_severity = validate_controlled_value(
        severity,
        field_name="incident severity",
        allowed=EVENT_SEVERITIES,
    )
    normalized_status = validate_controlled_value(
        status,
        field_name="incident status",
        allowed=INCIDENT_STATUSES,
    )
    return SecurityIncident(
        incident_id=incident_id,
        title=title,
        summary=summary,
        severity=normalized_severity,  # type: ignore[arg-type]
        status=normalized_status,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=updated_at,
        opened_at=opened_at,
        closed_at=closed_at,
        event_ids=event_ids,
        evidence_ids=evidence_ids,
        indicator_ids=indicator_ids,
        note_ids=note_ids,
        tags=tags,
    )


def validate_security_incident(incident: SecurityIncident) -> SecurityIncident:
    """Validate every field of a security incident and return a normalized record."""
    incident_id = validate_security_id(incident.incident_id, field_name="Incident ID")
    title = require_non_blank_text(
        incident.title,
        field_name="Incident title",
        max_length=MAX_INCIDENT_TITLE_LENGTH,
    )
    summary = require_non_blank_text(
        incident.summary,
        field_name="Incident summary",
        max_length=MAX_INCIDENT_SUMMARY_LENGTH,
    )
    severity = validate_controlled_value(
        incident.severity,
        field_name="incident severity",
        allowed=EVENT_SEVERITIES,
    )
    status = validate_controlled_value(
        incident.status,
        field_name="incident status",
        allowed=INCIDENT_STATUSES,
    )
    created_at = validate_utc_timestamp(incident.created_at, field_name="Created at")
    updated_at = validate_utc_timestamp(incident.updated_at, field_name="Updated at")
    opened_at = validate_utc_timestamp(incident.opened_at, field_name="Opened at")
    closed_at = validate_optional_utc_timestamp(
        incident.closed_at,
        field_name="Closed at",
    )

    if closed_at is not None and status not in CLOSED_INCIDENT_STATUSES:
        raise SecurityValidationError(
            "Closed timestamp may exist only for resolved or closed incidents."
        )
    if status in CLOSED_INCIDENT_STATUSES and closed_at is None:
        raise SecurityValidationError(
            "Resolved or closed incidents require a closed timestamp."
        )

    return SecurityIncident(
        incident_id=incident_id,
        title=title,
        summary=summary,
        severity=severity,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=updated_at,
        opened_at=opened_at,
        closed_at=closed_at,
        event_ids=normalize_id_list(incident.event_ids, field_name="Event ID"),
        evidence_ids=normalize_id_list(
            incident.evidence_ids,
            field_name="Evidence ID",
        ),
        indicator_ids=normalize_id_list(
            incident.indicator_ids,
            field_name="Indicator ID",
        ),
        note_ids=normalize_id_list(incident.note_ids, field_name="Note ID"),
        tags=normalize_tags(incident.tags),
    )
