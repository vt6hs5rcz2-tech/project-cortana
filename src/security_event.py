"""Immutable security event model for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.config import (
    MAX_EVENT_DESCRIPTION_LENGTH,
    MAX_EVENT_SOURCE_LENGTH,
    MAX_EVENT_TITLE_LENGTH,
)
from src.security_common import (
    EVENT_SEVERITIES,
    EVENT_STATUSES,
    EventSeverity,
    EventStatus,
    normalize_id_list,
    normalize_tags,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_security_id,
    validate_security_id,
    validate_utc_timestamp,
)


@dataclass(frozen=True)
class SecurityEvent:
    """Immutable cybersecurity event recorded by an explicit user command."""

    event_id: str
    event_type: str
    title: str
    description: str
    severity: EventSeverity
    status: EventStatus
    source: str
    observed_at: str
    created_at: str
    updated_at: str
    related_incident_id: str | None
    tags: tuple[str, ...]
    indicator_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def create_security_event(
    *,
    event_type: str,
    title: str,
    description: str,
    severity: str,
    source: str = "manual",
    observed_at: str | None = None,
    status: str = "new",
    related_incident_id: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    indicator_ids: list[str] | tuple[str, ...] | None = None,
    evidence_ids: list[str] | tuple[str, ...] | None = None,
) -> SecurityEvent:
    """Create a validated immutable security event."""
    now = utc_timestamp()
    return validate_security_event(
        _build_event_for_validation(
            event_id=str(uuid4()),
            event_type=event_type,
            title=title,
            description=description,
            severity=severity,
            status=status,
            source=source,
            observed_at=observed_at or now,
            created_at=now,
            updated_at=now,
            related_incident_id=related_incident_id,
            tags=tuple(tags or ()),
            indicator_ids=tuple(indicator_ids or ()),
            evidence_ids=tuple(evidence_ids or ()),
        )
    )


def _build_event_for_validation(
    *,
    event_id: str,
    event_type: str,
    title: str,
    description: str,
    severity: str,
    status: str,
    source: str,
    observed_at: str,
    created_at: str,
    updated_at: str,
    related_incident_id: str | None,
    tags: tuple[str, ...],
    indicator_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> SecurityEvent:
    """Build an event shell that validate_security_event will normalize."""
    normalized_severity = validate_controlled_value(
        severity,
        field_name="event severity",
        allowed=EVENT_SEVERITIES,
    )
    normalized_status = validate_controlled_value(
        status,
        field_name="event status",
        allowed=EVENT_STATUSES,
    )
    return SecurityEvent(
        event_id=event_id,
        event_type=event_type,
        title=title,
        description=description,
        severity=normalized_severity,  # type: ignore[arg-type]
        status=normalized_status,  # type: ignore[arg-type]
        source=source,
        observed_at=observed_at,
        created_at=created_at,
        updated_at=updated_at,
        related_incident_id=related_incident_id,
        tags=tags,
        indicator_ids=indicator_ids,
        evidence_ids=evidence_ids,
    )


def update_security_event_status(event: SecurityEvent, status: str) -> SecurityEvent:
    """Return a validated replacement event with an updated status."""
    return replace_security_event(event, status=status)


def replace_security_event(
    event: SecurityEvent,
    *,
    event_type: str | None = None,
    title: str | None = None,
    description: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    source: str | None = None,
    observed_at: str | None = None,
    related_incident_id: str | None = None,
    clear_related_incident_id: bool = False,
    tags: list[str] | tuple[str, ...] | None = None,
    indicator_ids: list[str] | tuple[str, ...] | None = None,
    evidence_ids: list[str] | tuple[str, ...] | None = None,
) -> SecurityEvent:
    """Return a validated replacement event, preserving identity and created_at."""
    related = event.related_incident_id
    if clear_related_incident_id:
        related = None
    elif related_incident_id is not None:
        related = related_incident_id

    return validate_security_event(
        _build_event_for_validation(
            event_id=event.event_id,
            event_type=event.event_type if event_type is None else event_type,
            title=event.title if title is None else title,
            description=event.description if description is None else description,
            severity=event.severity if severity is None else severity,
            status=event.status if status is None else status,
            source=event.source if source is None else source,
            observed_at=event.observed_at if observed_at is None else observed_at,
            created_at=event.created_at,
            updated_at=utc_timestamp(),
            related_incident_id=related,
            tags=event.tags if tags is None else tuple(tags),
            indicator_ids=(
                event.indicator_ids if indicator_ids is None else tuple(indicator_ids)
            ),
            evidence_ids=(
                event.evidence_ids if evidence_ids is None else tuple(evidence_ids)
            ),
        )
    )


def validate_security_event(event: SecurityEvent) -> SecurityEvent:
    """Validate every field of a security event and return a normalized record."""
    event_id = validate_security_id(event.event_id, field_name="Event ID")
    event_type = require_non_blank_text(
        event.event_type,
        field_name="Event type",
        max_length=MAX_EVENT_TITLE_LENGTH,
    )
    title = require_non_blank_text(
        event.title,
        field_name="Event title",
        max_length=MAX_EVENT_TITLE_LENGTH,
    )
    description = require_non_blank_text(
        event.description,
        field_name="Event description",
        max_length=MAX_EVENT_DESCRIPTION_LENGTH,
    )
    severity = validate_controlled_value(
        event.severity,
        field_name="event severity",
        allowed=EVENT_SEVERITIES,
    )
    status = validate_controlled_value(
        event.status,
        field_name="event status",
        allowed=EVENT_STATUSES,
    )
    source = require_non_blank_text(
        event.source,
        field_name="Event source",
        max_length=MAX_EVENT_SOURCE_LENGTH,
    )
    observed_at = validate_utc_timestamp(event.observed_at, field_name="Observed at")
    created_at = validate_utc_timestamp(event.created_at, field_name="Created at")
    updated_at = validate_utc_timestamp(event.updated_at, field_name="Updated at")
    related_incident_id = validate_optional_security_id(
        event.related_incident_id,
        field_name="Related incident ID",
    )
    tags = normalize_tags(event.tags)
    indicator_ids = normalize_id_list(event.indicator_ids, field_name="Indicator ID")
    evidence_ids = normalize_id_list(event.evidence_ids, field_name="Evidence ID")

    return SecurityEvent(
        event_id=event_id,
        event_type=event_type,
        title=title,
        description=description,
        severity=severity,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        source=source,
        observed_at=observed_at,
        created_at=created_at,
        updated_at=updated_at,
        related_incident_id=related_incident_id,
        tags=tags,
        indicator_ids=indicator_ids,
        evidence_ids=evidence_ids,
    )
