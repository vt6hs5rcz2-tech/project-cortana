"""Local slash-command handlers for the Milestone 8 security foundation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from src.command_argument_utils import extract_command_argument
from src.config import MAX_SECURITY_LIST_PREVIEW_CHARS
from src.evidence_store import EvidenceStore, EvidenceStoreError
from src.incident_repository import (
    IncidentRelationshipError,
    IncidentRepository,
    IncidentStorageError,
)
from src.security_common import (
    EVENT_SEVERITIES,
    EVENT_STATUSES,
    INCIDENT_STATUSES,
    INDICATOR_TYPES,
    NOTE_TYPES,
    BlankSecurityFieldError,
    InvalidSecurityEnumError,
    InvalidSecurityIdError,
    SecurityFieldTooLongError,
    SecurityValidationError,
    validate_security_id,
)
from src.security_custody import create_custody_entry
from src.security_event import (
    create_security_event,
    update_security_event_status,
)
from src.security_evidence import create_evidence_record
from src.security_incident import (
    create_security_incident,
    update_security_incident_status,
)
from src.security_indicator import (
    create_security_indicator,
    indicator_log_reference,
)
from src.security_note import create_incident_note

logger = logging.getLogger("ProjectCortana")

FIELD_DELIMITER = " | "

COMMAND_EVENT_NEW = "event-new"
COMMAND_EVENTS = "events"
COMMAND_EVENT = "event"
COMMAND_EVENT_STATUS = "event-status"
COMMAND_INCIDENT_NEW = "incident-new"
COMMAND_INCIDENTS = "incidents"
COMMAND_INCIDENT = "incident"
COMMAND_INCIDENT_STATUS = "incident-status"
COMMAND_INCIDENT_LINK_EVENT = "incident-link-event"
COMMAND_INCIDENT_UNLINK_EVENT = "incident-unlink-event"
COMMAND_INCIDENT_LINK_EVIDENCE = "incident-link-evidence"
COMMAND_INDICATOR_ADD = "indicator-add"
COMMAND_INDICATORS = "indicators"
COMMAND_INDICATOR = "indicator"
COMMAND_EVIDENCE_REGISTER = "evidence-register"
COMMAND_EVIDENCE = "evidence"
COMMAND_EVIDENCE_SHOW = "evidence-show"
COMMAND_EVIDENCE_VERIFY = "evidence-verify"
COMMAND_INCIDENT_ADD_NOTE = "incident-add-note"
COMMAND_INCIDENT_NOTES = "incident-notes"
COMMAND_INCIDENT_TIMELINE = "incident-timeline"

SECURITY_COMMAND_NAMES = frozenset(
    {
        COMMAND_EVENT_NEW,
        COMMAND_EVENTS,
        COMMAND_EVENT,
        COMMAND_EVENT_STATUS,
        COMMAND_INCIDENT_NEW,
        COMMAND_INCIDENTS,
        COMMAND_INCIDENT,
        COMMAND_INCIDENT_STATUS,
        COMMAND_INCIDENT_LINK_EVENT,
        COMMAND_INCIDENT_UNLINK_EVENT,
        COMMAND_INCIDENT_LINK_EVIDENCE,
        COMMAND_INDICATOR_ADD,
        COMMAND_INDICATORS,
        COMMAND_INDICATOR,
        COMMAND_EVIDENCE_REGISTER,
        COMMAND_EVIDENCE,
        COMMAND_EVIDENCE_SHOW,
        COMMAND_EVIDENCE_VERIFY,
        COMMAND_INCIDENT_ADD_NOTE,
        COMMAND_INCIDENT_NOTES,
        COMMAND_INCIDENT_TIMELINE,
    }
)

EVENT_NEW_USAGE = (
    "Cortana: Usage: /event-new <severity> | <title> | <description>"
)
EVENT_STATUS_USAGE = (
    "Cortana: Usage: /event-status <event-id> <status>"
)
EVENT_MISSING_ID = "Cortana: Please provide an event ID. Usage: /event <event-id>"
EVENT_NOT_FOUND_TEMPLATE = "Cortana: No saved event found with ID '{event_id}'."
EVENTS_EMPTY = "Cortana: No saved security events."

INCIDENT_NEW_USAGE = (
    "Cortana: Usage: /incident-new <severity> | <title> | <summary>"
)
INCIDENT_STATUS_USAGE = (
    "Cortana: Usage: /incident-status <incident-id> <status>"
)
INCIDENT_MISSING_ID = (
    "Cortana: Please provide an incident ID. Usage: /incident <incident-id>"
)
INCIDENT_NOT_FOUND_TEMPLATE = (
    "Cortana: No saved incident found with ID '{incident_id}'."
)
INCIDENTS_EMPTY = "Cortana: No saved security incidents."
INCIDENT_LINK_USAGE = (
    "Cortana: Usage: /incident-link-event <incident-id> <event-id>"
)
INCIDENT_UNLINK_USAGE = (
    "Cortana: Usage: /incident-unlink-event <incident-id> <event-id>"
)
INCIDENT_LINK_EVIDENCE_USAGE = (
    "Cortana: Usage: /incident-link-evidence <incident-id> <evidence-id>"
)

INDICATOR_ADD_USAGE = (
    "Cortana: Usage: /indicator-add <type> | <value> | <confidence>"
)
INDICATOR_MISSING_ID = (
    "Cortana: Please provide an indicator ID. Usage: /indicator <indicator-id>"
)
INDICATOR_NOT_FOUND_TEMPLATE = (
    "Cortana: No saved indicator found with ID '{indicator_id}'."
)
INDICATORS_EMPTY = "Cortana: No saved indicators."

EVIDENCE_REGISTER_USAGE = (
    "Cortana: Usage: /evidence-register <path> | <title> | <description>"
)
EVIDENCE_MISSING_ID = (
    "Cortana: Please provide an evidence ID. Usage: /evidence-show <evidence-id>"
)
EVIDENCE_VERIFY_MISSING_ID = (
    "Cortana: Please provide an evidence ID. Usage: /evidence-verify <evidence-id>"
)
EVIDENCE_NOT_FOUND_TEMPLATE = (
    "Cortana: No saved evidence found with ID '{evidence_id}'."
)
EVIDENCE_EMPTY = "Cortana: No saved evidence records."

NOTE_ADD_USAGE = (
    "Cortana: Usage: /incident-add-note <incident-id> | <note-type> | <text>"
)
NOTES_MISSING_ID = (
    "Cortana: Please provide an incident ID. Usage: /incident-notes <incident-id>"
)
TIMELINE_MISSING_ID = (
    "Cortana: Please provide an incident ID. "
    "Usage: /incident-timeline <incident-id>"
)
NOTES_EMPTY = "Cortana: No analyst notes are saved for that incident."
TIMELINE_EMPTY = "Cortana: No timeline entries are available for that incident."


@dataclass(frozen=True)
class SecurityCommandContext:
    """Inputs available to Milestone 8 security command handlers."""

    message: str
    incident_repository: IncidentRepository
    evidence_store: EvidenceStore


@dataclass(frozen=True)
class SecurityCommandResult:
    """Local message returned by a security command handler."""

    message: str


SecurityCommandHandler = Callable[[SecurityCommandContext], SecurityCommandResult]


def split_delimited_fields(argument: str, expected_count: int) -> list[str] | None:
    """Split multi-field arguments on the documented `` | `` delimiter."""
    parts = argument.split(FIELD_DELIMITER)
    if len(parts) != expected_count:
        return None
    return parts


def bounded_preview(text: str) -> str:
    """Return a bounded single-line preview for list command output."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= MAX_SECURITY_LIST_PREVIEW_CHARS:
        return cleaned
    return f"{cleaned[:MAX_SECURITY_LIST_PREVIEW_CHARS]}..."


def handle_security_command(
    command_name: str,
    context: SecurityCommandContext,
) -> SecurityCommandResult | None:
    """Dispatch a Milestone 8 security command, or return None when unknown."""
    handler = SECURITY_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    return handler(context)


def _handle_event_new(context: SecurityCommandContext) -> SecurityCommandResult:
    """Create one security event from delimited severity, title, and description."""
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return SecurityCommandResult(message=EVENT_NEW_USAGE)

    severity, title, description = fields
    try:
        event = create_security_event(
            event_type="manual",
            title=title,
            description=description,
            severity=severity,
        )
        saved = context.incident_repository.add_event(event)
    except InvalidSecurityEnumError:
        return SecurityCommandResult(
            message=(
                "Cortana: Invalid event severity. "
                f"Allowed values: {', '.join(sorted(EVENT_SEVERITIES))}."
            )
        )
    except (
        BlankSecurityFieldError,
        SecurityFieldTooLongError,
        SecurityValidationError,
    ):
        return SecurityCommandResult(message=EVENT_NEW_USAGE)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Security event created event_id=%s severity=%s status=%s",
        saved.event_id,
        saved.severity,
        saved.status,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Security event saved ({saved.event_id}). "
            f"Severity: {saved.severity}. Status: {saved.status}."
        )
    )


def _handle_events(context: SecurityCommandContext) -> SecurityCommandResult:
    """List saved security events with bounded description previews."""
    try:
        events = context.incident_repository.list_events()
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if not events:
        return SecurityCommandResult(message=EVENTS_EMPTY)

    lines = ["Cortana: Saved security events:"]
    for event in events:
        lines.append(
            f"  [{event.event_id}] {event.severity}/{event.status} {event.title}"
        )
        lines.append(f"    {bounded_preview(event.description)}")
    return SecurityCommandResult(message="\n".join(lines))


def _handle_event(context: SecurityCommandContext) -> SecurityCommandResult:
    """Show one full user-owned security event record."""
    event_id_argument = extract_command_argument(context.message).strip()
    if not event_id_argument:
        return SecurityCommandResult(message=EVENT_MISSING_ID)

    try:
        event_id = validate_security_id(event_id_argument, field_name="Event ID")
        event = context.incident_repository.get_event(event_id)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=EVENT_NOT_FOUND_TEMPLATE.format(event_id=event_id_argument)
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if event is None:
        return SecurityCommandResult(
            message=EVENT_NOT_FOUND_TEMPLATE.format(event_id=event_id_argument)
        )

    related = event.related_incident_id or "(none)"
    message = (
        "Cortana: Security event\n"
        f"  ID: {event.event_id}\n"
        f"  Type: {event.event_type}\n"
        f"  Title: {event.title}\n"
        f"  Severity: {event.severity}\n"
        f"  Status: {event.status}\n"
        f"  Source: {event.source}\n"
        f"  Observed at: {event.observed_at}\n"
        f"  Created at: {event.created_at}\n"
        f"  Updated at: {event.updated_at}\n"
        f"  Related incident: {related}\n"
        f"  Tags: {', '.join(event.tags) if event.tags else '(none)'}\n"
        f"  Indicator IDs: {', '.join(event.indicator_ids) if event.indicator_ids else '(none)'}\n"
        f"  Evidence IDs: {', '.join(event.evidence_ids) if event.evidence_ids else '(none)'}\n"
        "  Description:\n"
        f"{event.description}"
    )
    return SecurityCommandResult(message=message)


def _handle_event_status(context: SecurityCommandContext) -> SecurityCommandResult:
    """Update one event status through a validated replacement record."""
    argument = extract_command_argument(context.message).strip()
    parts = argument.split(maxsplit=1)
    if len(parts) != 2:
        return SecurityCommandResult(message=EVENT_STATUS_USAGE)

    event_id_argument, status = parts
    try:
        event_id = validate_security_id(event_id_argument, field_name="Event ID")
        event = context.incident_repository.get_event(event_id)
        if event is None:
            return SecurityCommandResult(
                message=EVENT_NOT_FOUND_TEMPLATE.format(event_id=event_id_argument)
            )
        updated = update_security_event_status(event, status)
        saved = context.incident_repository.update_event(updated)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=EVENT_NOT_FOUND_TEMPLATE.format(event_id=event_id_argument)
        )
    except InvalidSecurityEnumError:
        return SecurityCommandResult(
            message=(
                "Cortana: Invalid event status. "
                f"Allowed values: {', '.join(sorted(EVENT_STATUSES))}."
            )
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Security event status updated event_id=%s status=%s",
        saved.event_id,
        saved.status,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Event '{saved.event_id}' status updated to {saved.status}."
        )
    )


def _handle_incident_new(context: SecurityCommandContext) -> SecurityCommandResult:
    """Create one security incident from delimited severity, title, and summary."""
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return SecurityCommandResult(message=INCIDENT_NEW_USAGE)

    severity, title, summary = fields
    try:
        incident = create_security_incident(
            title=title,
            summary=summary,
            severity=severity,
        )
        saved = context.incident_repository.add_incident(incident)
    except InvalidSecurityEnumError:
        return SecurityCommandResult(
            message=(
                "Cortana: Invalid incident severity. "
                f"Allowed values: {', '.join(sorted(EVENT_SEVERITIES))}."
            )
        )
    except (
        BlankSecurityFieldError,
        SecurityFieldTooLongError,
        SecurityValidationError,
    ):
        return SecurityCommandResult(message=INCIDENT_NEW_USAGE)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Security incident created incident_id=%s severity=%s status=%s",
        saved.incident_id,
        saved.severity,
        saved.status,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Security incident saved ({saved.incident_id}). "
            f"Severity: {saved.severity}. Status: {saved.status}."
        )
    )


def _handle_incidents(context: SecurityCommandContext) -> SecurityCommandResult:
    """List saved incidents with bounded summary previews."""
    try:
        incidents = context.incident_repository.list_incidents()
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if not incidents:
        return SecurityCommandResult(message=INCIDENTS_EMPTY)

    lines = ["Cortana: Saved security incidents:"]
    for incident in incidents:
        lines.append(
            f"  [{incident.incident_id}] {incident.severity}/{incident.status} "
            f"{incident.title}"
        )
        lines.append(f"    {bounded_preview(incident.summary)}")
    return SecurityCommandResult(message="\n".join(lines))


def _handle_incident(context: SecurityCommandContext) -> SecurityCommandResult:
    """Show one full user-owned security incident record."""
    incident_id_argument = extract_command_argument(context.message).strip()
    if not incident_id_argument:
        return SecurityCommandResult(message=INCIDENT_MISSING_ID)

    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        incident = context.incident_repository.get_incident(incident_id)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                incident_id=incident_id_argument
            )
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if incident is None:
        return SecurityCommandResult(
            message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                incident_id=incident_id_argument
            )
        )

    closed = incident.closed_at or "(none)"
    message = (
        "Cortana: Security incident\n"
        f"  ID: {incident.incident_id}\n"
        f"  Title: {incident.title}\n"
        f"  Severity: {incident.severity}\n"
        f"  Status: {incident.status}\n"
        f"  Created at: {incident.created_at}\n"
        f"  Updated at: {incident.updated_at}\n"
        f"  Opened at: {incident.opened_at}\n"
        f"  Closed at: {closed}\n"
        f"  Event IDs: {', '.join(incident.event_ids) if incident.event_ids else '(none)'}\n"
        f"  Evidence IDs: {', '.join(incident.evidence_ids) if incident.evidence_ids else '(none)'}\n"
        f"  Indicator IDs: {', '.join(incident.indicator_ids) if incident.indicator_ids else '(none)'}\n"
        f"  Note IDs: {', '.join(incident.note_ids) if incident.note_ids else '(none)'}\n"
        f"  Tags: {', '.join(incident.tags) if incident.tags else '(none)'}\n"
        "  Summary:\n"
        f"{incident.summary}"
    )
    return SecurityCommandResult(message=message)


def _handle_incident_status(context: SecurityCommandContext) -> SecurityCommandResult:
    """Update one incident status, including closed-timestamp rules."""
    argument = extract_command_argument(context.message).strip()
    parts = argument.split(maxsplit=1)
    if len(parts) != 2:
        return SecurityCommandResult(message=INCIDENT_STATUS_USAGE)

    incident_id_argument, status = parts
    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        incident = context.incident_repository.get_incident(incident_id)
        if incident is None:
            return SecurityCommandResult(
                message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                    incident_id=incident_id_argument
                )
            )
        updated = update_security_incident_status(incident, status)
        saved = context.incident_repository.update_incident(updated)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                incident_id=incident_id_argument
            )
        )
    except InvalidSecurityEnumError:
        return SecurityCommandResult(
            message=(
                "Cortana: Invalid incident status. "
                f"Allowed values: {', '.join(sorted(INCIDENT_STATUSES))}."
            )
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Security incident status updated incident_id=%s status=%s",
        saved.incident_id,
        saved.status,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Incident '{saved.incident_id}' status updated to {saved.status}."
        )
    )


def _handle_incident_link_event(
    context: SecurityCommandContext,
) -> SecurityCommandResult:
    """Link an event to an incident on both sides."""
    argument = extract_command_argument(context.message).strip()
    parts = argument.split()
    if len(parts) != 2:
        return SecurityCommandResult(message=INCIDENT_LINK_USAGE)

    incident_id_argument, event_id_argument = parts
    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        event_id = validate_security_id(event_id_argument, field_name="Event ID")
        incident, event = context.incident_repository.link_event_to_incident(
            incident_id,
            event_id,
        )
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message="Cortana: Incident and event IDs must be valid UUIDs."
        )
    except IncidentRelationshipError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Linked event to incident incident_id=%s event_id=%s",
        incident.incident_id,
        event.event_id,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Linked event '{event.event_id}' to incident "
            f"'{incident.incident_id}'."
        )
    )


def _handle_incident_unlink_event(
    context: SecurityCommandContext,
) -> SecurityCommandResult:
    """Unlink an event from an incident on both sides."""
    argument = extract_command_argument(context.message).strip()
    parts = argument.split()
    if len(parts) != 2:
        return SecurityCommandResult(message=INCIDENT_UNLINK_USAGE)

    incident_id_argument, event_id_argument = parts
    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        event_id = validate_security_id(event_id_argument, field_name="Event ID")
        incident, event = context.incident_repository.unlink_event_from_incident(
            incident_id,
            event_id,
        )
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message="Cortana: Incident and event IDs must be valid UUIDs."
        )
    except IncidentRelationshipError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Unlinked event from incident incident_id=%s event_id=%s",
        incident.incident_id,
        event.event_id,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Unlinked event '{event.event_id}' from incident "
            f"'{incident.incident_id}'."
        )
    )


def _handle_incident_link_evidence(
    context: SecurityCommandContext,
) -> SecurityCommandResult:
    """Link registered evidence to an incident on both sides (idempotent)."""
    argument = extract_command_argument(context.message).strip()
    parts = argument.split()
    if len(parts) != 2:
        return SecurityCommandResult(message=INCIDENT_LINK_EVIDENCE_USAGE)

    incident_id_argument, evidence_id_argument = parts
    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        evidence_id = validate_security_id(
            evidence_id_argument,
            field_name="Evidence ID",
        )
        incident, evidence = context.incident_repository.link_evidence_to_incident(
            incident_id,
            evidence_id,
        )
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message="Cortana: Incident and evidence IDs must be valid UUIDs."
        )
    except IncidentRelationshipError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Linked evidence to incident incident_id=%s evidence_id=%s",
        incident.incident_id,
        evidence.evidence_id,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Linked evidence '{evidence.evidence_id}' to incident "
            f"'{incident.incident_id}'."
        )
    )


def _handle_indicator_add(context: SecurityCommandContext) -> SecurityCommandResult:
    """Add one indicator with type, value, and confidence."""
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return SecurityCommandResult(message=INDICATOR_ADD_USAGE)

    indicator_type, value, confidence_text = fields
    try:
        confidence = int(confidence_text.strip())
    except ValueError:
        return SecurityCommandResult(
            message="Cortana: Indicator confidence must be an integer from 0 to 100."
        )

    try:
        indicator = create_security_indicator(
            indicator_type=indicator_type,
            value=value,
            confidence=confidence,
        )
        saved = context.incident_repository.add_indicator(indicator)
    except InvalidSecurityEnumError:
        return SecurityCommandResult(
            message=(
                "Cortana: Invalid indicator type. "
                f"Allowed values: {', '.join(sorted(INDICATOR_TYPES))}."
            )
        )
    except SecurityValidationError:
        return SecurityCommandResult(
            message="Cortana: The indicator value or confidence is invalid."
        )
    except IncidentRelationshipError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Security indicator added indicator_id=%s %s confidence=%s",
        saved.indicator_id,
        indicator_log_reference(saved),
        saved.confidence,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Indicator saved ({saved.indicator_id}). "
            f"Type: {saved.indicator_type}. Confidence: {saved.confidence}."
        )
    )


def _handle_indicators(context: SecurityCommandContext) -> SecurityCommandResult:
    """List saved indicators without dumping sensitive values in bulk logs."""
    try:
        indicators = context.incident_repository.list_indicators()
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if not indicators:
        return SecurityCommandResult(message=INDICATORS_EMPTY)

    lines = ["Cortana: Saved indicators:"]
    for indicator in indicators:
        lines.append(
            f"  [{indicator.indicator_id}] {indicator.indicator_type} "
            f"confidence={indicator.confidence} value={indicator.normalized_value}"
        )
    return SecurityCommandResult(message="\n".join(lines))


def _handle_indicator(context: SecurityCommandContext) -> SecurityCommandResult:
    """Show one full indicator record."""
    indicator_id_argument = extract_command_argument(context.message).strip()
    if not indicator_id_argument:
        return SecurityCommandResult(message=INDICATOR_MISSING_ID)

    try:
        indicator_id = validate_security_id(
            indicator_id_argument,
            field_name="Indicator ID",
        )
        indicator = context.incident_repository.get_indicator(indicator_id)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=INDICATOR_NOT_FOUND_TEMPLATE.format(
                indicator_id=indicator_id_argument
            )
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if indicator is None:
        return SecurityCommandResult(
            message=INDICATOR_NOT_FOUND_TEMPLATE.format(
                indicator_id=indicator_id_argument
            )
        )

    notes = indicator.notes or "(none)"
    message = (
        "Cortana: Security indicator\n"
        f"  ID: {indicator.indicator_id}\n"
        f"  Type: {indicator.indicator_type}\n"
        f"  Normalized value: {indicator.normalized_value}\n"
        f"  Original value: {indicator.original_value}\n"
        f"  Confidence: {indicator.confidence}\n"
        f"  First seen: {indicator.first_seen_at}\n"
        f"  Last seen: {indicator.last_seen_at}\n"
        f"  Created at: {indicator.created_at}\n"
        f"  Tags: {', '.join(indicator.tags) if indicator.tags else '(none)'}\n"
        f"  Related event IDs: {', '.join(indicator.related_event_ids) if indicator.related_event_ids else '(none)'}\n"
        f"  Related incident IDs: {', '.join(indicator.related_incident_ids) if indicator.related_incident_ids else '(none)'}\n"
        f"  Notes: {notes}"
    )
    return SecurityCommandResult(message=message)


def _rollback_copied_evidence(store: EvidenceStore, evidence_id: str) -> None:
    """Best-effort rollback of a copied evidence binary after metadata failure."""
    store.discard_stored_copy(evidence_id)


def _handle_evidence_register(context: SecurityCommandContext) -> SecurityCommandResult:
    """Register evidence by copying bytes and recording metadata plus custody."""
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return SecurityCommandResult(message=EVIDENCE_REGISTER_USAGE)

    path_argument, title, description = fields
    evidence_id = str(uuid4())
    copied_bytes = False

    try:
        sha256_hash, source_size, original_filename = (
            context.evidence_store.copy_from_path(
                path_argument,
                evidence_id=evidence_id,
            )
        )
        copied_bytes = True
        evidence = create_evidence_record(
            evidence_id=evidence_id,
            evidence_type="file",
            title=title,
            description=description,
            sha256_hash=sha256_hash,
            source_size_bytes=source_size,
            collector="local-user",
            storage_status="copied",
            original_filename=original_filename,
        )
        registered = create_custody_entry(
            evidence_id=evidence_id,
            action="registered",
            actor="local-user",
            reason="Explicit /evidence-register command",
            resulting_hash=sha256_hash,
        )
        copied = create_custody_entry(
            evidence_id=evidence_id,
            action="copied",
            actor="local-user",
            reason="Local evidence byte copy completed",
            previous_hash=sha256_hash,
            resulting_hash=sha256_hash,
        )
        saved = context.incident_repository.add_evidence(
            evidence,
            [registered, copied],
        )
    except EvidenceStoreError as error:
        return SecurityCommandResult(message=error.user_message)
    except (
        BlankSecurityFieldError,
        SecurityFieldTooLongError,
        SecurityValidationError,
    ):
        if copied_bytes:
            _rollback_copied_evidence(context.evidence_store, evidence_id)
        return SecurityCommandResult(message=EVIDENCE_REGISTER_USAGE)
    except IncidentStorageError as error:
        if copied_bytes:
            _rollback_copied_evidence(context.evidence_store, evidence_id)
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Evidence registered evidence_id=%s storage_status=%s size_bytes=%s",
        saved.evidence_id,
        saved.storage_status,
        saved.source_size_bytes,
    )
    filename = saved.original_filename or "(none)"
    return SecurityCommandResult(
        message=(
            f"Cortana: Evidence registered ({saved.evidence_id}). "
            f"Filename: {filename}. SHA-256: {saved.sha256_hash}. "
            f"Storage: {saved.storage_status}."
        )
    )


def _handle_evidence(context: SecurityCommandContext) -> SecurityCommandResult:
    """List evidence metadata without exposing storage paths."""
    try:
        records = context.incident_repository.list_evidence()
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if not records:
        return SecurityCommandResult(message=EVIDENCE_EMPTY)

    lines = ["Cortana: Saved evidence records:"]
    for evidence in records:
        filename = evidence.original_filename or "(none)"
        lines.append(
            f"  [{evidence.evidence_id}] {evidence.title} "
            f"file={filename} status={evidence.storage_status}"
        )
        lines.append(f"    {bounded_preview(evidence.description)}")
    return SecurityCommandResult(message="\n".join(lines))


def _handle_evidence_show(context: SecurityCommandContext) -> SecurityCommandResult:
    """Show one evidence metadata record without storage paths."""
    evidence_id_argument = extract_command_argument(context.message).strip()
    if not evidence_id_argument:
        return SecurityCommandResult(message=EVIDENCE_MISSING_ID)

    try:
        evidence_id = validate_security_id(
            evidence_id_argument,
            field_name="Evidence ID",
        )
        evidence = context.incident_repository.get_evidence(evidence_id)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=EVIDENCE_NOT_FOUND_TEMPLATE.format(
                evidence_id=evidence_id_argument
            )
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if evidence is None:
        return SecurityCommandResult(
            message=EVIDENCE_NOT_FOUND_TEMPLATE.format(
                evidence_id=evidence_id_argument
            )
        )

    filename = evidence.original_filename or "(none)"
    message = (
        "Cortana: Evidence record\n"
        f"  ID: {evidence.evidence_id}\n"
        f"  Type: {evidence.evidence_type}\n"
        f"  Title: {evidence.title}\n"
        f"  Original filename: {filename}\n"
        f"  SHA-256: {evidence.sha256_hash}\n"
        f"  Source size (bytes): {evidence.source_size_bytes}\n"
        f"  Collected at: {evidence.collected_at}\n"
        f"  Recorded at: {evidence.recorded_at}\n"
        f"  Collector: {evidence.collector}\n"
        f"  Storage status: {evidence.storage_status}\n"
        f"  Related event IDs: {', '.join(evidence.related_event_ids) if evidence.related_event_ids else '(none)'}\n"
        f"  Related incident IDs: {', '.join(evidence.related_incident_ids) if evidence.related_incident_ids else '(none)'}\n"
        f"  Custody entry IDs: {', '.join(evidence.chain_of_custody_entry_ids) if evidence.chain_of_custody_entry_ids else '(none)'}\n"
        f"  Tags: {', '.join(evidence.tags) if evidence.tags else '(none)'}\n"
        "  Description:\n"
        f"{evidence.description}"
    )
    return SecurityCommandResult(message=message)


def _handle_evidence_verify(context: SecurityCommandContext) -> SecurityCommandResult:
    """Verify a stored evidence copy and append a custody verification entry."""
    evidence_id_argument = extract_command_argument(context.message).strip()
    if not evidence_id_argument:
        return SecurityCommandResult(message=EVIDENCE_VERIFY_MISSING_ID)

    try:
        evidence_id = validate_security_id(
            evidence_id_argument,
            field_name="Evidence ID",
        )
        evidence = context.incident_repository.get_evidence(evidence_id)
        if evidence is None:
            return SecurityCommandResult(
                message=EVIDENCE_NOT_FOUND_TEMPLATE.format(
                    evidence_id=evidence_id_argument
                )
            )

        result = context.evidence_store.verify_stored_hash(
            evidence.evidence_id,
            evidence.sha256_hash,
        )
        if result == "match":
            reason = "Stored evidence hash matched recorded digest"
            resulting_hash = evidence.sha256_hash
            user_message = (
                f"Cortana: Evidence '{evidence.evidence_id}' verified. "
                "Hash match."
            )
        elif result == "mismatch":
            reason = "Stored evidence hash did not match recorded digest"
            resulting_hash = None
            user_message = (
                f"Cortana: Evidence '{evidence.evidence_id}' verification failed. "
                "Hash mismatch."
            )
        else:
            reason = "Stored evidence copy was missing"
            resulting_hash = None
            user_message = (
                f"Cortana: Evidence '{evidence.evidence_id}' verification failed. "
                "Stored copy missing."
            )

        entry = create_custody_entry(
            evidence_id=evidence.evidence_id,
            action="verified",
            actor="local-user",
            reason=reason,
            previous_hash=evidence.sha256_hash,
            resulting_hash=resulting_hash,
        )
        context.incident_repository.append_custody_entry(evidence.evidence_id, entry)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=EVIDENCE_NOT_FOUND_TEMPLATE.format(
                evidence_id=evidence_id_argument
            )
        )
    except EvidenceStoreError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Evidence verified evidence_id=%s action=verified result=%s",
        evidence_id_argument.strip(),
        result,
    )
    return SecurityCommandResult(message=user_message)


def _handle_incident_add_note(context: SecurityCommandContext) -> SecurityCommandResult:
    """Add one analyst note to an incident without logging note text."""
    fields = split_delimited_fields(extract_command_argument(context.message), 3)
    if fields is None:
        return SecurityCommandResult(message=NOTE_ADD_USAGE)

    incident_id_argument, note_type, text = fields
    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        note = create_incident_note(
            incident_id=incident_id,
            author="local-user",
            text=text,
            note_type=note_type,
        )
        saved = context.incident_repository.add_note(note)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                incident_id=incident_id_argument.strip()
            )
        )
    except InvalidSecurityEnumError:
        return SecurityCommandResult(
            message=(
                "Cortana: Invalid note type. "
                f"Allowed values: {', '.join(sorted(NOTE_TYPES))}."
            )
        )
    except (
        BlankSecurityFieldError,
        SecurityFieldTooLongError,
        SecurityValidationError,
    ):
        return SecurityCommandResult(message=NOTE_ADD_USAGE)
    except IncidentRelationshipError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    logger.info(
        "Incident note added note_id=%s incident_id=%s note_type=%s",
        saved.note_id,
        saved.incident_id,
        saved.note_type,
    )
    return SecurityCommandResult(
        message=(
            f"Cortana: Note saved ({saved.note_id}) on incident "
            f"'{saved.incident_id}'."
        )
    )


def _handle_incident_notes(context: SecurityCommandContext) -> SecurityCommandResult:
    """List analyst notes for one incident with bounded text previews."""
    incident_id_argument = extract_command_argument(context.message).strip()
    if not incident_id_argument:
        return SecurityCommandResult(message=NOTES_MISSING_ID)

    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        incident = context.incident_repository.get_incident(incident_id)
        if incident is None:
            return SecurityCommandResult(
                message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                    incident_id=incident_id_argument
                )
            )
        notes = context.incident_repository.list_notes(incident_id)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                incident_id=incident_id_argument
            )
        )
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if not notes:
        return SecurityCommandResult(message=NOTES_EMPTY)

    lines = [f"Cortana: Analyst notes for incident '{incident_id}':"]
    for note in notes:
        lines.append(
            f"  [{note.note_id}] {note.note_type} by {note.author} "
            f"at {note.created_at}"
        )
        lines.append(f"    {bounded_preview(note.text)}")
    return SecurityCommandResult(message="\n".join(lines))


def _handle_incident_timeline(
    context: SecurityCommandContext,
) -> SecurityCommandResult:
    """Show a derived chronological timeline for one incident."""
    incident_id_argument = extract_command_argument(context.message).strip()
    if not incident_id_argument:
        return SecurityCommandResult(message=TIMELINE_MISSING_ID)

    try:
        incident_id = validate_security_id(
            incident_id_argument,
            field_name="Incident ID",
        )
        timeline = context.incident_repository.build_timeline(incident_id)
    except InvalidSecurityIdError:
        return SecurityCommandResult(
            message=INCIDENT_NOT_FOUND_TEMPLATE.format(
                incident_id=incident_id_argument
            )
        )
    except IncidentRelationshipError as error:
        return SecurityCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return SecurityCommandResult(message=error.user_message)

    if not timeline:
        return SecurityCommandResult(message=TIMELINE_EMPTY)

    lines = [f"Cortana: Timeline for incident '{incident_id}':"]
    for entry in timeline:
        lines.append(
            f"  [{entry.timestamp}] {entry.entry_type} {entry.entry_id}: "
            f"{entry.summary}"
        )
    return SecurityCommandResult(message="\n".join(lines))


SECURITY_COMMAND_HANDLERS: dict[str, SecurityCommandHandler] = {
    COMMAND_EVENT_NEW: _handle_event_new,
    COMMAND_EVENTS: _handle_events,
    COMMAND_EVENT: _handle_event,
    COMMAND_EVENT_STATUS: _handle_event_status,
    COMMAND_INCIDENT_NEW: _handle_incident_new,
    COMMAND_INCIDENTS: _handle_incidents,
    COMMAND_INCIDENT: _handle_incident,
    COMMAND_INCIDENT_STATUS: _handle_incident_status,
    COMMAND_INCIDENT_LINK_EVENT: _handle_incident_link_event,
    COMMAND_INCIDENT_UNLINK_EVENT: _handle_incident_unlink_event,
    COMMAND_INCIDENT_LINK_EVIDENCE: _handle_incident_link_evidence,
    COMMAND_INDICATOR_ADD: _handle_indicator_add,
    COMMAND_INDICATORS: _handle_indicators,
    COMMAND_INDICATOR: _handle_indicator,
    COMMAND_EVIDENCE_REGISTER: _handle_evidence_register,
    COMMAND_EVIDENCE: _handle_evidence,
    COMMAND_EVIDENCE_SHOW: _handle_evidence_show,
    COMMAND_EVIDENCE_VERIFY: _handle_evidence_verify,
    COMMAND_INCIDENT_ADD_NOTE: _handle_incident_add_note,
    COMMAND_INCIDENT_NOTES: _handle_incident_notes,
    COMMAND_INCIDENT_TIMELINE: _handle_incident_timeline,
}
