"""Coordinated local JSON persistence for security incident records."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.config import INCIDENT_REPOSITORY_SCHEMA_VERSION
from src.security_custody import ChainOfCustodyEntry, validate_custody_entry
from src.security_event import (
    SecurityEvent,
    replace_security_event,
    validate_security_event,
)
from src.security_evidence import (
    EvidenceRecord,
    replace_evidence_record,
    validate_evidence_record,
)
from src.security_incident import (
    SecurityIncident,
    replace_security_incident,
    validate_security_incident,
)
from src.security_indicator import SecurityIndicator, validate_security_indicator
from src.security_note import IncidentNote, validate_incident_note
from src.security_timeline import IncidentTimelineEntry, build_incident_timeline

logger = logging.getLogger("ProjectCortana")

INCIDENT_LOAD_ERROR_MESSAGE = (
    "Cortana: Incident repository data could not be loaded safely. "
    "The existing repository file was left unchanged for inspection."
)
INCIDENT_SAVE_ERROR_MESSAGE = (
    "Cortana: Incident repository data could not be updated safely."
)


class IncidentStorageError(RuntimeError):
    """Raised when incident repository data cannot be loaded or saved safely."""

    def __init__(self, message: str = INCIDENT_LOAD_ERROR_MESSAGE) -> None:
        super().__init__(message)
        self.user_message = message


class IncidentRelationshipError(IncidentStorageError):
    """Raised when a relationship change is invalid or inconsistent."""


@dataclass
class _RepositoryState:
    """In-memory snapshot of the coordinated incident repository."""

    events: list[SecurityEvent] = field(default_factory=list)
    incidents: list[SecurityIncident] = field(default_factory=list)
    indicators: list[SecurityIndicator] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    custody_entries: list[ChainOfCustodyEntry] = field(default_factory=list)
    notes: list[IncidentNote] = field(default_factory=list)


class IncidentRepository(Protocol):
    """Protocol for coordinated security-incident persistence."""

    def list_events(self) -> list[SecurityEvent]:
        """Return saved security events in storage order."""

    def get_event(self, event_id: str) -> SecurityEvent | None:
        """Return one event by ID, or None when not found."""

    def add_event(self, event: SecurityEvent) -> SecurityEvent:
        """Persist one validated security event."""

    def update_event(self, event: SecurityEvent) -> SecurityEvent:
        """Replace one existing event with a validated record."""

    def list_incidents(self) -> list[SecurityIncident]:
        """Return saved incidents in storage order."""

    def get_incident(self, incident_id: str) -> SecurityIncident | None:
        """Return one incident by ID, or None when not found."""

    def add_incident(self, incident: SecurityIncident) -> SecurityIncident:
        """Persist one validated security incident."""

    def update_incident(self, incident: SecurityIncident) -> SecurityIncident:
        """Replace one existing incident with a validated record."""

    def link_event_to_incident(
        self,
        incident_id: str,
        event_id: str,
    ) -> tuple[SecurityIncident, SecurityEvent]:
        """Link an event and incident on both sides consistently."""

    def unlink_event_from_incident(
        self,
        incident_id: str,
        event_id: str,
    ) -> tuple[SecurityIncident, SecurityEvent]:
        """Unlink an event and incident on both sides consistently."""

    def list_indicators(self) -> list[SecurityIndicator]:
        """Return saved indicators in storage order."""

    def get_indicator(self, indicator_id: str) -> SecurityIndicator | None:
        """Return one indicator by ID, or None when not found."""

    def find_indicator(
        self,
        indicator_type: str,
        normalized_value: str,
    ) -> SecurityIndicator | None:
        """Return an indicator matching type and normalized value, if any."""

    def add_indicator(self, indicator: SecurityIndicator) -> SecurityIndicator:
        """Persist one validated indicator, rejecting duplicates."""

    def list_evidence(self) -> list[EvidenceRecord]:
        """Return saved evidence metadata in storage order."""

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        """Return one evidence record by ID, or None when not found."""

    def add_evidence(
        self,
        evidence: EvidenceRecord,
        custody_entries: list[ChainOfCustodyEntry],
    ) -> EvidenceRecord:
        """Persist evidence metadata and related custody entries together."""

    def append_custody_entry(
        self,
        evidence_id: str,
        entry: ChainOfCustodyEntry,
    ) -> ChainOfCustodyEntry:
        """Append one custody entry and link it to the evidence record."""

    def list_custody_entries(
        self,
        evidence_id: str | None = None,
    ) -> list[ChainOfCustodyEntry]:
        """Return custody entries, optionally filtered by evidence ID."""

    def add_note(self, note: IncidentNote) -> IncidentNote:
        """Persist one note and link it to its incident."""

    def list_notes(self, incident_id: str) -> list[IncidentNote]:
        """Return notes for one incident in storage order."""

    def build_timeline(self, incident_id: str) -> list[IncidentTimelineEntry]:
        """Derive a local timeline for one incident without persisting it."""

    def event_count(self) -> int:
        """Return saved event count."""

    def incident_count(self) -> int:
        """Return saved incident count."""

    def indicator_count(self) -> int:
        """Return saved indicator count."""

    def evidence_count(self) -> int:
        """Return saved evidence count."""


class JsonIncidentRepository:
    """UTF-8 JSON-backed coordinated repository for incident foundation data."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._state: _RepositoryState | None = None
        self._load_error: IncidentStorageError | None = None

    @property
    def file_path(self) -> Path:
        """Return the configured repository file path."""
        return self._file_path

    def list_events(self) -> list[SecurityEvent]:
        """Return a copy of saved events."""
        return list(self._ensure_loaded().events)

    def get_event(self, event_id: str) -> SecurityEvent | None:
        """Return one event by exact ID match."""
        for event in self._ensure_loaded().events:
            if event.event_id == event_id:
                return event
        return None

    def add_event(self, event: SecurityEvent) -> SecurityEvent:
        """Validate and persist one new event."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_security_event(event)
        if any(item.event_id == validated.event_id for item in state.events):
            raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)
        if validated.related_incident_id is not None:
            if self._find_incident(state, validated.related_incident_id) is None:
                raise IncidentRelationshipError(
                    "Cortana: Related incident was not found."
                )
        state.events.append(validated)
        self._persist(state)
        return validated

    def update_event(self, event: SecurityEvent) -> SecurityEvent:
        """Replace one existing event after validation."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_security_event(event)
        index = self._event_index(state, validated.event_id)
        if index is None:
            raise IncidentRelationshipError("Cortana: Event was not found.")
        if validated.related_incident_id is not None:
            if self._find_incident(state, validated.related_incident_id) is None:
                raise IncidentRelationshipError(
                    "Cortana: Related incident was not found."
                )
        state.events[index] = validated
        self._persist(state)
        return validated

    def list_incidents(self) -> list[SecurityIncident]:
        """Return a copy of saved incidents."""
        return list(self._ensure_loaded().incidents)

    def get_incident(self, incident_id: str) -> SecurityIncident | None:
        """Return one incident by exact ID match."""
        return self._find_incident(self._ensure_loaded(), incident_id)

    def add_incident(self, incident: SecurityIncident) -> SecurityIncident:
        """Validate and persist one new incident."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_security_incident(incident)
        if any(item.incident_id == validated.incident_id for item in state.incidents):
            raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)
        self._assert_ids_exist(state, event_ids=validated.event_ids)
        state.incidents.append(validated)
        self._persist(state)
        return validated

    def update_incident(self, incident: SecurityIncident) -> SecurityIncident:
        """Replace one existing incident after validation."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_security_incident(incident)
        index = self._incident_index(state, validated.incident_id)
        if index is None:
            raise IncidentRelationshipError("Cortana: Incident was not found.")
        self._assert_ids_exist(
            state,
            event_ids=validated.event_ids,
            evidence_ids=validated.evidence_ids,
            indicator_ids=validated.indicator_ids,
            note_ids=validated.note_ids,
        )
        state.incidents[index] = validated
        self._persist(state)
        return validated

    def link_event_to_incident(
        self,
        incident_id: str,
        event_id: str,
    ) -> tuple[SecurityIncident, SecurityEvent]:
        """Link an event to an incident with one coordinated persistence write."""
        state = self._clone_state(self._ensure_loaded())
        incident = self._find_incident(state, incident_id)
        event = self._find_event(state, event_id)
        if incident is None:
            raise IncidentRelationshipError("Cortana: Incident was not found.")
        if event is None:
            raise IncidentRelationshipError("Cortana: Event was not found.")
        if event_id in incident.event_ids:
            raise IncidentRelationshipError(
                "Cortana: That event is already linked to this incident."
            )
        if (
            event.related_incident_id is not None
            and event.related_incident_id != incident_id
        ):
            raise IncidentRelationshipError(
                "Cortana: That event is already linked to a different incident."
            )

        updated_incident = replace_security_incident(
            incident,
            event_ids=[*incident.event_ids, event_id],
        )
        updated_event = replace_security_event(
            event,
            related_incident_id=incident_id,
        )
        state.incidents[self._incident_index(state, incident_id) or 0] = updated_incident
        state.events[self._event_index(state, event_id) or 0] = updated_event
        self._persist(state)
        return updated_incident, updated_event

    def unlink_event_from_incident(
        self,
        incident_id: str,
        event_id: str,
    ) -> tuple[SecurityIncident, SecurityEvent]:
        """Unlink an event from an incident with one coordinated persistence write."""
        state = self._clone_state(self._ensure_loaded())
        incident = self._find_incident(state, incident_id)
        event = self._find_event(state, event_id)
        if incident is None:
            raise IncidentRelationshipError("Cortana: Incident was not found.")
        if event is None:
            raise IncidentRelationshipError("Cortana: Event was not found.")
        if event_id not in incident.event_ids:
            raise IncidentRelationshipError(
                "Cortana: That event is not linked to this incident."
            )

        updated_incident = replace_security_incident(
            incident,
            event_ids=[item for item in incident.event_ids if item != event_id],
        )
        updated_event = replace_security_event(
            event,
            clear_related_incident_id=True,
        )
        state.incidents[self._incident_index(state, incident_id) or 0] = updated_incident
        state.events[self._event_index(state, event_id) or 0] = updated_event
        self._persist(state)
        return updated_incident, updated_event

    def list_indicators(self) -> list[SecurityIndicator]:
        """Return a copy of saved indicators."""
        return list(self._ensure_loaded().indicators)

    def get_indicator(self, indicator_id: str) -> SecurityIndicator | None:
        """Return one indicator by exact ID match."""
        for indicator in self._ensure_loaded().indicators:
            if indicator.indicator_id == indicator_id:
                return indicator
        return None

    def find_indicator(
        self,
        indicator_type: str,
        normalized_value: str,
    ) -> SecurityIndicator | None:
        """Return a duplicate candidate by type and normalized value."""
        for indicator in self._ensure_loaded().indicators:
            if (
                indicator.indicator_type == indicator_type
                and indicator.normalized_value == normalized_value
            ):
                return indicator
        return None

    def add_indicator(self, indicator: SecurityIndicator) -> SecurityIndicator:
        """Validate and persist one new indicator, rejecting duplicates."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_security_indicator(indicator)
        if any(item.indicator_id == validated.indicator_id for item in state.indicators):
            raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)
        for existing in state.indicators:
            if (
                existing.indicator_type == validated.indicator_type
                and existing.normalized_value == validated.normalized_value
            ):
                raise IncidentRelationshipError(
                    "Cortana: An indicator with the same type and value already exists "
                    f"({existing.indicator_id})."
                )
        self._assert_ids_exist(
            state,
            event_ids=validated.related_event_ids,
            incident_ids=validated.related_incident_ids,
        )
        state.indicators.append(validated)
        self._persist(state)
        return validated

    def list_evidence(self) -> list[EvidenceRecord]:
        """Return a copy of saved evidence metadata."""
        return list(self._ensure_loaded().evidence)

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        """Return one evidence record by exact ID match."""
        for evidence in self._ensure_loaded().evidence:
            if evidence.evidence_id == evidence_id:
                return evidence
        return None

    def add_evidence(
        self,
        evidence: EvidenceRecord,
        custody_entries: list[ChainOfCustodyEntry],
    ) -> EvidenceRecord:
        """Persist evidence metadata and custody entries in one write."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_evidence_record(evidence)
        if any(item.evidence_id == validated.evidence_id for item in state.evidence):
            raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)

        validated_entries = [validate_custody_entry(entry) for entry in custody_entries]
        for entry in validated_entries:
            if entry.evidence_id != validated.evidence_id:
                raise IncidentRelationshipError(
                    "Cortana: Custody entry evidence ID mismatch."
                )
            if any(
                existing.entry_id == entry.entry_id for existing in state.custody_entries
            ):
                raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)

        linked_ids = list(validated.chain_of_custody_entry_ids)
        for entry in validated_entries:
            if entry.entry_id not in linked_ids:
                linked_ids.append(entry.entry_id)

        validated = replace_evidence_record(
            validated,
            chain_of_custody_entry_ids=linked_ids,
        )
        state.evidence.append(validated)
        state.custody_entries.extend(validated_entries)
        self._persist(state)
        return validated

    def append_custody_entry(
        self,
        evidence_id: str,
        entry: ChainOfCustodyEntry,
    ) -> ChainOfCustodyEntry:
        """Append one custody entry and update the evidence link list."""
        state = self._clone_state(self._ensure_loaded())
        evidence = self._find_evidence(state, evidence_id)
        if evidence is None:
            raise IncidentRelationshipError("Cortana: Evidence was not found.")

        validated = validate_custody_entry(entry)
        if validated.evidence_id != evidence_id:
            raise IncidentRelationshipError(
                "Cortana: Custody entry evidence ID mismatch."
            )
        if any(item.entry_id == validated.entry_id for item in state.custody_entries):
            raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)

        updated_evidence = replace_evidence_record(
            evidence,
            chain_of_custody_entry_ids=[
                *evidence.chain_of_custody_entry_ids,
                validated.entry_id,
            ],
        )
        index = self._evidence_index(state, evidence_id)
        if index is None:
            raise IncidentRelationshipError("Cortana: Evidence was not found.")
        state.evidence[index] = updated_evidence
        state.custody_entries.append(validated)
        self._persist(state)
        return validated

    def list_custody_entries(
        self,
        evidence_id: str | None = None,
    ) -> list[ChainOfCustodyEntry]:
        """Return custody entries, optionally filtered by evidence ID."""
        entries = self._ensure_loaded().custody_entries
        if evidence_id is None:
            return list(entries)
        return [entry for entry in entries if entry.evidence_id == evidence_id]

    def add_note(self, note: IncidentNote) -> IncidentNote:
        """Persist one note and link it onto the parent incident."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_incident_note(note)
        incident = self._find_incident(state, validated.incident_id)
        if incident is None:
            raise IncidentRelationshipError("Cortana: Incident was not found.")
        if any(item.note_id == validated.note_id for item in state.notes):
            raise IncidentStorageError(INCIDENT_LOAD_ERROR_MESSAGE)

        updated_incident = replace_security_incident(
            incident,
            note_ids=[*incident.note_ids, validated.note_id],
        )
        state.incidents[self._incident_index(state, validated.incident_id) or 0] = (
            updated_incident
        )
        state.notes.append(validated)
        self._persist(state)
        return validated

    def list_notes(self, incident_id: str) -> list[IncidentNote]:
        """Return notes for one incident in storage order."""
        return [
            note
            for note in self._ensure_loaded().notes
            if note.incident_id == incident_id
        ]

    def build_timeline(self, incident_id: str) -> list[IncidentTimelineEntry]:
        """Derive a local timeline for one incident without mutating storage."""
        state = self._ensure_loaded()
        incident = self._find_incident(state, incident_id)
        if incident is None:
            raise IncidentRelationshipError("Cortana: Incident was not found.")

        events = [
            event for event in state.events if event.event_id in incident.event_ids
        ]
        notes = [note for note in state.notes if note.note_id in incident.note_ids]
        evidence_ids = set(incident.evidence_ids)
        for event in events:
            evidence_ids.update(event.evidence_ids)
        custody_entries = [
            entry
            for entry in state.custody_entries
            if entry.evidence_id in evidence_ids
        ]
        return build_incident_timeline(
            events=events,
            notes=notes,
            custody_entries=custody_entries,
        )

    def event_count(self) -> int:
        """Return saved event count."""
        return len(self._ensure_loaded().events)

    def incident_count(self) -> int:
        """Return saved incident count."""
        return len(self._ensure_loaded().incidents)

    def indicator_count(self) -> int:
        """Return saved indicator count."""
        return len(self._ensure_loaded().indicators)

    def evidence_count(self) -> int:
        """Return saved evidence count."""
        return len(self._ensure_loaded().evidence)

    def _ensure_loaded(self) -> _RepositoryState:
        """Load repository state once, preserving any prior corruption error."""
        if self._load_error is not None:
            raise self._load_error
        if self._state is None:
            try:
                self._state = self._load_from_disk()
            except IncidentStorageError as error:
                self._load_error = error
                raise
        return self._state

    def _load_from_disk(self) -> _RepositoryState:
        """Load and validate the repository without modifying the file."""
        if not self._file_path.exists():
            return _RepositoryState()

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
        except OSError as error:
            logger.error(
                "Unable to read incident repository due to OS error type: %s",
                type(error).__name__,
            )
            raise IncidentStorageError from error

        if raw_text.strip() == "":
            logger.error("Incident repository file is empty.")
            raise IncidentStorageError

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("Incident repository contains malformed JSON.")
            raise IncidentStorageError from None

        return self._parse_payload(payload)

    def _parse_payload(self, payload: object) -> _RepositoryState:
        """Validate top-level schema and convert all record collections."""
        if not isinstance(payload, dict):
            logger.error("Incident repository top-level JSON structure is invalid.")
            raise IncidentStorageError

        version = payload.get("version")
        if version != INCIDENT_REPOSITORY_SCHEMA_VERSION:
            logger.error("Incident repository schema version is unsupported.")
            raise IncidentStorageError

        state = _RepositoryState(
            events=self._parse_events(payload.get("events")),
            incidents=self._parse_incidents(payload.get("incidents")),
            indicators=self._parse_indicators(payload.get("indicators")),
            evidence=self._parse_evidence(payload.get("evidence")),
            custody_entries=self._parse_custody(payload.get("custody_entries")),
            notes=self._parse_notes(payload.get("notes")),
        )
        self._validate_referential_integrity(state)
        return state

    def _parse_events(self, payload: object) -> list[SecurityEvent]:
        """Parse and validate the events collection."""
        if not isinstance(payload, list):
            raise IncidentStorageError
        records: list[SecurityEvent] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise IncidentStorageError
            try:
                record = validate_security_event(
                    SecurityEvent(
                        event_id=_require_string(item, "event_id"),
                        event_type=_require_string(item, "event_type"),
                        title=_require_string(item, "title"),
                        description=_require_string(item, "description"),
                        severity=_require_string(item, "severity"),  # type: ignore[arg-type]
                        status=_require_string(item, "status"),  # type: ignore[arg-type]
                        source=_require_string(item, "source"),
                        observed_at=_require_string(item, "observed_at"),
                        created_at=_require_string(item, "created_at"),
                        updated_at=_require_string(item, "updated_at"),
                        related_incident_id=_optional_string(
                            item,
                            "related_incident_id",
                        ),
                        tags=tuple(_require_string_list(item, "tags")),
                        indicator_ids=tuple(_require_string_list(item, "indicator_ids")),
                        evidence_ids=tuple(_require_string_list(item, "evidence_ids")),
                    )
                )
            except Exception:
                logger.error("Incident repository contains a malformed event record.")
                raise IncidentStorageError from None
            if record.event_id in seen:
                logger.error("Incident repository contains duplicate event IDs.")
                raise IncidentStorageError
            seen.add(record.event_id)
            records.append(record)
        return records

    def _parse_incidents(self, payload: object) -> list[SecurityIncident]:
        """Parse and validate the incidents collection."""
        if not isinstance(payload, list):
            raise IncidentStorageError
        records: list[SecurityIncident] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise IncidentStorageError
            try:
                record = validate_security_incident(
                    SecurityIncident(
                        incident_id=_require_string(item, "incident_id"),
                        title=_require_string(item, "title"),
                        summary=_require_string(item, "summary"),
                        severity=_require_string(item, "severity"),  # type: ignore[arg-type]
                        status=_require_string(item, "status"),  # type: ignore[arg-type]
                        created_at=_require_string(item, "created_at"),
                        updated_at=_require_string(item, "updated_at"),
                        opened_at=_require_string(item, "opened_at"),
                        closed_at=_optional_string(item, "closed_at"),
                        event_ids=tuple(_require_string_list(item, "event_ids")),
                        evidence_ids=tuple(_require_string_list(item, "evidence_ids")),
                        indicator_ids=tuple(
                            _require_string_list(item, "indicator_ids")
                        ),
                        note_ids=tuple(_require_string_list(item, "note_ids")),
                        tags=tuple(_require_string_list(item, "tags")),
                    )
                )
            except Exception:
                logger.error("Incident repository contains a malformed incident record.")
                raise IncidentStorageError from None
            if record.incident_id in seen:
                logger.error("Incident repository contains duplicate incident IDs.")
                raise IncidentStorageError
            seen.add(record.incident_id)
            records.append(record)
        return records

    def _parse_indicators(self, payload: object) -> list[SecurityIndicator]:
        """Parse and validate the indicators collection."""
        if not isinstance(payload, list):
            raise IncidentStorageError
        records: list[SecurityIndicator] = []
        seen: set[str] = set()
        seen_values: set[tuple[str, str]] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise IncidentStorageError
            try:
                record = validate_security_indicator(
                    SecurityIndicator(
                        indicator_id=_require_string(item, "indicator_id"),
                        indicator_type=_require_string(item, "indicator_type"),  # type: ignore[arg-type]
                        normalized_value=_require_string(item, "normalized_value"),
                        original_value=_require_string(item, "original_value"),
                        confidence=_require_int(item, "confidence"),
                        first_seen_at=_require_string(item, "first_seen_at"),
                        last_seen_at=_require_string(item, "last_seen_at"),
                        created_at=_require_string(item, "created_at"),
                        tags=tuple(_require_string_list(item, "tags")),
                        related_event_ids=tuple(
                            _require_string_list(item, "related_event_ids")
                        ),
                        related_incident_ids=tuple(
                            _require_string_list(item, "related_incident_ids")
                        ),
                        notes=_optional_string(item, "notes"),
                    )
                )
            except Exception:
                logger.error(
                    "Incident repository contains a malformed indicator record."
                )
                raise IncidentStorageError from None
            value_key = (record.indicator_type, record.normalized_value)
            if record.indicator_id in seen or value_key in seen_values:
                logger.error("Incident repository contains duplicate indicators.")
                raise IncidentStorageError
            seen.add(record.indicator_id)
            seen_values.add(value_key)
            records.append(record)
        return records

    def _parse_evidence(self, payload: object) -> list[EvidenceRecord]:
        """Parse and validate the evidence collection."""
        if not isinstance(payload, list):
            raise IncidentStorageError
        records: list[EvidenceRecord] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise IncidentStorageError
            try:
                record = validate_evidence_record(
                    EvidenceRecord(
                        evidence_id=_require_string(item, "evidence_id"),
                        evidence_type=_require_string(item, "evidence_type"),  # type: ignore[arg-type]
                        title=_require_string(item, "title"),
                        description=_require_string(item, "description"),
                        original_filename=_optional_string(item, "original_filename"),
                        sha256_hash=_require_string(item, "sha256_hash"),
                        source_size_bytes=_require_int(item, "source_size_bytes"),
                        collected_at=_require_string(item, "collected_at"),
                        recorded_at=_require_string(item, "recorded_at"),
                        collector=_require_string(item, "collector"),
                        storage_status=_require_string(item, "storage_status"),  # type: ignore[arg-type]
                        related_event_ids=tuple(
                            _require_string_list(item, "related_event_ids")
                        ),
                        related_incident_ids=tuple(
                            _require_string_list(item, "related_incident_ids")
                        ),
                        chain_of_custody_entry_ids=tuple(
                            _require_string_list(item, "chain_of_custody_entry_ids")
                        ),
                        tags=tuple(_require_string_list(item, "tags")),
                    )
                )
            except Exception:
                logger.error("Incident repository contains a malformed evidence record.")
                raise IncidentStorageError from None
            if record.evidence_id in seen:
                logger.error("Incident repository contains duplicate evidence IDs.")
                raise IncidentStorageError
            seen.add(record.evidence_id)
            records.append(record)
        return records

    def _parse_custody(self, payload: object) -> list[ChainOfCustodyEntry]:
        """Parse and validate the custody entries collection."""
        if not isinstance(payload, list):
            raise IncidentStorageError
        records: list[ChainOfCustodyEntry] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise IncidentStorageError
            try:
                record = validate_custody_entry(
                    ChainOfCustodyEntry(
                        entry_id=_require_string(item, "entry_id"),
                        evidence_id=_require_string(item, "evidence_id"),
                        action=_require_string(item, "action"),  # type: ignore[arg-type]
                        actor=_require_string(item, "actor"),
                        timestamp=_require_string(item, "timestamp"),
                        reason=_require_string(item, "reason"),
                        previous_hash=_optional_string(item, "previous_hash"),
                        resulting_hash=_optional_string(item, "resulting_hash"),
                        notes=_optional_string(item, "notes"),
                    )
                )
            except Exception:
                logger.error("Incident repository contains a malformed custody record.")
                raise IncidentStorageError from None
            if record.entry_id in seen:
                logger.error("Incident repository contains duplicate custody IDs.")
                raise IncidentStorageError
            seen.add(record.entry_id)
            records.append(record)
        return records

    def _parse_notes(self, payload: object) -> list[IncidentNote]:
        """Parse and validate the notes collection."""
        if not isinstance(payload, list):
            raise IncidentStorageError
        records: list[IncidentNote] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise IncidentStorageError
            try:
                record = validate_incident_note(
                    IncidentNote(
                        note_id=_require_string(item, "note_id"),
                        incident_id=_require_string(item, "incident_id"),
                        author=_require_string(item, "author"),
                        text=_require_string(item, "text"),
                        created_at=_require_string(item, "created_at"),
                        updated_at=_require_string(item, "updated_at"),
                        note_type=_require_string(item, "note_type"),  # type: ignore[arg-type]
                        tags=tuple(_require_string_list(item, "tags")),
                    )
                )
            except Exception:
                logger.error("Incident repository contains a malformed note record.")
                raise IncidentStorageError from None
            if record.note_id in seen:
                logger.error("Incident repository contains duplicate note IDs.")
                raise IncidentStorageError
            seen.add(record.note_id)
            records.append(record)
        return records

    def _validate_referential_integrity(self, state: _RepositoryState) -> None:
        """Fail closed when stored relationship IDs do not resolve."""
        event_ids = {event.event_id for event in state.events}
        incident_ids = {incident.incident_id for incident in state.incidents}
        indicator_ids = {indicator.indicator_id for indicator in state.indicators}
        evidence_ids = {evidence.evidence_id for evidence in state.evidence}
        custody_ids = {entry.entry_id for entry in state.custody_entries}
        note_ids = {note.note_id for note in state.notes}

        for event in state.events:
            if (
                event.related_incident_id is not None
                and event.related_incident_id not in incident_ids
            ):
                logger.error("Incident repository has dangling event incident reference.")
                raise IncidentStorageError
            if not set(event.indicator_ids).issubset(indicator_ids):
                logger.error("Incident repository has dangling event indicator reference.")
                raise IncidentStorageError
            if not set(event.evidence_ids).issubset(evidence_ids):
                logger.error("Incident repository has dangling event evidence reference.")
                raise IncidentStorageError

        for incident in state.incidents:
            if not set(incident.event_ids).issubset(event_ids):
                logger.error("Incident repository has dangling incident event reference.")
                raise IncidentStorageError
            if not set(incident.evidence_ids).issubset(evidence_ids):
                logger.error(
                    "Incident repository has dangling incident evidence reference."
                )
                raise IncidentStorageError
            if not set(incident.indicator_ids).issubset(indicator_ids):
                logger.error(
                    "Incident repository has dangling incident indicator reference."
                )
                raise IncidentStorageError
            if not set(incident.note_ids).issubset(note_ids):
                logger.error("Incident repository has dangling incident note reference.")
                raise IncidentStorageError

        for indicator in state.indicators:
            if not set(indicator.related_event_ids).issubset(event_ids):
                logger.error(
                    "Incident repository has dangling indicator event reference."
                )
                raise IncidentStorageError
            if not set(indicator.related_incident_ids).issubset(incident_ids):
                logger.error(
                    "Incident repository has dangling indicator incident reference."
                )
                raise IncidentStorageError

        for evidence in state.evidence:
            if not set(evidence.related_event_ids).issubset(event_ids):
                logger.error("Incident repository has dangling evidence event reference.")
                raise IncidentStorageError
            if not set(evidence.related_incident_ids).issubset(incident_ids):
                logger.error(
                    "Incident repository has dangling evidence incident reference."
                )
                raise IncidentStorageError
            if not set(evidence.chain_of_custody_entry_ids).issubset(custody_ids):
                logger.error(
                    "Incident repository has dangling evidence custody reference."
                )
                raise IncidentStorageError

        for entry in state.custody_entries:
            if entry.evidence_id not in evidence_ids:
                logger.error("Incident repository has dangling custody evidence reference.")
                raise IncidentStorageError

        for note in state.notes:
            if note.incident_id not in incident_ids:
                logger.error("Incident repository has dangling note incident reference.")
                raise IncidentStorageError

        for incident in state.incidents:
            for linked_event_id in incident.event_ids:
                linked_event = self._find_event(state, linked_event_id)
                if (
                    linked_event is None
                    or linked_event.related_incident_id != incident.incident_id
                ):
                    logger.error(
                        "Incident repository has inconsistent event-incident link."
                    )
                    raise IncidentStorageError

    def _persist(self, state: _RepositoryState) -> None:
        """Atomically write the coordinated repository as UTF-8 JSON."""
        self._validate_referential_integrity(state)
        payload = {
            "version": INCIDENT_REPOSITORY_SCHEMA_VERSION,
            "events": [_serialize_event(event) for event in state.events],
            "incidents": [
                _serialize_incident(incident) for incident in state.incidents
            ],
            "indicators": [
                _serialize_indicator(indicator) for indicator in state.indicators
            ],
            "evidence": [_serialize_evidence(evidence) for evidence in state.evidence],
            "custody_entries": [
                _serialize_custody(entry) for entry in state.custody_entries
            ],
            "notes": [_serialize_note(note) for note in state.notes],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._state = self._clone_state(state)

    def _atomic_write(self, content: str) -> None:
        """Write JSON to a temporary file, flush/fsync, then replace the target."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".incidents-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._file_path)
            temporary_path = None
        except OSError as error:
            logger.error(
                "Unable to write incident repository due to OS error type: %s",
                type(error).__name__,
            )
            raise IncidentStorageError(INCIDENT_SAVE_ERROR_MESSAGE) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _clone_state(self, state: _RepositoryState) -> _RepositoryState:
        """Return a shallow copy of repository collections for transactional edits."""
        return _RepositoryState(
            events=list(state.events),
            incidents=list(state.incidents),
            indicators=list(state.indicators),
            evidence=list(state.evidence),
            custody_entries=list(state.custody_entries),
            notes=list(state.notes),
        )

    def _assert_ids_exist(
        self,
        state: _RepositoryState,
        *,
        event_ids: tuple[str, ...] | list[str] = (),
        incident_ids: tuple[str, ...] | list[str] = (),
        evidence_ids: tuple[str, ...] | list[str] = (),
        indicator_ids: tuple[str, ...] | list[str] = (),
        note_ids: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Reject relationship writes that reference missing records."""
        known_events = {event.event_id for event in state.events}
        known_incidents = {incident.incident_id for incident in state.incidents}
        known_evidence = {evidence.evidence_id for evidence in state.evidence}
        known_indicators = {indicator.indicator_id for indicator in state.indicators}
        known_notes = {note.note_id for note in state.notes}

        if not set(event_ids).issubset(known_events):
            raise IncidentRelationshipError("Cortana: Referenced event was not found.")
        if not set(incident_ids).issubset(known_incidents):
            raise IncidentRelationshipError(
                "Cortana: Referenced incident was not found."
            )
        if not set(evidence_ids).issubset(known_evidence):
            raise IncidentRelationshipError(
                "Cortana: Referenced evidence was not found."
            )
        if not set(indicator_ids).issubset(known_indicators):
            raise IncidentRelationshipError(
                "Cortana: Referenced indicator was not found."
            )
        if not set(note_ids).issubset(known_notes):
            raise IncidentRelationshipError("Cortana: Referenced note was not found.")

    def _find_event(
        self,
        state: _RepositoryState,
        event_id: str,
    ) -> SecurityEvent | None:
        for event in state.events:
            if event.event_id == event_id:
                return event
        return None

    def _find_incident(
        self,
        state: _RepositoryState,
        incident_id: str,
    ) -> SecurityIncident | None:
        for incident in state.incidents:
            if incident.incident_id == incident_id:
                return incident
        return None

    def _find_evidence(
        self,
        state: _RepositoryState,
        evidence_id: str,
    ) -> EvidenceRecord | None:
        for evidence in state.evidence:
            if evidence.evidence_id == evidence_id:
                return evidence
        return None

    def _event_index(self, state: _RepositoryState, event_id: str) -> int | None:
        for index, event in enumerate(state.events):
            if event.event_id == event_id:
                return index
        return None

    def _incident_index(self, state: _RepositoryState, incident_id: str) -> int | None:
        for index, incident in enumerate(state.incidents):
            if incident.incident_id == incident_id:
                return index
        return None

    def _evidence_index(self, state: _RepositoryState, evidence_id: str) -> int | None:
        for index, evidence in enumerate(state.evidence):
            if evidence.evidence_id == evidence_id:
                return index
        return None


def _serialize_event(event: SecurityEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "title": event.title,
        "description": event.description,
        "severity": event.severity,
        "status": event.status,
        "source": event.source,
        "observed_at": event.observed_at,
        "created_at": event.created_at,
        "updated_at": event.updated_at,
        "related_incident_id": event.related_incident_id,
        "tags": list(event.tags),
        "indicator_ids": list(event.indicator_ids),
        "evidence_ids": list(event.evidence_ids),
    }


def _serialize_incident(incident: SecurityIncident) -> dict[str, object]:
    return {
        "incident_id": incident.incident_id,
        "title": incident.title,
        "summary": incident.summary,
        "severity": incident.severity,
        "status": incident.status,
        "created_at": incident.created_at,
        "updated_at": incident.updated_at,
        "opened_at": incident.opened_at,
        "closed_at": incident.closed_at,
        "event_ids": list(incident.event_ids),
        "evidence_ids": list(incident.evidence_ids),
        "indicator_ids": list(incident.indicator_ids),
        "note_ids": list(incident.note_ids),
        "tags": list(incident.tags),
    }


def _serialize_indicator(indicator: SecurityIndicator) -> dict[str, object]:
    return {
        "indicator_id": indicator.indicator_id,
        "indicator_type": indicator.indicator_type,
        "normalized_value": indicator.normalized_value,
        "original_value": indicator.original_value,
        "confidence": indicator.confidence,
        "first_seen_at": indicator.first_seen_at,
        "last_seen_at": indicator.last_seen_at,
        "created_at": indicator.created_at,
        "tags": list(indicator.tags),
        "related_event_ids": list(indicator.related_event_ids),
        "related_incident_ids": list(indicator.related_incident_ids),
        "notes": indicator.notes,
    }


def _serialize_evidence(evidence: EvidenceRecord) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type,
        "title": evidence.title,
        "description": evidence.description,
        "original_filename": evidence.original_filename,
        "sha256_hash": evidence.sha256_hash,
        "source_size_bytes": evidence.source_size_bytes,
        "collected_at": evidence.collected_at,
        "recorded_at": evidence.recorded_at,
        "collector": evidence.collector,
        "storage_status": evidence.storage_status,
        "related_event_ids": list(evidence.related_event_ids),
        "related_incident_ids": list(evidence.related_incident_ids),
        "chain_of_custody_entry_ids": list(evidence.chain_of_custody_entry_ids),
        "tags": list(evidence.tags),
    }


def _serialize_custody(entry: ChainOfCustodyEntry) -> dict[str, object]:
    return {
        "entry_id": entry.entry_id,
        "evidence_id": entry.evidence_id,
        "action": entry.action,
        "actor": entry.actor,
        "timestamp": entry.timestamp,
        "reason": entry.reason,
        "previous_hash": entry.previous_hash,
        "resulting_hash": entry.resulting_hash,
        "notes": entry.notes,
    }


def _serialize_note(note: IncidentNote) -> dict[str, object]:
    return {
        "note_id": note.note_id,
        "incident_id": note.incident_id,
        "author": note.author,
        "text": note.text,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
        "note_type": note.note_type,
        "tags": list(note.tags),
    }


def _require_string(item: dict[object, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise IncidentStorageError
    return value


def _optional_string(item: dict[object, object], key: str) -> str | None:
    if key not in item or item.get(key) is None:
        return None
    value = item.get(key)
    if not isinstance(value, str):
        raise IncidentStorageError
    return value


def _require_int(item: dict[object, object], key: str) -> int:
    value = item.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise IncidentStorageError
    return value


def _require_string_list(item: dict[object, object], key: str) -> list[str]:
    value = item.get(key)
    if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
        raise IncidentStorageError
    return list(value)
