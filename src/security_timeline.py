"""Derived incident timeline helpers for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass

from src.security_common import TimelineEntryType
from src.security_custody import ChainOfCustodyEntry
from src.security_event import SecurityEvent
from src.security_note import IncidentNote


@dataclass(frozen=True)
class IncidentTimelineEntry:
    """Derived timeline entry; never persisted as a source of truth."""

    timestamp: str
    entry_type: TimelineEntryType
    entry_id: str
    summary: str
    source_record_id: str


def build_incident_timeline(
    *,
    events: list[SecurityEvent] | tuple[SecurityEvent, ...],
    notes: list[IncidentNote] | tuple[IncidentNote, ...],
    custody_entries: list[ChainOfCustodyEntry] | tuple[ChainOfCustodyEntry, ...],
) -> list[IncidentTimelineEntry]:
    """Build a chronologically sorted timeline from related local records."""
    entries: list[IncidentTimelineEntry] = []

    for event in events:
        entries.append(
            IncidentTimelineEntry(
                timestamp=event.observed_at,
                entry_type="event",
                entry_id=event.event_id,
                summary=f"Event ({event.severity}/{event.status}): {event.title}",
                source_record_id=event.event_id,
            )
        )

    for note in notes:
        entries.append(
            IncidentTimelineEntry(
                timestamp=note.created_at,
                entry_type="note",
                entry_id=note.note_id,
                summary=f"Note ({note.note_type}) by {note.author}",
                source_record_id=note.note_id,
            )
        )

    for custody in custody_entries:
        entries.append(
            IncidentTimelineEntry(
                timestamp=custody.timestamp,
                entry_type="custody",
                entry_id=custody.entry_id,
                summary=(
                    f"Custody {custody.action} for evidence {custody.evidence_id}"
                ),
                source_record_id=custody.entry_id,
            )
        )

    entries.sort(
        key=lambda item: (
            item.timestamp,
            _entry_type_rank(item.entry_type),
            item.entry_id,
        )
    )
    return entries


def _entry_type_rank(entry_type: str) -> int:
    """Deterministic tie-break rank for equal timestamps."""
    order = {"event": 0, "custody": 1, "note": 2}
    return order.get(entry_type, 99)
