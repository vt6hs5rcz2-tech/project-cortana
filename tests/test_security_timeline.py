"""Tests for derived incident timelines."""

from pathlib import Path

from src.security_custody import create_custody_entry
from src.security_event import create_security_event
from src.security_incident import create_security_incident
from src.security_note import create_incident_note
from src.security_timeline import build_incident_timeline
from tests.security_helpers import incident_repository


def test_timeline_sorting_tie_break_and_no_persistence(tmp_path: Path) -> None:
    """Timelines should sort deterministically and never become source-of-truth storage."""
    repo = incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary",
            severity="low",
        )
    )
    event = repo.add_event(
        create_security_event(
            event_type="alert",
            title="Event A",
            description="Body",
            severity="low",
            observed_at="2026-01-01T00:00:00.000000Z",
        )
    )
    repo.link_event_to_incident(incident.incident_id, event.event_id)
    note = repo.add_note(
        create_incident_note(
            incident_id=incident.incident_id,
            author="analyst",
            text="Note text",
            note_type="observation",
        )
    )

    # Force note created_at equal to event for tie-break coverage via direct builder.
    entries = build_incident_timeline(
        events=[event],
        notes=[
            type(note)(
                note_id=note.note_id,
                incident_id=note.incident_id,
                author=note.author,
                text=note.text,
                created_at=event.observed_at,
                updated_at=note.updated_at,
                note_type=note.note_type,
                tags=note.tags,
            )
        ],
        custody_entries=[
            create_custody_entry(
                evidence_id="99999999-9999-9999-9999-999999999999",
                action="registered",
                actor="analyst",
                reason="synthetic",
                timestamp=event.observed_at,
            )
        ],
    )

    assert [entry.entry_type for entry in entries] == ["event", "custody", "note"]
    assert entries[0].entry_id == event.event_id

    linked_event = repo.get_event(event.event_id)
    assert linked_event is not None
    before = repo.file_path.read_text(encoding="utf-8")
    derived = repo.build_timeline(incident.incident_id)
    after = repo.file_path.read_text(encoding="utf-8")
    assert before == after
    assert any(item.entry_type == "event" for item in derived)
    assert any(item.entry_type == "note" for item in derived)
    assert repo.get_event(event.event_id) == linked_event
