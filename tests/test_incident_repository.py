"""Tests for Milestone 8 coordinated incident repository persistence."""

import logging
from pathlib import Path

import pytest

from src.config import INCIDENT_REPOSITORY_SCHEMA_VERSION, PROJECT_ROOT
from src.incident_repository import (
    IncidentRelationshipError,
    IncidentStorageError,
    JsonIncidentRepository,
)
from src.security_event import create_security_event
from src.security_incident import create_security_incident
from src.security_indicator import create_security_indicator
from src.security_note import create_incident_note
from tests.security_helpers import incident_repository


def test_round_trip_and_fresh_instance_reload(tmp_path: Path) -> None:
    """Saved records should reload from disk in a new repository instance."""
    store = incident_repository(tmp_path)
    event = store.add_event(
        create_security_event(
            event_type="alert",
            title="Event",
            description="Desc",
            severity="low",
        )
    )
    incident = store.add_incident(
        create_security_incident(
            title="Incident",
            summary="Summary",
            severity="medium",
        )
    )
    store.link_event_to_incident(incident.incident_id, event.event_id)

    reloaded = incident_repository(tmp_path)
    assert reloaded.event_count() == 1
    assert reloaded.incident_count() == 1
    linked_event = reloaded.get_event(event.event_id)
    linked_incident = reloaded.get_incident(incident.incident_id)
    assert linked_event is not None
    assert linked_incident is not None
    assert linked_event.related_incident_id == incident.incident_id
    assert event.event_id in linked_incident.event_ids


def test_atomic_write_and_temp_cleanup(tmp_path: Path) -> None:
    """Persisted output should be complete JSON with no leftover temps."""
    store = incident_repository(tmp_path)
    store.add_event(
        create_security_event(
            event_type="alert",
            title="Atomic",
            description="Desc",
            severity="low",
        )
    )

    raw = store.file_path.read_text(encoding="utf-8")
    leftover = list(tmp_path.glob(".incidents-*.tmp"))
    assert f'"version": {INCIDENT_REPOSITORY_SCHEMA_VERSION}' in raw
    assert leftover == []


def test_malformed_and_empty_files_fail_closed(tmp_path: Path) -> None:
    """Malformed or empty repository files should fail closed and stay unchanged."""
    path = tmp_path / "incidents.json"
    original = "{not-json"
    path.write_text(original, encoding="utf-8")
    store = JsonIncidentRepository(path)

    with pytest.raises(IncidentStorageError):
        store.list_events()
    assert path.read_text(encoding="utf-8") == original

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("", encoding="utf-8")
    empty_store = JsonIncidentRepository(empty_path)
    with pytest.raises(IncidentStorageError):
        empty_store.list_events()
    assert empty_path.read_text(encoding="utf-8") == ""


def test_duplicate_indicator_rejected(tmp_path: Path) -> None:
    """Duplicate type+normalized-value indicators should be rejected."""
    store = incident_repository(tmp_path)
    store.add_indicator(
        create_security_indicator(
            indicator_type="domain",
            value="Example.COM",
            confidence=50,
        )
    )

    with pytest.raises(IncidentRelationshipError):
        store.add_indicator(
            create_security_indicator(
                indicator_type="domain",
                value="example.com",
                confidence=70,
            )
        )


def test_inconsistent_references_fail_closed(tmp_path: Path) -> None:
    """Dangling references in stored JSON should fail closed without repair."""
    path = tmp_path / "incidents.json"
    path.write_text(
        """
{
  "version": 1,
  "events": [],
  "incidents": [
    {
      "incident_id": "11111111-1111-1111-1111-111111111111",
      "title": "Broken",
      "summary": "Missing event link",
      "severity": "low",
      "status": "open",
      "created_at": "2026-01-01T00:00:00.000000Z",
      "updated_at": "2026-01-01T00:00:00.000000Z",
      "opened_at": "2026-01-01T00:00:00.000000Z",
      "closed_at": null,
      "event_ids": ["22222222-2222-2222-2222-222222222222"],
      "evidence_ids": [],
      "indicator_ids": [],
      "note_ids": [],
      "tags": []
    }
  ],
  "indicators": [],
  "evidence": [],
  "custody_entries": [],
  "notes": []
}
""".strip(),
        encoding="utf-8",
    )
    store = JsonIncidentRepository(path)
    with pytest.raises(IncidentStorageError):
        store.list_incidents()
    assert "Broken" in path.read_text(encoding="utf-8")


def test_link_unlink_are_bidirectional(tmp_path: Path) -> None:
    """Link and unlink should update both sides in one persistence operation."""
    store = incident_repository(tmp_path)
    event = store.add_event(
        create_security_event(
            event_type="alert",
            title="Event",
            description="Desc",
            severity="low",
        )
    )
    incident = store.add_incident(
        create_security_incident(
            title="Incident",
            summary="Summary",
            severity="low",
        )
    )

    store.link_event_to_incident(incident.incident_id, event.event_id)
    with pytest.raises(IncidentRelationshipError):
        store.link_event_to_incident(incident.incident_id, event.event_id)

    store.unlink_event_from_incident(incident.incident_id, event.event_id)
    reloaded_event = store.get_event(event.event_id)
    reloaded_incident = store.get_incident(incident.incident_id)
    assert reloaded_event is not None
    assert reloaded_incident is not None
    assert reloaded_event.related_incident_id is None
    assert reloaded_event.event_id not in reloaded_incident.event_ids


def test_note_links_to_incident_without_logging_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Notes should link to incidents and never log note text."""
    store = incident_repository(tmp_path)
    incident = store.add_incident(
        create_security_incident(
            title="Incident",
            summary="Summary",
            severity="low",
        )
    )
    marker = "NOTE_SECRET_MARKER_XYZ"
    with caplog.at_level(logging.INFO, logger="ProjectCortana"):
        note = store.add_note(
            create_incident_note(
                incident_id=incident.incident_id,
                author="analyst",
                text=marker,
                note_type="observation",
            )
        )

    reloaded = store.get_incident(incident.incident_id)
    assert reloaded is not None
    assert note.note_id in reloaded.note_ids
    assert marker not in caplog.text


def test_default_path_helper_outside_project() -> None:
    """Default repository path helper should stay outside the project tree."""
    from src.config import get_default_incident_repository_file_path

    path = get_default_incident_repository_file_path()
    assert PROJECT_ROOT not in path.parents


def test_single_instance_limitation_is_documented() -> None:
    """README should document the single-instance repository limitation."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "one application instance" in readme.lower() or "one-instance" in readme.lower()
    assert "last-writer-wins" in readme.lower() or "last writer wins" in readme.lower()
