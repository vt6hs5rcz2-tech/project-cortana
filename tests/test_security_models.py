"""Tests for Milestone 8 immutable security models."""

from dataclasses import FrozenInstanceError
from datetime import datetime
from hashlib import sha256

import pytest

from src.config import (
    MAX_EVENT_DESCRIPTION_LENGTH,
    MAX_EVENT_TITLE_LENGTH,
    MAX_NOTE_TEXT_LENGTH,
)
from src.security_common import (
    BlankSecurityFieldError,
    InvalidSecurityEnumError,
    InvalidSecurityIdError,
    SecurityFieldTooLongError,
    SecurityValidationError,
    normalize_tags,
)
from src.security_custody import create_custody_entry
from src.security_event import (
    create_security_event,
    replace_security_event,
    update_security_event_status,
)
from src.security_evidence import create_evidence_record
from src.security_incident import (
    create_security_incident,
    update_security_incident_status,
)
from src.security_indicator import create_security_indicator, normalize_indicator_value
from src.security_note import create_incident_note, update_incident_note


def test_security_event_immutability_and_valid_construction() -> None:
    """Events should be immutable validated UUID-backed UTC records."""
    event = create_security_event(
        event_type="alert",
        title=" Suspicious login ",
        description=" User reported phishing ",
        severity=" High ",
    )

    assert len(event.event_id) == 36
    assert event.title == "Suspicious login"
    assert event.description == "User reported phishing"
    assert event.severity == "high"
    assert event.status == "new"
    assert event.created_at.endswith("Z")
    parsed = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
    utc_offset = parsed.utcoffset()
    assert utc_offset is not None
    assert utc_offset.total_seconds() == 0

    with pytest.raises(FrozenInstanceError):
        event.title = "changed"  # type: ignore[misc]


def test_security_event_rejects_blank_and_overlong_fields() -> None:
    """Blank and oversized event fields should fail closed."""
    with pytest.raises(BlankSecurityFieldError):
        create_security_event(
            event_type="alert",
            title="   ",
            description="desc",
            severity="low",
        )

    with pytest.raises(SecurityFieldTooLongError):
        create_security_event(
            event_type="alert",
            title="x" * (MAX_EVENT_TITLE_LENGTH + 1),
            description="desc",
            severity="low",
        )

    with pytest.raises(SecurityFieldTooLongError):
        create_security_event(
            event_type="alert",
            title="title",
            description="x" * (MAX_EVENT_DESCRIPTION_LENGTH + 1),
            severity="low",
        )


def test_security_event_status_update_preserves_identity() -> None:
    """Safe updates should create replacement records with the same ID."""
    event = create_security_event(
        event_type="alert",
        title="title",
        description="desc",
        severity="medium",
    )
    updated = update_security_event_status(event, "investigating")

    assert updated.event_id == event.event_id
    assert updated.created_at == event.created_at
    assert updated.status == "investigating"
    assert updated.updated_at >= event.updated_at


def test_tag_deduplication_preserves_order() -> None:
    """Tags should keep first-seen order and drop case-insensitive duplicates."""
    assert normalize_tags(["Alpha", "beta", "alpha", "Gamma"]) == (
        "Alpha",
        "beta",
        "Gamma",
    )


def test_incident_closed_timestamp_rules() -> None:
    """Closed timestamps exist only for resolved/closed and clear on reopen."""
    incident = create_security_incident(
        title="Incident",
        summary="Summary",
        severity="high",
    )
    closed = update_security_incident_status(incident, "closed")
    assert closed.closed_at is not None

    reopened = update_security_incident_status(closed, "investigating")
    assert reopened.closed_at is None
    assert reopened.status == "investigating"

    with pytest.raises(InvalidSecurityEnumError):
        update_security_incident_status(incident, "not-a-status")


def test_indicator_normalization_and_hash_validation() -> None:
    """Indicators should normalize safely without network lookups."""
    assert normalize_indicator_value("ipv4", "192.168.1.10") == "192.168.1.10"
    assert normalize_indicator_value("domain", "Example.COM.") == "example.com"
    assert normalize_indicator_value("email", "User@Example.COM") == "User@example.com"

    digest = sha256(b"abc").hexdigest()
    indicator = create_security_indicator(
        indicator_type="sha256",
        value=digest.upper(),
        confidence=80,
    )
    assert indicator.normalized_value == digest
    assert indicator.original_value == digest.upper()

    with pytest.raises(SecurityValidationError):
        create_security_indicator(
            indicator_type="md5",
            value="not-a-hash",
            confidence=10,
        )

    with pytest.raises(SecurityValidationError):
        create_security_indicator(
            indicator_type="ipv4",
            value="999.1.1.1",
            confidence=10,
        )

    with pytest.raises(SecurityValidationError):
        create_security_indicator(
            indicator_type="generic",
            value="value",
            confidence=101,
        )


def test_evidence_and_custody_models() -> None:
    """Evidence and custody entries should validate hashes and remain immutable."""
    digest = sha256(b"evidence").hexdigest()
    evidence = create_evidence_record(
        evidence_type="file",
        title="Capture",
        description="Local copy",
        sha256_hash=digest,
        source_size_bytes=8,
        collector="analyst",
        storage_status="copied",
        original_filename="note.txt",
    )
    assert evidence.storage_status == "copied"

    with pytest.raises(FrozenInstanceError):
        evidence.title = "x"  # type: ignore[misc]

    entry = create_custody_entry(
        evidence_id=evidence.evidence_id,
        action="registered",
        actor="analyst",
        reason="registered locally",
        resulting_hash=digest,
    )
    assert entry.action == "registered"
    with pytest.raises(FrozenInstanceError):
        entry.reason = "changed"  # type: ignore[misc]


def test_incident_note_edit_preserves_created_at() -> None:
    """Editing a note should preserve created_at and never allow blank text."""
    incident = create_security_incident(
        title="Incident",
        summary="Summary",
        severity="low",
    )
    note = create_incident_note(
        incident_id=incident.incident_id,
        author="analyst",
        text="Initial observation",
        note_type="observation",
    )
    updated = update_incident_note(note, text="Updated observation")

    assert updated.note_id == note.note_id
    assert updated.created_at == note.created_at
    assert updated.text == "Updated observation"

    with pytest.raises(SecurityFieldTooLongError):
        create_incident_note(
            incident_id=incident.incident_id,
            author="analyst",
            text="x" * (MAX_NOTE_TEXT_LENGTH + 1),
            note_type="observation",
        )


def test_invalid_uuid_rejected() -> None:
    """Malformed UUID identifiers should be rejected."""
    with pytest.raises(InvalidSecurityIdError):
        replace_security_event(
            create_security_event(
                event_type="alert",
                title="t",
                description="d",
                severity="low",
            ),
            related_incident_id="not-a-uuid",
        )
