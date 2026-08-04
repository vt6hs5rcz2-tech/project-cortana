"""Immutable incident analyst note model for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.config import MAX_CUSTODY_ACTOR_LENGTH, MAX_NOTE_TEXT_LENGTH
from src.security_common import (
    NOTE_TYPES,
    NoteType,
    normalize_tags,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_security_id,
    validate_utc_timestamp,
)


@dataclass(frozen=True)
class IncidentNote:
    """Immutable analyst note attached to an incident by an explicit user command."""

    note_id: str
    incident_id: str
    author: str
    text: str
    created_at: str
    updated_at: str
    note_type: NoteType
    tags: tuple[str, ...]


def create_incident_note(
    *,
    incident_id: str,
    author: str,
    text: str,
    note_type: str,
    tags: list[str] | tuple[str, ...] | None = None,
) -> IncidentNote:
    """Create a validated immutable incident note."""
    now = utc_timestamp()
    normalized_type = validate_controlled_value(
        note_type,
        field_name="note type",
        allowed=NOTE_TYPES,
    )
    return validate_incident_note(
        IncidentNote(
            note_id=str(uuid4()),
            incident_id=incident_id,
            author=author,
            text=text,
            created_at=now,
            updated_at=now,
            note_type=normalized_type,  # type: ignore[arg-type]
            tags=tuple(tags or ()),
        )
    )


def update_incident_note(
    note: IncidentNote,
    *,
    text: str | None = None,
    note_type: str | None = None,
    author: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> IncidentNote:
    """Return a validated updated note, preserving note_id, incident_id, and created_at."""
    next_type = note.note_type if note_type is None else note_type
    normalized_type = validate_controlled_value(
        next_type,
        field_name="note type",
        allowed=NOTE_TYPES,
    )
    return validate_incident_note(
        IncidentNote(
            note_id=note.note_id,
            incident_id=note.incident_id,
            author=note.author if author is None else author,
            text=note.text if text is None else text,
            created_at=note.created_at,
            updated_at=utc_timestamp(),
            note_type=normalized_type,  # type: ignore[arg-type]
            tags=note.tags if tags is None else tuple(tags),
        )
    )


def validate_incident_note(note: IncidentNote) -> IncidentNote:
    """Validate every field of an incident note and return a normalized record."""
    note_id = validate_security_id(note.note_id, field_name="Note ID")
    incident_id = validate_security_id(note.incident_id, field_name="Incident ID")
    author = require_non_blank_text(
        note.author,
        field_name="Note author",
        max_length=MAX_CUSTODY_ACTOR_LENGTH,
    )
    text = require_non_blank_text(
        note.text,
        field_name="Note text",
        max_length=MAX_NOTE_TEXT_LENGTH,
    )
    note_type = validate_controlled_value(
        note.note_type,
        field_name="note type",
        allowed=NOTE_TYPES,
    )
    return IncidentNote(
        note_id=note_id,
        incident_id=incident_id,
        author=author,
        text=text,
        created_at=validate_utc_timestamp(note.created_at, field_name="Created at"),
        updated_at=validate_utc_timestamp(note.updated_at, field_name="Updated at"),
        note_type=note_type,  # type: ignore[arg-type]
        tags=normalize_tags(note.tags),
    )
