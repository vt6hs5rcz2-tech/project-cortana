"""Immutable chain-of-custody entry model for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.config import (
    MAX_CUSTODY_ACTOR_LENGTH,
    MAX_CUSTODY_NOTES_LENGTH,
    MAX_CUSTODY_REASON_LENGTH,
)
from src.security_common import (
    CUSTODY_ACTIONS,
    CustodyAction,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_text,
    validate_security_id,
    validate_utc_timestamp,
)
from src.security_evidence import validate_sha256_hex


@dataclass(frozen=True)
class ChainOfCustodyEntry:
    """Immutable append-only chain-of-custody entry for stored evidence."""

    entry_id: str
    evidence_id: str
    action: CustodyAction
    actor: str
    timestamp: str
    reason: str
    previous_hash: str | None
    resulting_hash: str | None
    notes: str | None


def create_custody_entry(
    *,
    evidence_id: str,
    action: str,
    actor: str,
    reason: str,
    previous_hash: str | None = None,
    resulting_hash: str | None = None,
    notes: str | None = None,
    timestamp: str | None = None,
) -> ChainOfCustodyEntry:
    """Create a validated immutable chain-of-custody entry."""
    normalized_action = validate_controlled_value(
        action,
        field_name="custody action",
        allowed=CUSTODY_ACTIONS,
    )
    return validate_custody_entry(
        ChainOfCustodyEntry(
            entry_id=str(uuid4()),
            evidence_id=evidence_id,
            action=normalized_action,  # type: ignore[arg-type]
            actor=actor,
            timestamp=timestamp or utc_timestamp(),
            reason=reason,
            previous_hash=previous_hash,
            resulting_hash=resulting_hash,
            notes=notes,
        )
    )


def validate_custody_entry(entry: ChainOfCustodyEntry) -> ChainOfCustodyEntry:
    """Validate every field of a custody entry and return a normalized record."""
    entry_id = validate_security_id(entry.entry_id, field_name="Custody entry ID")
    evidence_id = validate_security_id(entry.evidence_id, field_name="Evidence ID")
    action = validate_controlled_value(
        entry.action,
        field_name="custody action",
        allowed=CUSTODY_ACTIONS,
    )
    actor = require_non_blank_text(
        entry.actor,
        field_name="Custody actor",
        max_length=MAX_CUSTODY_ACTOR_LENGTH,
    )
    reason = require_non_blank_text(
        entry.reason,
        field_name="Custody reason",
        max_length=MAX_CUSTODY_REASON_LENGTH,
    )
    previous_hash = (
        None
        if entry.previous_hash is None
        else validate_sha256_hex(entry.previous_hash)
    )
    resulting_hash = (
        None
        if entry.resulting_hash is None
        else validate_sha256_hex(entry.resulting_hash)
    )

    return ChainOfCustodyEntry(
        entry_id=entry_id,
        evidence_id=evidence_id,
        action=action,  # type: ignore[arg-type]
        actor=actor,
        timestamp=validate_utc_timestamp(
            entry.timestamp,
            field_name="Custody timestamp",
        ),
        reason=reason,
        previous_hash=previous_hash,
        resulting_hash=resulting_hash,
        notes=validate_optional_text(
            entry.notes,
            field_name="Custody notes",
            max_length=MAX_CUSTODY_NOTES_LENGTH,
        ),
    )
