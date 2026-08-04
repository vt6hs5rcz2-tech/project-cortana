"""Immutable evidence metadata model for Project Cortana."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.config import (
    MAX_CUSTODY_ACTOR_LENGTH,
    MAX_EVIDENCE_DESCRIPTION_LENGTH,
    MAX_EVIDENCE_FILENAME_LENGTH,
    MAX_EVIDENCE_SOURCE_BYTES,
    MAX_EVIDENCE_TITLE_LENGTH,
)
from src.security_common import (
    EVIDENCE_STORAGE_STATUSES,
    EVIDENCE_TYPES,
    EvidenceStorageStatus,
    EvidenceType,
    SecurityValidationError,
    normalize_id_list,
    normalize_tags,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_text,
    validate_security_id,
    validate_utc_timestamp,
)

SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class EvidenceRecord:
    """Immutable evidence metadata recorded by an explicit user command."""

    evidence_id: str
    evidence_type: EvidenceType
    title: str
    description: str
    original_filename: str | None
    sha256_hash: str
    source_size_bytes: int
    collected_at: str
    recorded_at: str
    collector: str
    storage_status: EvidenceStorageStatus
    related_event_ids: tuple[str, ...]
    related_incident_ids: tuple[str, ...]
    chain_of_custody_entry_ids: tuple[str, ...]
    tags: tuple[str, ...]


def create_evidence_record(
    *,
    evidence_type: str,
    title: str,
    description: str,
    sha256_hash: str,
    source_size_bytes: int,
    collector: str,
    storage_status: str,
    evidence_id: str | None = None,
    original_filename: str | None = None,
    collected_at: str | None = None,
    related_event_ids: list[str] | tuple[str, ...] | None = None,
    related_incident_ids: list[str] | tuple[str, ...] | None = None,
    chain_of_custody_entry_ids: list[str] | tuple[str, ...] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> EvidenceRecord:
    """Create a validated immutable evidence metadata record."""
    now = utc_timestamp()
    normalized_type = validate_controlled_value(
        evidence_type,
        field_name="evidence type",
        allowed=EVIDENCE_TYPES,
    )
    normalized_storage = validate_controlled_value(
        storage_status,
        field_name="evidence storage status",
        allowed=EVIDENCE_STORAGE_STATUSES,
    )
    return validate_evidence_record(
        EvidenceRecord(
            evidence_id=evidence_id or str(uuid4()),
            evidence_type=normalized_type,  # type: ignore[arg-type]
            title=title,
            description=description,
            original_filename=original_filename,
            sha256_hash=sha256_hash,
            source_size_bytes=source_size_bytes,
            collected_at=collected_at or now,
            recorded_at=now,
            collector=collector,
            storage_status=normalized_storage,  # type: ignore[arg-type]
            related_event_ids=tuple(related_event_ids or ()),
            related_incident_ids=tuple(related_incident_ids or ()),
            chain_of_custody_entry_ids=tuple(chain_of_custody_entry_ids or ()),
            tags=tuple(tags or ()),
        )
    )


def replace_evidence_record(
    evidence: EvidenceRecord,
    *,
    storage_status: str | None = None,
    related_event_ids: list[str] | tuple[str, ...] | None = None,
    related_incident_ids: list[str] | tuple[str, ...] | None = None,
    chain_of_custody_entry_ids: list[str] | tuple[str, ...] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> EvidenceRecord:
    """Return a validated replacement evidence record."""
    next_storage = (
        evidence.storage_status if storage_status is None else storage_status
    )
    normalized_storage = validate_controlled_value(
        next_storage,
        field_name="evidence storage status",
        allowed=EVIDENCE_STORAGE_STATUSES,
    )
    return validate_evidence_record(
        EvidenceRecord(
            evidence_id=evidence.evidence_id,
            evidence_type=evidence.evidence_type,
            title=evidence.title,
            description=evidence.description,
            original_filename=evidence.original_filename,
            sha256_hash=evidence.sha256_hash,
            source_size_bytes=evidence.source_size_bytes,
            collected_at=evidence.collected_at,
            recorded_at=evidence.recorded_at,
            collector=evidence.collector,
            storage_status=normalized_storage,  # type: ignore[arg-type]
            related_event_ids=(
                evidence.related_event_ids
                if related_event_ids is None
                else tuple(related_event_ids)
            ),
            related_incident_ids=(
                evidence.related_incident_ids
                if related_incident_ids is None
                else tuple(related_incident_ids)
            ),
            chain_of_custody_entry_ids=(
                evidence.chain_of_custody_entry_ids
                if chain_of_custody_entry_ids is None
                else tuple(chain_of_custody_entry_ids)
            ),
            tags=evidence.tags if tags is None else tuple(tags),
        )
    )


def validate_evidence_record(evidence: EvidenceRecord) -> EvidenceRecord:
    """Validate every field of an evidence record and return a normalized record."""
    evidence_id = validate_security_id(evidence.evidence_id, field_name="Evidence ID")
    evidence_type = validate_controlled_value(
        evidence.evidence_type,
        field_name="evidence type",
        allowed=EVIDENCE_TYPES,
    )
    title = require_non_blank_text(
        evidence.title,
        field_name="Evidence title",
        max_length=MAX_EVIDENCE_TITLE_LENGTH,
    )
    description = require_non_blank_text(
        evidence.description,
        field_name="Evidence description",
        max_length=MAX_EVIDENCE_DESCRIPTION_LENGTH,
    )
    original_filename = validate_optional_text(
        evidence.original_filename,
        field_name="Original filename",
        max_length=MAX_EVIDENCE_FILENAME_LENGTH,
    )
    if original_filename is not None and (
        "/" in original_filename or "\\" in original_filename
    ):
        raise SecurityValidationError(
            "Original filename cannot contain path separators."
        )

    sha256_hash = validate_sha256_hex(evidence.sha256_hash)
    source_size_bytes = _validate_source_size(evidence.source_size_bytes)
    collector = require_non_blank_text(
        evidence.collector,
        field_name="Collector",
        max_length=MAX_CUSTODY_ACTOR_LENGTH,
    )
    storage_status = validate_controlled_value(
        evidence.storage_status,
        field_name="evidence storage status",
        allowed=EVIDENCE_STORAGE_STATUSES,
    )

    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence_type=evidence_type,  # type: ignore[arg-type]
        title=title,
        description=description,
        original_filename=original_filename,
        sha256_hash=sha256_hash,
        source_size_bytes=source_size_bytes,
        collected_at=validate_utc_timestamp(
            evidence.collected_at,
            field_name="Collected at",
        ),
        recorded_at=validate_utc_timestamp(
            evidence.recorded_at,
            field_name="Recorded at",
        ),
        collector=collector,
        storage_status=storage_status,  # type: ignore[arg-type]
        related_event_ids=normalize_id_list(
            evidence.related_event_ids,
            field_name="Related event ID",
        ),
        related_incident_ids=normalize_id_list(
            evidence.related_incident_ids,
            field_name="Related incident ID",
        ),
        chain_of_custody_entry_ids=normalize_id_list(
            evidence.chain_of_custody_entry_ids,
            field_name="Chain-of-custody entry ID",
        ),
        tags=normalize_tags(evidence.tags),
    )


def validate_sha256_hex(value: str) -> str:
    """Validate and return a lowercase SHA-256 hex digest."""
    cleaned = value.strip().lower()
    if len(cleaned) != SHA256_HEX_LENGTH:
        raise SecurityValidationError("Evidence hash must be SHA-256 hex.")
    if any(character not in "0123456789abcdef" for character in cleaned):
        raise SecurityValidationError("Evidence hash must be SHA-256 hex.")
    return cleaned


def _validate_source_size(source_size_bytes: int) -> int:
    """Validate evidence source size against centralized limits."""
    if not isinstance(source_size_bytes, int) or isinstance(source_size_bytes, bool):
        raise SecurityValidationError("Source size must be an integer.")
    if source_size_bytes < 0:
        raise SecurityValidationError("Source size cannot be negative.")
    if source_size_bytes > MAX_EVIDENCE_SOURCE_BYTES:
        raise SecurityValidationError(
            "Evidence source exceeds the maximum size of "
            f"{MAX_EVIDENCE_SOURCE_BYTES} bytes."
        )
    return source_size_bytes
