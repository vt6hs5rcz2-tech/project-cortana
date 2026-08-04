"""Immutable security indicator model for Project Cortana."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from src.config import (
    INDICATOR_CONFIDENCE_MAX,
    INDICATOR_CONFIDENCE_MIN,
    MAX_INDICATOR_NOTES_LENGTH,
    MAX_INDICATOR_VALUE_LENGTH,
)
from src.security_common import (
    INDICATOR_TYPES,
    IndicatorType,
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

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_DOMAIN_LABEL_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
)
SHA256_HEX_LENGTH = 64
SHA1_HEX_LENGTH = 40
MD5_HEX_LENGTH = 32


@dataclass(frozen=True)
class SecurityIndicator:
    """Immutable indicator of compromise recorded by an explicit user command."""

    indicator_id: str
    indicator_type: IndicatorType
    normalized_value: str
    original_value: str
    confidence: int
    first_seen_at: str
    last_seen_at: str
    created_at: str
    tags: tuple[str, ...]
    related_event_ids: tuple[str, ...]
    related_incident_ids: tuple[str, ...]
    notes: str | None


def create_security_indicator(
    *,
    indicator_type: str,
    value: str,
    confidence: int,
    notes: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    related_event_ids: list[str] | tuple[str, ...] | None = None,
    related_incident_ids: list[str] | tuple[str, ...] | None = None,
    first_seen_at: str | None = None,
    last_seen_at: str | None = None,
) -> SecurityIndicator:
    """Create a validated immutable security indicator."""
    now = utc_timestamp()
    original_value = require_non_blank_text(
        value,
        field_name="Indicator value",
        max_length=MAX_INDICATOR_VALUE_LENGTH,
    )
    normalized_type = validate_controlled_value(
        indicator_type,
        field_name="indicator type",
        allowed=INDICATOR_TYPES,
    )
    normalized_value = normalize_indicator_value(normalized_type, original_value)
    return validate_security_indicator(
        SecurityIndicator(
            indicator_id=str(uuid4()),
            indicator_type=normalized_type,  # type: ignore[arg-type]
            normalized_value=normalized_value,
            original_value=original_value,
            confidence=confidence,
            first_seen_at=first_seen_at or now,
            last_seen_at=last_seen_at or now,
            created_at=now,
            tags=tuple(tags or ()),
            related_event_ids=tuple(related_event_ids or ()),
            related_incident_ids=tuple(related_incident_ids or ()),
            notes=notes,
        )
    )


def replace_security_indicator(
    indicator: SecurityIndicator,
    *,
    confidence: int | None = None,
    notes: str | None = None,
    clear_notes: bool = False,
    tags: list[str] | tuple[str, ...] | None = None,
    related_event_ids: list[str] | tuple[str, ...] | None = None,
    related_incident_ids: list[str] | tuple[str, ...] | None = None,
    last_seen_at: str | None = None,
) -> SecurityIndicator:
    """Return a validated replacement indicator, preserving identity and created_at."""
    next_notes = indicator.notes
    if clear_notes:
        next_notes = None
    elif notes is not None:
        next_notes = notes

    return validate_security_indicator(
        SecurityIndicator(
            indicator_id=indicator.indicator_id,
            indicator_type=indicator.indicator_type,
            normalized_value=indicator.normalized_value,
            original_value=indicator.original_value,
            confidence=indicator.confidence if confidence is None else confidence,
            first_seen_at=indicator.first_seen_at,
            last_seen_at=(
                indicator.last_seen_at if last_seen_at is None else last_seen_at
            ),
            created_at=indicator.created_at,
            tags=indicator.tags if tags is None else tuple(tags),
            related_event_ids=(
                indicator.related_event_ids
                if related_event_ids is None
                else tuple(related_event_ids)
            ),
            related_incident_ids=(
                indicator.related_incident_ids
                if related_incident_ids is None
                else tuple(related_incident_ids)
            ),
            notes=next_notes,
        )
    )


def normalize_indicator_value(indicator_type: str, original_value: str) -> str:
    """Normalize an indicator value safely and deterministically without network I/O."""
    cleaned = original_value.strip()
    if not cleaned:
        raise SecurityValidationError("Indicator value cannot be blank.")

    if indicator_type == "ipv4":
        try:
            return str(ipaddress.IPv4Address(cleaned))
        except ValueError as error:
            raise SecurityValidationError("Invalid IPv4 indicator value.") from error

    if indicator_type == "ipv6":
        try:
            return str(ipaddress.IPv6Address(cleaned))
        except ValueError as error:
            raise SecurityValidationError("Invalid IPv6 indicator value.") from error

    if indicator_type == "domain":
        domain = cleaned.rstrip(".").lower()
        if not _DOMAIN_LABEL_RE.fullmatch(domain):
            raise SecurityValidationError("Invalid domain indicator value.")
        return domain

    if indicator_type == "email":
        if cleaned.count("@") != 1:
            raise SecurityValidationError("Invalid email indicator value.")
        local_part, domain_part = cleaned.split("@", 1)
        if not local_part or not domain_part:
            raise SecurityValidationError("Invalid email indicator value.")
        normalized_domain = normalize_indicator_value("domain", domain_part)
        return f"{local_part}@{normalized_domain}"

    if indicator_type == "url":
        parts = urlsplit(cleaned)
        if not parts.scheme or not parts.netloc:
            raise SecurityValidationError("Invalid URL indicator value.")
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))

    if indicator_type == "sha256":
        return _normalize_hash(cleaned, SHA256_HEX_LENGTH, "SHA-256")

    if indicator_type == "sha1":
        return _normalize_hash(cleaned, SHA1_HEX_LENGTH, "SHA-1")

    if indicator_type == "md5":
        return _normalize_hash(cleaned, MD5_HEX_LENGTH, "MD5")

    if indicator_type == "filename":
        if "/" in cleaned or "\\" in cleaned:
            raise SecurityValidationError(
                "Filename indicators cannot contain path separators."
            )
        return cleaned

    if indicator_type in {"process", "registry-key", "generic"}:
        return cleaned

    raise SecurityValidationError(f"Unsupported indicator type: {indicator_type}.")


def validate_security_indicator(indicator: SecurityIndicator) -> SecurityIndicator:
    """Validate every field of a security indicator and return a normalized record."""
    indicator_id = validate_security_id(
        indicator.indicator_id,
        field_name="Indicator ID",
    )
    indicator_type = validate_controlled_value(
        indicator.indicator_type,
        field_name="indicator type",
        allowed=INDICATOR_TYPES,
    )
    original_value = require_non_blank_text(
        indicator.original_value,
        field_name="Original indicator value",
        max_length=MAX_INDICATOR_VALUE_LENGTH,
    )
    expected_normalized = normalize_indicator_value(indicator_type, original_value)
    normalized_value = require_non_blank_text(
        indicator.normalized_value,
        field_name="Normalized indicator value",
        max_length=MAX_INDICATOR_VALUE_LENGTH,
    )
    if normalized_value != expected_normalized:
        raise SecurityValidationError(
            "Normalized indicator value does not match the original value."
        )

    if not isinstance(indicator.confidence, int) or isinstance(
        indicator.confidence, bool
    ):
        raise SecurityValidationError("Indicator confidence must be an integer.")
    if (
        indicator.confidence < INDICATOR_CONFIDENCE_MIN
        or indicator.confidence > INDICATOR_CONFIDENCE_MAX
    ):
        raise SecurityValidationError(
            "Indicator confidence must be between "
            f"{INDICATOR_CONFIDENCE_MIN} and {INDICATOR_CONFIDENCE_MAX}."
        )

    return SecurityIndicator(
        indicator_id=indicator_id,
        indicator_type=indicator_type,  # type: ignore[arg-type]
        normalized_value=normalized_value,
        original_value=original_value,
        confidence=indicator.confidence,
        first_seen_at=validate_utc_timestamp(
            indicator.first_seen_at,
            field_name="First seen at",
        ),
        last_seen_at=validate_utc_timestamp(
            indicator.last_seen_at,
            field_name="Last seen at",
        ),
        created_at=validate_utc_timestamp(
            indicator.created_at,
            field_name="Created at",
        ),
        tags=normalize_tags(indicator.tags),
        related_event_ids=normalize_id_list(
            indicator.related_event_ids,
            field_name="Related event ID",
        ),
        related_incident_ids=normalize_id_list(
            indicator.related_incident_ids,
            field_name="Related incident ID",
        ),
        notes=validate_optional_text(
            indicator.notes,
            field_name="Indicator notes",
            max_length=MAX_INDICATOR_NOTES_LENGTH,
        ),
    )


def indicator_log_reference(indicator: SecurityIndicator) -> str:
    """Return a redacted log-safe reference for an indicator."""
    digest_prefix = normalized_value_fingerprint(indicator.normalized_value)
    return f"type={indicator.indicator_type} fp={digest_prefix}"


def normalized_value_fingerprint(normalized_value: str) -> str:
    """Return a short non-reversible-looking fingerprint for logs."""
    return hashlib.sha256(normalized_value.encode("utf-8")).hexdigest()[:12]


def _normalize_hash(value: str, expected_length: int, label: str) -> str:
    """Validate an exact-length hexadecimal hash and return lowercase form."""
    cleaned = value.strip().lower()
    if len(cleaned) != expected_length or _HEX_RE.fullmatch(cleaned) is None:
        raise SecurityValidationError(
            f"{label} indicator must be exactly {expected_length} hexadecimal characters."
        )
    return cleaned
