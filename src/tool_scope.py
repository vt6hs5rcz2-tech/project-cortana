"""Authorized scope model for defensive tool execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.config import MAX_TOOL_SCOPE_NAME_LENGTH, MAX_TOOL_SCOPE_NOTES_LENGTH
from src.tool_common import (
    TOOL_TARGET_TYPES,
    ToolAuthorizationError,
    ToolTargetType,
    ToolValidationError,
    is_expired,
    normalize_string_tuple,
    require_non_blank_text,
    utc_timestamp,
    validate_optional_text,
    validate_optional_utc_timestamp,
    validate_tool_id,
    validate_utc_timestamp,
    validate_uuid,
)


@dataclass(frozen=True)
class AuthorizedScope:
    """Immutable authorized scope binding tools to allowed targets."""

    scope_id: str
    scope_name: str
    allowed_tool_ids: tuple[str, ...]
    allowed_target_types: tuple[ToolTargetType, ...]
    allowed_local_path_roots: tuple[str, ...]
    allowed_incident_ids: tuple[str, ...]
    expires_at: str | None
    created_at: str
    active: bool
    notes: str | None


def create_authorized_scope(
    *,
    scope_name: str,
    allowed_tool_ids: list[str] | tuple[str, ...],
    allowed_target_types: list[str] | tuple[str, ...],
    allowed_local_path_roots: list[str] | tuple[str, ...] | None = None,
    allowed_incident_ids: list[str] | tuple[str, ...] | None = None,
    expires_at: str | None = None,
    notes: str | None = None,
    active: bool = True,
    prohibited_tool_ids: frozenset[str] | set[str] | None = None,
) -> AuthorizedScope:
    """Create a validated immutable authorized scope."""
    now = utc_timestamp()
    cleaned_name = require_non_blank_text(
        scope_name,
        field_name="Scope name",
        max_length=MAX_TOOL_SCOPE_NAME_LENGTH,
    )
    tool_ids = _normalize_tool_ids(allowed_tool_ids)
    if prohibited_tool_ids:
        blocked = sorted(set(tool_ids) & set(prohibited_tool_ids))
        if blocked:
            raise ToolAuthorizationError(
                "Scope cannot authorize prohibited tools."
            )

    targets = normalize_string_tuple(
        allowed_target_types,
        field_name="allowed target types",
        allowed=TOOL_TARGET_TYPES,
        allow_empty=False,
    )
    roots = _normalize_path_roots(allowed_local_path_roots)
    incidents = _normalize_incident_ids(allowed_incident_ids)
    cleaned_notes = validate_optional_text(
        notes,
        field_name="Scope notes",
        max_length=MAX_TOOL_SCOPE_NOTES_LENGTH,
    )
    cleaned_expires = validate_optional_utc_timestamp(
        expires_at,
        field_name="Scope expiry",
    )

    return validate_authorized_scope(
        AuthorizedScope(
            scope_id=str(uuid4()),
            scope_name=cleaned_name,
            allowed_tool_ids=tool_ids,
            allowed_target_types=targets,  # type: ignore[arg-type]
            allowed_local_path_roots=roots,
            allowed_incident_ids=incidents,
            expires_at=cleaned_expires,
            created_at=now,
            active=bool(active),
            notes=cleaned_notes,
        )
    )


def disable_authorized_scope(scope: AuthorizedScope) -> AuthorizedScope:
    """Return a validated replacement scope marked inactive."""
    return validate_authorized_scope(
        AuthorizedScope(
            scope_id=scope.scope_id,
            scope_name=scope.scope_name,
            allowed_tool_ids=scope.allowed_tool_ids,
            allowed_target_types=scope.allowed_target_types,
            allowed_local_path_roots=scope.allowed_local_path_roots,
            allowed_incident_ids=scope.allowed_incident_ids,
            expires_at=scope.expires_at,
            created_at=scope.created_at,
            active=False,
            notes=scope.notes,
        )
    )


def validate_authorized_scope(scope: AuthorizedScope) -> AuthorizedScope:
    """Validate an authorized scope record."""
    scope_id = validate_uuid(scope.scope_id, field_name="Scope ID")
    scope_name = require_non_blank_text(
        scope.scope_name,
        field_name="Scope name",
        max_length=MAX_TOOL_SCOPE_NAME_LENGTH,
    )
    tool_ids = _normalize_tool_ids(scope.allowed_tool_ids)
    targets = normalize_string_tuple(
        scope.allowed_target_types,
        field_name="allowed target types",
        allowed=TOOL_TARGET_TYPES,
        allow_empty=False,
    )
    roots = tuple(scope.allowed_local_path_roots)
    for root in roots:
        if not isinstance(root, str) or not root.strip():
            raise ToolValidationError("Allowed path roots must be non-blank strings.")
    incidents = _normalize_incident_ids(scope.allowed_incident_ids)
    created_at = validate_utc_timestamp(scope.created_at, field_name="Scope created_at")
    expires_at = validate_optional_utc_timestamp(
        scope.expires_at,
        field_name="Scope expiry",
    )
    notes = validate_optional_text(
        scope.notes,
        field_name="Scope notes",
        max_length=MAX_TOOL_SCOPE_NOTES_LENGTH,
    )
    return AuthorizedScope(
        scope_id=scope_id,
        scope_name=scope_name,
        allowed_tool_ids=tool_ids,
        allowed_target_types=targets,  # type: ignore[arg-type]
        allowed_local_path_roots=roots,
        allowed_incident_ids=incidents,
        expires_at=expires_at,
        created_at=created_at,
        active=bool(scope.active),
        notes=notes,
    )


def assert_scope_usable(scope: AuthorizedScope) -> None:
    """Reject inactive or expired scopes."""
    if not scope.active:
        raise ToolAuthorizationError("Scope is inactive.")
    if is_expired(scope.expires_at):
        raise ToolAuthorizationError("Scope has expired.")


def assert_tool_authorized(scope: AuthorizedScope, tool_id: str) -> None:
    """Reject tools that are not explicitly allowed by the scope."""
    assert_scope_usable(scope)
    cleaned_tool_id = validate_tool_id(tool_id)
    if not scope.allowed_tool_ids:
        raise ToolAuthorizationError(
            "Scope authorizes no tools."
        )
    if cleaned_tool_id not in scope.allowed_tool_ids:
        raise ToolAuthorizationError(
            "Scope does not authorize the requested tool."
        )


def assert_target_type_authorized(
    scope: AuthorizedScope,
    target_type: str,
) -> None:
    """Reject target types outside the authorized scope."""
    assert_scope_usable(scope)
    cleaned = target_type.strip().lower()
    if cleaned not in scope.allowed_target_types:
        raise ToolAuthorizationError(
            "Scope does not authorize the requested target type."
        )


def assert_incident_authorized(
    scope: AuthorizedScope,
    incident_id: str | None,
) -> None:
    """Reject unauthorized incident links when the scope restricts incidents."""
    if incident_id is None:
        return
    assert_scope_usable(scope)
    cleaned = validate_uuid(incident_id, field_name="Incident ID")
    if scope.allowed_incident_ids and cleaned not in scope.allowed_incident_ids:
        raise ToolAuthorizationError(
            "Scope does not authorize the requested incident."
        )


def assert_path_authorized(scope: AuthorizedScope, path_value: str) -> Path:
    """Canonicalize a path and ensure it remains under an allowed root."""
    assert_scope_usable(scope)
    if not scope.allowed_local_path_roots:
        raise ToolAuthorizationError(
            "Scope does not authorize local file paths."
        )

    try:
        candidate = Path(path_value).expanduser()
        if candidate.is_symlink():
            raise ToolAuthorizationError(
                "Symbolic links and reparse points are not allowed as tool targets."
            )
        resolved = candidate.resolve(strict=False)
    except OSError as error:
        raise ToolAuthorizationError(
            "Target path could not be validated safely."
        ) from error

    # Reject symlink parents by walking after resolve is not enough on all platforms;
    # also reject if any path component in the unresolved chain is a symlink.
    _reject_symlink_escape(candidate)

    for root in scope.allowed_local_path_roots:
        try:
            root_path = Path(root)
            if root_path.is_symlink():
                continue
            root_resolved = root_path.resolve(strict=False)
        except OSError:
            continue
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue

    raise ToolAuthorizationError(
        "Target path is outside the authorized scope roots."
    )


def canonicalize_path_root(path_value: str) -> str:
    """Canonicalize one allowed path root for persistence."""
    cleaned = require_non_blank_text(
        path_value,
        field_name="Allowed path root",
        max_length=4096,
    )
    try:
        path = Path(cleaned).expanduser()
        if path.is_symlink():
            raise ToolValidationError(
                "Allowed path roots cannot be symbolic links."
            )
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise ToolValidationError(
            "Allowed path root could not be canonicalized."
        ) from error

    if not resolved.exists():
        # Allow declaring a root that will exist; still canonicalize the path string.
        return os.fspath(resolved)
    if not resolved.is_dir():
        raise ToolValidationError("Allowed path roots must be directories.")
    return os.fspath(resolved)


def _normalize_tool_ids(
    values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ToolValidationError("Allowed tool IDs must be a list or tuple.")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tool_id = validate_tool_id(value)
        if tool_id in seen:
            continue
        seen.add(tool_id)
        normalized.append(tool_id)
    return tuple(normalized)


def _normalize_incident_ids(
    values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ToolValidationError("Allowed incident IDs must be a list or tuple.")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        incident_id = validate_uuid(str(value), field_name="Incident ID")
        if incident_id in seen:
            continue
        seen.add(incident_id)
        normalized.append(incident_id)
    return tuple(normalized)


def _normalize_path_roots(
    values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ToolValidationError("Allowed path roots must be a list or tuple.")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ToolValidationError("Each allowed path root must be a string.")
        cleaned = value.strip()
        if cleaned.lower() == "none":
            continue
        root = canonicalize_path_root(cleaned)
        if root in seen:
            continue
        seen.add(root)
        normalized.append(root)
    return tuple(normalized)


def _reject_symlink_escape(path: Path) -> None:
    """Reject paths that traverse through symbolic links."""
    try:
        current = path if path.is_absolute() else Path.cwd() / path
        parts = current.parts
        progressive = Path(parts[0]) if parts else Path(".")
        for part in parts[1:]:
            progressive = progressive / part
            if progressive.exists() and progressive.is_symlink():
                raise ToolAuthorizationError(
                    "Symbolic links and reparse points are not allowed as tool targets."
                )
    except ToolAuthorizationError:
        raise
    except OSError as error:
        raise ToolAuthorizationError(
            "Target path could not be validated safely."
        ) from error
