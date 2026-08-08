"""Thin bridge from incident-linked evidence to ordinary text-search requests.

Creates ToolExecutionRequest records only. Execution remains governed by the
existing dry-run, approval, and DefensiveToolExecutor pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.evidence_store import EvidenceStore, EvidenceStoreError
from src.incident_repository import IncidentRepository
from src.tool_common import (
    ToolParameterError,
    ToolValidationError,
    fingerprint_display_prefix,
    validate_uuid,
)
from src.tool_definition import validate_parameters_against_schema
from src.tool_policy import assert_requestable, initial_request_status
from src.tool_registry import ToolRegistry
from src.tool_repository import ToolControlRepository
from src.tool_request import create_tool_execution_request
from src.tool_scope import (
    AuthorizedScope,
    assert_incident_authorized,
    assert_path_authorized,
    assert_tool_authorized,
)


class IncidentEvidenceSearchError(ToolValidationError):
    """Raised when an incident-scoped evidence search request cannot be built."""


@dataclass(frozen=True)
class EvidenceTextSearchRequestCreated:
    """Safe operator-facing metadata for a created text-search request."""

    request_id: str
    tool_id: str
    request_status: str
    fingerprint_prefix: str
    incident_id: str
    evidence_id: str


def create_evidence_text_search_request(
    *,
    incident_repository: IncidentRepository,
    evidence_store: EvidenceStore,
    tool_registry: ToolRegistry,
    tool_repository: ToolControlRepository,
    incident_id: str,
    evidence_id: str,
    scope_id: str,
    query: str,
    max_matches: int | None = None,
    justification: str = (
        "Incident-scoped text search of registered evidence via /evidence-search."
    ),
) -> EvidenceTextSearchRequestCreated:
    """Authorize and create one ordinary text-search ToolExecutionRequest."""
    cleaned_incident_id = validate_uuid(incident_id, field_name="Incident ID")
    cleaned_evidence_id = validate_uuid(evidence_id, field_name="Evidence ID")
    cleaned_scope_id = validate_uuid(scope_id, field_name="Scope ID")

    incident = incident_repository.get_incident(cleaned_incident_id)
    if incident is None:
        raise IncidentEvidenceSearchError("Incident was not found.")

    evidence = incident_repository.get_evidence(cleaned_evidence_id)
    if evidence is None:
        raise IncidentEvidenceSearchError("Evidence was not found.")

    if cleaned_evidence_id not in incident.evidence_ids:
        raise IncidentEvidenceSearchError(
            "Evidence is not linked to the requested incident."
        )
    if cleaned_incident_id not in evidence.related_incident_ids:
        raise IncidentEvidenceSearchError(
            "Evidence is not linked to the requested incident."
        )

    scope = tool_repository.get_scope(cleaned_scope_id)
    if scope is None:
        raise IncidentEvidenceSearchError("Authorized scope was not found.")

    definition = tool_registry.require("text-search")
    assert_requestable(definition)
    assert_tool_authorized(scope, definition.tool_id)
    assert_incident_authorized(scope, cleaned_incident_id)

    try:
        resolved_path = evidence_store.resolve_stored_evidence_path(cleaned_evidence_id)
    except EvidenceStoreError as error:
        raise IncidentEvidenceSearchError(str(error.user_message)) from error

    assert_path_authorized(scope, os.fspath(resolved_path))

    raw_parameters: dict[str, object] = {
        "path": os.fspath(resolved_path),
        "query": query,
    }
    if max_matches is not None:
        raw_parameters["max_matches"] = max_matches

    try:
        normalized = validate_parameters_against_schema(
            raw_parameters,
            definition.parameter_schema,
        )
    except (ToolParameterError, ToolValidationError) as error:
        raise IncidentEvidenceSearchError(str(error)) from error

    request = create_tool_execution_request(
        tool_id=definition.tool_id,
        normalized_parameters=normalized,
        scope_id=scope.scope_id,
        justification=justification,
        request_status=initial_request_status(definition),
        incident_id=cleaned_incident_id,
    )
    saved = tool_repository.add_request(request)
    return EvidenceTextSearchRequestCreated(
        request_id=saved.request_id,
        tool_id=saved.tool_id,
        request_status=saved.request_status,
        fingerprint_prefix=fingerprint_display_prefix(saved.fingerprint),
        incident_id=cleaned_incident_id,
        evidence_id=cleaned_evidence_id,
    )


def authorize_scope_for_incident_context(
    *,
    tool_repository: ToolControlRepository,
    scope_id: str,
    incident_id: str,
) -> AuthorizedScope:
    """Load a scope and authorize the incident for bounded context display."""
    cleaned_incident_id = validate_uuid(incident_id, field_name="Incident ID")
    cleaned_scope_id = validate_uuid(scope_id, field_name="Scope ID")
    scope = tool_repository.get_scope(cleaned_scope_id)
    if scope is None:
        raise IncidentEvidenceSearchError("Authorized scope was not found.")
    assert_incident_authorized(scope, cleaned_incident_id)
    return scope
