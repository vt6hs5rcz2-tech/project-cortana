"""Bounded in-memory repository for prepared analyses and results."""

from __future__ import annotations

from src.config import MAX_RETAINED_INCIDENT_ANALYSES
from src.incident_analysis_common import (
    IncidentAnalysisConflictError,
    IncidentAnalysisLimitError,
    IncidentAnalysisValidationError,
)
from src.incident_analysis_models import (
    IncidentAnalysisRequest,
    IncidentAnalysisResult,
)
from src.tool_common import validate_uuid


class InMemoryIncidentAnalysisRepository:
    """Process-local analysis store with immutable identity and bounded retention."""

    def __init__(
        self,
        *,
        max_analyses: int = MAX_RETAINED_INCIDENT_ANALYSES,
    ) -> None:
        if max_analyses < 1:
            raise IncidentAnalysisValidationError(
                "max_analyses must be at least 1."
            )
        self._max_analyses = max_analyses
        self._prepared: dict[str, IncidentAnalysisRequest] = {}
        self._results: dict[str, IncidentAnalysisResult] = {}
        self._order: list[str] = []

    def store_prepared(self, request: IncidentAnalysisRequest) -> IncidentAnalysisRequest:
        """Store one prepared analysis request, rejecting duplicate IDs."""
        analysis_id = validate_uuid(request.analysis_id, field_name="Analysis ID")
        if analysis_id in self._prepared or analysis_id in self._results:
            raise IncidentAnalysisConflictError(
                "An analysis with that ID already exists."
            )
        if not request.previewed:
            raise IncidentAnalysisValidationError(
                "Prepared analyses must be marked previewed."
            )
        self._prepared[analysis_id] = request
        self._order.append(analysis_id)
        self._enforce_retention()
        return request

    def get_prepared(self, analysis_id: str) -> IncidentAnalysisRequest | None:
        """Return one prepared request by ID, or None when missing."""
        try:
            cleaned = validate_uuid(analysis_id, field_name="Analysis ID")
        except Exception:
            return None
        return self._prepared.get(cleaned)

    def store_result(self, result: IncidentAnalysisResult) -> IncidentAnalysisResult:
        """Store one completed/failed result for a previously prepared analysis."""
        analysis_id = validate_uuid(result.analysis_id, field_name="Analysis ID")
        prepared = self._prepared.get(analysis_id)
        if prepared is None:
            raise IncidentAnalysisValidationError(
                "Analysis must be prepared before storing a result."
            )
        if analysis_id in self._results:
            raise IncidentAnalysisConflictError(
                "Analysis result text cannot be replaced after completion."
            )
        if (
            result.incident_id != prepared.incident_id
            or result.scope_id != prepared.scope_id
            or result.analysis_kind != prepared.analysis_kind
            or result.packet_fingerprint != prepared.packet_fingerprint
        ):
            raise IncidentAnalysisConflictError(
                "Analysis result identity does not match the prepared request."
            )
        self._results[analysis_id] = result
        return result

    def get_result(self, analysis_id: str) -> IncidentAnalysisResult | None:
        """Return one analysis result by ID, or None when missing."""
        try:
            cleaned = validate_uuid(analysis_id, field_name="Analysis ID")
        except Exception:
            return None
        return self._results.get(cleaned)

    def mark_note_saved(
        self,
        analysis_id: str,
        *,
        note_id: str,
    ) -> IncidentAnalysisResult:
        """Mark an analysis as having saved exactly one incident note."""
        existing = self.get_result(analysis_id)
        if existing is None:
            raise IncidentAnalysisValidationError("Analysis result was not found.")
        if existing.status != "completed":
            raise IncidentAnalysisValidationError(
                "Only completed analyses can be saved as notes."
            )
        if existing.note_saved:
            raise IncidentAnalysisConflictError(
                "Analysis note was already saved."
            )
        updated = IncidentAnalysisResult(
            analysis_id=existing.analysis_id,
            incident_id=existing.incident_id,
            scope_id=existing.scope_id,
            analysis_kind=existing.analysis_kind,
            packet_fingerprint=existing.packet_fingerprint,
            boundary_token=existing.boundary_token,
            advisory_output=existing.advisory_output,
            created_at=existing.created_at,
            completed_at=existing.completed_at,
            status=existing.status,
            error_code=existing.error_code,
            note_saved=True,
            saved_note_id=validate_uuid(note_id, field_name="Note ID"),
            selected_event_count=existing.selected_event_count,
            selected_indicator_count=existing.selected_indicator_count,
            selected_note_count=existing.selected_note_count,
            selected_workflow_count=existing.selected_workflow_count,
            packet_character_count=existing.packet_character_count,
            output_character_count=existing.output_character_count,
        )
        self._results[updated.analysis_id] = updated
        return updated

    def list_results(self) -> list[IncidentAnalysisResult]:
        """Return retained results in insertion order."""
        return [
            self._results[analysis_id]
            for analysis_id in self._order
            if analysis_id in self._results
        ]

    def analysis_count(self) -> int:
        """Return the number of retained prepared/result analysis IDs."""
        return len(self._order)

    def _enforce_retention(self) -> None:
        while len(self._order) > self._max_analyses:
            removable_index = None
            for index, analysis_id in enumerate(self._order):
                result = self._results.get(analysis_id)
                if result is not None and result.note_saved:
                    removable_index = index
                    break
            if removable_index is None:
                removable_index = 0
            removed_id = self._order.pop(removable_index)
            self._prepared.pop(removed_id, None)
            self._results.pop(removed_id, None)
        if len(self._order) > self._max_analyses:
            raise IncidentAnalysisLimitError(
                "Retained incident analysis limit exceeded."
            )
