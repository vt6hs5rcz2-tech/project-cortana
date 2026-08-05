"""Incident analysis service: the sole AI entry point for Milestone 12."""

from __future__ import annotations

import logging
from uuid import uuid4

from src.ai_service import OpenAIClient, generate_incident_analysis_response
from src.config import (
    AI_INCIDENT_ANALYSIS_ENABLED,
    AI_INCIDENT_ANALYSIS_NOTE_AUTHOR,
    AI_INCIDENT_ANALYSIS_NOTE_TAG,
    AI_INCIDENT_ANALYSIS_NOTE_TYPE,
    AI_INCIDENT_NOTE_SAVE_ENABLED,
    MAX_INCIDENT_ANALYSIS_OUTPUT_CHARS,
    MAX_NOTE_TEXT_LENGTH,
)
from src.incident_analysis_audit import InMemoryIncidentAnalysisAuditLog
from src.incident_analysis_common import (
    AnalysisKind,
    IncidentAnalysisAuthorizationError,
    IncidentAnalysisConflictError,
    IncidentAnalysisDisabledError,
    IncidentAnalysisNoteSaveDisabledError,
    IncidentAnalysisValidationError,
    bound_analysis_text,
    generate_incident_analysis_boundary_token,
)
from src.incident_analysis_models import (
    IncidentAnalysisRequest,
    IncidentAnalysisResult,
)
from src.incident_analysis_packet import build_incident_analysis_packet
from src.incident_analysis_repository import InMemoryIncidentAnalysisRepository
from src.incident_repository import IncidentRepository
from src.security_note import create_incident_note
from src.settings import Settings
from src.tool_common import utc_timestamp, validate_uuid
from src.tool_repository import ToolControlRepository
from src.tool_scope import assert_incident_authorized
from src.workflow_repository import WorkflowRunRepository

logger = logging.getLogger("ProjectCortana")

AI_FALLBACK_OUTPUT = (
    "Advisory analysis unavailable. The AI service could not complete this "
    "request. No verified findings were produced."
)
AI_ERROR_CODE = "ai_service_error"

KIND_USER_PROMPTS: dict[AnalysisKind, str] = {
    "summary": (
        "Provide a cautious advisory summary of this single incident packet. "
        "Label all conclusions as advisory and untrusted. Do not claim verified "
        "fact or evidence. Do not invent missing details."
    ),
    "gaps": (
        "Identify investigative gaps suggested by this single incident packet. "
        "Label all conclusions as advisory and untrusted. Do not invent evidence."
    ),
    "investigation_questions": (
        "Propose careful defensive investigation questions based only on this "
        "single incident packet. Label all output as advisory and untrusted."
    ),
    "report_draft": (
        "Draft a cautious advisory incident report outline based only on this "
        "single incident packet. Label all content as advisory and untrusted."
    ),
    "remediation_checklist": (
        "Propose a cautious defensive remediation checklist based only on this "
        "single incident packet. Do not recommend offensive or destructive "
        "actions. Label all content as advisory and untrusted."
    ),
    "outcome_explanation": (
        "Explain possible outcomes suggested by this single incident packet in "
        "cautious advisory terms. Do not claim certainty. Do not invent facts."
    ),
}


class IncidentAnalysisService:
    """Authorize, prepare, run, and save incident analyses under explicit control."""

    def __init__(
        self,
        *,
        incident_repository: IncidentRepository,
        tool_repository: ToolControlRepository,
        workflow_run_repository: WorkflowRunRepository,
        analysis_repository: InMemoryIncidentAnalysisRepository,
        audit_log: InMemoryIncidentAnalysisAuditLog,
        settings: Settings,
        client: OpenAIClient | None,
    ) -> None:
        self._incident_repository = incident_repository
        self._tool_repository = tool_repository
        self._workflow_run_repository = workflow_run_repository
        self._analysis_repository = analysis_repository
        self._audit_log = audit_log
        self._settings = settings
        self._client = client

    def prepare(
        self,
        *,
        analysis_kind: AnalysisKind,
        incident_id: str,
        scope_id: str,
        selected_event_ids: tuple[str, ...],
        selected_indicator_ids: tuple[str, ...],
        selected_note_ids: tuple[str, ...],
        selected_workflow_run_ids: tuple[str, ...],
    ) -> IncidentAnalysisRequest:
        """Authorize selections, build a packet, and store a previewed request."""
        self._assert_analysis_enabled()
        cleaned_incident_id = validate_uuid(incident_id, field_name="Incident ID")
        cleaned_scope_id = validate_uuid(scope_id, field_name="Scope ID")
        self._authorize(cleaned_scope_id, cleaned_incident_id)

        packet = build_incident_analysis_packet(
            incident_repository=self._incident_repository,
            workflow_run_repository=self._workflow_run_repository,
            incident_id=cleaned_incident_id,
            selected_event_ids=selected_event_ids,
            selected_indicator_ids=selected_indicator_ids,
            selected_note_ids=selected_note_ids,
            selected_workflow_run_ids=selected_workflow_run_ids,
        )
        # Revalidate immediately before retaining prepared state.
        self._authorize(cleaned_scope_id, cleaned_incident_id)

        request = IncidentAnalysisRequest(
            analysis_id=str(uuid4()),
            incident_id=cleaned_incident_id,
            scope_id=cleaned_scope_id,
            analysis_kind=analysis_kind,
            selected_event_ids=selected_event_ids,
            selected_indicator_ids=selected_indicator_ids,
            selected_note_ids=selected_note_ids,
            selected_workflow_run_ids=selected_workflow_run_ids,
            created_at=utc_timestamp(),
            previewed=True,
            packet_fingerprint=packet.packet_fingerprint,
            boundary_token=generate_incident_analysis_boundary_token(),
            packet=packet,
        )
        stored = self._analysis_repository.store_prepared(request)
        self._audit_log.append(
            analysis_id=stored.analysis_id,
            incident_id=stored.incident_id,
            scope_id=stored.scope_id,
            analysis_kind=stored.analysis_kind,
            action="analysis_prepared",
            selected_event_count=packet.selected_event_count,
            selected_indicator_count=packet.selected_indicator_count,
            selected_note_count=packet.selected_note_count,
            selected_workflow_count=packet.selected_workflow_count,
            packet_character_count=packet.character_count,
            status="prepared",
        )
        return stored

    def run(self, analysis_id: str) -> IncidentAnalysisResult:
        """Revalidate authorization and execute AI analysis for one prepared request."""
        self._assert_analysis_enabled()
        prepared = self._analysis_repository.get_prepared(analysis_id)
        if prepared is None:
            raise IncidentAnalysisValidationError(
                f"No prepared analysis found with ID '{analysis_id}'."
            )
        if self._analysis_repository.get_result(prepared.analysis_id) is not None:
            raise IncidentAnalysisConflictError(
                "That analysis has already been run."
            )

        self._authorize(prepared.scope_id, prepared.incident_id)
        if prepared.packet.packet_fingerprint != prepared.packet_fingerprint:
            raise IncidentAnalysisConflictError(
                "Prepared packet fingerprint mismatch."
            )

        self._audit_log.append(
            analysis_id=prepared.analysis_id,
            incident_id=prepared.incident_id,
            scope_id=prepared.scope_id,
            analysis_kind=prepared.analysis_kind,
            action="analysis_confirmed",
            selected_event_count=prepared.packet.selected_event_count,
            selected_indicator_count=prepared.packet.selected_indicator_count,
            selected_note_count=prepared.packet.selected_note_count,
            selected_workflow_count=prepared.packet.selected_workflow_count,
            packet_character_count=prepared.packet.character_count,
            status="confirmed",
        )

        if self._client is None:
            result = self._store_failed(
                prepared,
                error_code="ai_unavailable",
                advisory_output=AI_FALLBACK_OUTPUT,
            )
            return result

        try:
            raw = generate_incident_analysis_response(
                client=self._client,
                settings=self._settings,
                question=KIND_USER_PROMPTS[prepared.analysis_kind],
                packet=prepared.packet,
                boundary_token=prepared.boundary_token,
            )
            cleaned = " ".join(raw.split()).strip()
            if not cleaned:
                cleaned = AI_FALLBACK_OUTPUT
                error_code: str | None = "empty_ai_output"
                status = "failed"
            else:
                error_code = None
                status = "completed"
            bounded = bound_analysis_text(
                cleaned,
                max_length=MAX_INCIDENT_ANALYSIS_OUTPUT_CHARS,
            )
        except Exception as error:
            logger.error(
                "Incident analysis AI call failed error_type=%s",
                type(error).__name__,
            )
            return self._store_failed(
                prepared,
                error_code=AI_ERROR_CODE,
                advisory_output=AI_FALLBACK_OUTPUT,
            )

        # Revalidate again before retaining the analysis result.
        self._authorize(prepared.scope_id, prepared.incident_id)
        completed_at = utc_timestamp()
        result = IncidentAnalysisResult(
            analysis_id=prepared.analysis_id,
            incident_id=prepared.incident_id,
            scope_id=prepared.scope_id,
            analysis_kind=prepared.analysis_kind,
            packet_fingerprint=prepared.packet_fingerprint,
            boundary_token=prepared.boundary_token,
            advisory_output=bounded,
            created_at=prepared.created_at,
            completed_at=completed_at,
            status=status,  # type: ignore[arg-type]
            error_code=error_code,
            note_saved=False,
            saved_note_id=None,
            selected_event_count=prepared.packet.selected_event_count,
            selected_indicator_count=prepared.packet.selected_indicator_count,
            selected_note_count=prepared.packet.selected_note_count,
            selected_workflow_count=prepared.packet.selected_workflow_count,
            packet_character_count=prepared.packet.character_count,
            output_character_count=len(bounded),
        )
        stored = self._analysis_repository.store_result(result)
        self._audit_log.append(
            analysis_id=stored.analysis_id,
            incident_id=stored.incident_id,
            scope_id=stored.scope_id,
            analysis_kind=stored.analysis_kind,
            action=(
                "analysis_completed"
                if stored.status == "completed"
                else "analysis_failed"
            ),
            selected_event_count=stored.selected_event_count,
            selected_indicator_count=stored.selected_indicator_count,
            selected_note_count=stored.selected_note_count,
            selected_workflow_count=stored.selected_workflow_count,
            packet_character_count=stored.packet_character_count,
            output_character_count=stored.output_character_count,
            status=stored.status,
            error_code=stored.error_code,
        )
        return stored

    def save_note(self, analysis_id: str) -> IncidentAnalysisResult:
        """Persist one verbatim AI-assisted note after reauthorization."""
        self._assert_analysis_enabled()
        self._assert_note_save_enabled()
        result = self._analysis_repository.get_result(analysis_id)
        if result is None:
            raise IncidentAnalysisValidationError(
                f"No completed analysis found with ID '{analysis_id}'."
            )
        if result.status != "completed":
            raise IncidentAnalysisValidationError(
                "Only completed analyses can be saved as notes."
            )
        if result.note_saved:
            raise IncidentAnalysisConflictError(
                "That analysis note was already saved."
            )

        prepared = self._analysis_repository.get_prepared(result.analysis_id)
        if prepared is None:
            raise IncidentAnalysisConflictError(
                "Prepared analysis identity is no longer available."
            )
        if (
            prepared.incident_id != result.incident_id
            or prepared.scope_id != result.scope_id
            or prepared.packet_fingerprint != result.packet_fingerprint
            or prepared.analysis_kind != result.analysis_kind
        ):
            raise IncidentAnalysisConflictError(
                "Analysis identity or packet fingerprint mismatch."
            )

        try:
            self._authorize(result.scope_id, result.incident_id)
        except IncidentAnalysisAuthorizationError:
            self._audit_log.append(
                analysis_id=result.analysis_id,
                incident_id=result.incident_id,
                scope_id=result.scope_id,
                analysis_kind=result.analysis_kind,
                action="analysis_note_save_denied",
                selected_event_count=result.selected_event_count,
                selected_indicator_count=result.selected_indicator_count,
                selected_note_count=result.selected_note_count,
                selected_workflow_count=result.selected_workflow_count,
                packet_character_count=result.packet_character_count,
                output_character_count=result.output_character_count,
                status="denied",
                error_code="authorization_denied",
            )
            raise

        note_text = build_analysis_note_text(result)
        if len(note_text) > MAX_NOTE_TEXT_LENGTH:
            self._audit_log.append(
                analysis_id=result.analysis_id,
                incident_id=result.incident_id,
                scope_id=result.scope_id,
                analysis_kind=result.analysis_kind,
                action="analysis_note_save_denied",
                selected_event_count=result.selected_event_count,
                selected_indicator_count=result.selected_indicator_count,
                selected_note_count=result.selected_note_count,
                selected_workflow_count=result.selected_workflow_count,
                packet_character_count=result.packet_character_count,
                output_character_count=result.output_character_count,
                status="denied",
                error_code="note_too_long",
            )
            raise IncidentAnalysisValidationError(
                "The analysis note exceeds the maximum note length."
            )

        note = create_incident_note(
            incident_id=result.incident_id,
            author=AI_INCIDENT_ANALYSIS_NOTE_AUTHOR,
            text=note_text,
            note_type=AI_INCIDENT_ANALYSIS_NOTE_TYPE,
            tags=(AI_INCIDENT_ANALYSIS_NOTE_TAG,),
        )
        saved = self._incident_repository.add_note(note)
        updated = self._analysis_repository.mark_note_saved(
            result.analysis_id,
            note_id=saved.note_id,
        )
        self._audit_log.append(
            analysis_id=updated.analysis_id,
            incident_id=updated.incident_id,
            scope_id=updated.scope_id,
            analysis_kind=updated.analysis_kind,
            action="analysis_note_saved",
            selected_event_count=updated.selected_event_count,
            selected_indicator_count=updated.selected_indicator_count,
            selected_note_count=updated.selected_note_count,
            selected_workflow_count=updated.selected_workflow_count,
            packet_character_count=updated.packet_character_count,
            output_character_count=updated.output_character_count,
            status="saved",
            saved_note_id=updated.saved_note_id,
        )
        return updated

    def get_prepared(self, analysis_id: str) -> IncidentAnalysisRequest | None:
        """Return one prepared analysis by ID."""
        return self._analysis_repository.get_prepared(analysis_id)

    def get_result(self, analysis_id: str) -> IncidentAnalysisResult | None:
        """Return one analysis result by ID."""
        return self._analysis_repository.get_result(analysis_id)

    def analysis_count(self) -> int:
        """Return retained analysis count for status reporting."""
        return self._analysis_repository.analysis_count()

    def _store_failed(
        self,
        prepared: IncidentAnalysisRequest,
        *,
        error_code: str,
        advisory_output: str,
    ) -> IncidentAnalysisResult:
        result = IncidentAnalysisResult(
            analysis_id=prepared.analysis_id,
            incident_id=prepared.incident_id,
            scope_id=prepared.scope_id,
            analysis_kind=prepared.analysis_kind,
            packet_fingerprint=prepared.packet_fingerprint,
            boundary_token=prepared.boundary_token,
            advisory_output=advisory_output,
            created_at=prepared.created_at,
            completed_at=utc_timestamp(),
            status="failed",
            error_code=error_code,
            note_saved=False,
            saved_note_id=None,
            selected_event_count=prepared.packet.selected_event_count,
            selected_indicator_count=prepared.packet.selected_indicator_count,
            selected_note_count=prepared.packet.selected_note_count,
            selected_workflow_count=prepared.packet.selected_workflow_count,
            packet_character_count=prepared.packet.character_count,
            output_character_count=len(advisory_output),
        )
        stored = self._analysis_repository.store_result(result)
        self._audit_log.append(
            analysis_id=stored.analysis_id,
            incident_id=stored.incident_id,
            scope_id=stored.scope_id,
            analysis_kind=stored.analysis_kind,
            action="analysis_failed",
            selected_event_count=stored.selected_event_count,
            selected_indicator_count=stored.selected_indicator_count,
            selected_note_count=stored.selected_note_count,
            selected_workflow_count=stored.selected_workflow_count,
            packet_character_count=stored.packet_character_count,
            output_character_count=stored.output_character_count,
            status="failed",
            error_code=stored.error_code,
        )
        return stored

    def _authorize(self, scope_id: str, incident_id: str) -> None:
        incident = self._incident_repository.get_incident(incident_id)
        if incident is None:
            raise IncidentAnalysisAuthorizationError(
                "Linked incident was not found or is no longer authorized."
            )
        scope = self._tool_repository.get_scope(scope_id)
        if scope is None:
            raise IncidentAnalysisAuthorizationError(
                "Authorized scope was not found."
            )
        try:
            assert_incident_authorized(scope, incident_id)
        except Exception as error:
            raise IncidentAnalysisAuthorizationError(str(error)) from error

    def _assert_analysis_enabled(self) -> None:
        if not AI_INCIDENT_ANALYSIS_ENABLED:
            raise IncidentAnalysisDisabledError(
                "Incident AI analysis is disabled."
            )

    def _assert_note_save_enabled(self) -> None:
        if not AI_INCIDENT_ANALYSIS_ENABLED or not AI_INCIDENT_NOTE_SAVE_ENABLED:
            raise IncidentAnalysisNoteSaveDisabledError(
                "Incident AI analysis note saving is disabled."
            )


def build_analysis_note_text(result: IncidentAnalysisResult) -> str:
    """Build the fixed provenance banner plus verbatim advisory output."""
    banner = (
        "[AI-assisted draft — human-reviewed and approved. "
        f"Analysis ID: {result.analysis_id}]"
    )
    return f"{banner}\n\n{result.advisory_output}"
