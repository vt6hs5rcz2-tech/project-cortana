"""Slash commands for incident-scoped evidence search and AI-off context display."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.evidence_store import EvidenceStore
from src.incident_analysis_common import (
    ANALYSIS_NONE_SELECTION,
    IncidentAnalysisAuthorizationError,
    IncidentAnalysisLimitError,
    IncidentAnalysisValidationError,
)
from src.incident_analysis_packet import build_bounded_incident_context_for_display
from src.incident_evidence_search import (
    IncidentEvidenceSearchError,
    authorize_scope_for_incident_context,
    create_evidence_text_search_request,
)
from src.incident_repository import IncidentRepository, IncidentStorageError
from src.command_argument_utils import extract_command_argument
from src.security_commands import split_delimited_fields
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolIdError,
    ToolAuthorizationError,
    ToolValidationError,
    validate_uuid,
)
from src.tool_registry import ToolRegistry
from src.tool_repository import ToolControlRepository, ToolStorageError
from src.workflow_repository import WorkflowRunRepository

logger = logging.getLogger("ProjectCortana")

COMMAND_EVIDENCE_SEARCH = "evidence-search"
COMMAND_INCIDENT_CONTEXT = "incident-context"

INCIDENT_EVIDENCE_COMMAND_NAMES = frozenset(
    {
        COMMAND_EVIDENCE_SEARCH,
        COMMAND_INCIDENT_CONTEXT,
    }
)

EVIDENCE_SEARCH_USAGE = (
    "Cortana: Usage: /evidence-search <incident-id> | <evidence-id> | "
    "<scope-id> | <query> [| <max-matches>]"
)
INCIDENT_CONTEXT_USAGE = (
    "Cortana: Usage: /incident-context <incident-id> | <scope-id> | "
    "<event-ids> | <indicator-ids> | <note-ids> | <workflow-run-ids>"
)


@dataclass(frozen=True)
class IncidentEvidenceCommandContext:
    """Inputs for Milestone 17 incident-evidence command handlers."""

    message: str
    incident_repository: IncidentRepository
    evidence_store: EvidenceStore
    tool_registry: ToolRegistry
    tool_repository: ToolControlRepository
    workflow_run_repository: WorkflowRunRepository


@dataclass(frozen=True)
class IncidentEvidenceCommandResult:
    """Local message returned by a Milestone 17 command handler."""

    message: str


IncidentEvidenceCommandHandler = Callable[
    [IncidentEvidenceCommandContext],
    IncidentEvidenceCommandResult,
]


def handle_incident_evidence_command(
    command_name: str,
    context: IncidentEvidenceCommandContext,
) -> IncidentEvidenceCommandResult | None:
    """Dispatch a Milestone 17 command, or return None when unknown."""
    handler = INCIDENT_EVIDENCE_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    return handler(context)


def _parse_id_list(field: str, *, field_name: str) -> tuple[str, ...]:
    cleaned = field.strip()
    if cleaned == ANALYSIS_NONE_SELECTION:
        return ()
    if not cleaned:
        raise IncidentAnalysisValidationError(
            f"{field_name} cannot be blank. Use '-' for none."
        )
    parts = [part.strip() for part in cleaned.split(",")]
    if any(not part for part in parts):
        raise IncidentAnalysisValidationError(
            f"{field_name} contains a blank ID."
        )
    ids = [validate_uuid(part, field_name=field_name) for part in parts]
    if len(ids) != len(set(ids)):
        raise IncidentAnalysisValidationError(
            f"{field_name} contains duplicate IDs."
        )
    return tuple(ids)


def _handle_evidence_search(
    context: IncidentEvidenceCommandContext,
) -> IncidentEvidenceCommandResult:
    argument = extract_command_argument(context.message).strip()
    fields = split_delimited_fields(argument, 4)
    max_matches: int | None = None
    if fields is None:
        fields = split_delimited_fields(argument, 5)
        if fields is None:
            return IncidentEvidenceCommandResult(message=EVIDENCE_SEARCH_USAGE)
        incident_id, evidence_id, scope_id, query, max_matches_text = fields
        try:
            max_matches = int(max_matches_text.strip())
        except ValueError:
            return IncidentEvidenceCommandResult(message=EVIDENCE_SEARCH_USAGE)
    else:
        incident_id, evidence_id, scope_id, query = fields

    try:
        created = create_evidence_text_search_request(
            incident_repository=context.incident_repository,
            evidence_store=context.evidence_store,
            tool_registry=context.tool_registry,
            tool_repository=context.tool_repository,
            incident_id=incident_id,
            evidence_id=evidence_id,
            scope_id=scope_id,
            query=query,
            max_matches=max_matches,
        )
    except ToolAuthorizationError:
        return IncidentEvidenceCommandResult(
            message=(
                "Cortana: That incident, evidence path, or tool is not "
                "authorized for the supplied scope."
            )
        )
    except IncidentEvidenceSearchError as error:
        message = str(error)
        if "was not found" in message or "not linked" in message:
            return IncidentEvidenceCommandResult(message=f"Cortana: {message}")
        if "Stored evidence" in message:
            return IncidentEvidenceCommandResult(message=message)
        return IncidentEvidenceCommandResult(message=EVIDENCE_SEARCH_USAGE)
    except (
        ToolValidationError,
        BlankToolFieldError,
        InvalidToolIdError,
    ):
        return IncidentEvidenceCommandResult(message=EVIDENCE_SEARCH_USAGE)
    except ToolStorageError as error:
        return IncidentEvidenceCommandResult(message=error.user_message)
    except IncidentStorageError as error:
        return IncidentEvidenceCommandResult(message=error.user_message)

    logger.info(
        "Evidence text-search request created request_id=%s incident_id=%s "
        "evidence_id=%s",
        created.request_id,
        created.incident_id,
        created.evidence_id,
    )
    return IncidentEvidenceCommandResult(
        message=(
            "Cortana: Evidence text-search request created "
            f"({created.request_id}).\n"
            f"  Tool: {created.tool_id}\n"
            f"  Status: {created.request_status}\n"
            f"  Fingerprint: {created.fingerprint_prefix}\n"
            "Continue with /tool-dry-run, /tool-approve, and /tool-run."
        )
    )


def _handle_incident_context(
    context: IncidentEvidenceCommandContext,
) -> IncidentEvidenceCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 6)
    if fields is None:
        return IncidentEvidenceCommandResult(message=INCIDENT_CONTEXT_USAGE)

    (
        incident_id,
        scope_id,
        event_ids_text,
        indicator_ids_text,
        note_ids_text,
        workflow_ids_text,
    ) = fields
    try:
        authorize_scope_for_incident_context(
            tool_repository=context.tool_repository,
            scope_id=scope_id,
            incident_id=incident_id,
        )
        selected_event_ids = _parse_id_list(
            event_ids_text,
            field_name="Event IDs",
        )
        selected_indicator_ids = _parse_id_list(
            indicator_ids_text,
            field_name="Indicator IDs",
        )
        selected_note_ids = _parse_id_list(
            note_ids_text,
            field_name="Note IDs",
        )
        selected_workflow_run_ids = _parse_id_list(
            workflow_ids_text,
            field_name="Workflow run IDs",
        )
        packet = build_bounded_incident_context_for_display(
            incident_repository=context.incident_repository,
            workflow_run_repository=context.workflow_run_repository,
            incident_id=incident_id.strip(),
            selected_event_ids=selected_event_ids,
            selected_indicator_ids=selected_indicator_ids,
            selected_note_ids=selected_note_ids,
            selected_workflow_run_ids=selected_workflow_run_ids,
        )
    except (
        IncidentEvidenceSearchError,
        IncidentAnalysisAuthorizationError,
        ToolAuthorizationError,
    ):
        return IncidentEvidenceCommandResult(
            message=(
                "Cortana: That incident or scope could not be authorized "
                "for bounded context display."
            )
        )
    except IncidentAnalysisLimitError:
        return IncidentEvidenceCommandResult(
            message=(
                "Cortana: The bounded incident context exceeds configured limits."
            )
        )
    except (
        IncidentAnalysisValidationError,
        BlankToolFieldError,
        InvalidToolIdError,
        ToolValidationError,
    ):
        return IncidentEvidenceCommandResult(message=INCIDENT_CONTEXT_USAGE)
    except IncidentStorageError as error:
        return IncidentEvidenceCommandResult(message=error.user_message)

    logger.info(
        "Bounded incident context assembled incident_id=%s chars=%s",
        packet.incident_id,
        packet.character_count,
    )
    return IncidentEvidenceCommandResult(
        message=(
            "Cortana: Bounded incident context (AI-off):\n"
            f"  Incident ID: {packet.incident_id}\n"
            f"  Selected events: {packet.selected_event_count}\n"
            f"  Selected indicators: {packet.selected_indicator_count}\n"
            f"  Selected notes: {packet.selected_note_count}\n"
            f"  Selected workflow summaries: {packet.selected_workflow_count}\n"
            f"  Selected tool summaries: {packet.selected_tool_summary_count}\n"
            f"  Packet characters: {packet.character_count}\n"
            f"  Packet fingerprint: {packet.packet_fingerprint[:12]}\n"
            "  omitted_evidence: true\n"
            "  omitted_custody: true\n"
            "  omitted_structured_tool_data: true\n"
            "This display does not call AI. Use /incident-analysis-prepare "
            "only when AI analysis is explicitly enabled and intended."
        )
    )


INCIDENT_EVIDENCE_COMMAND_HANDLERS: dict[str, IncidentEvidenceCommandHandler] = {
    COMMAND_EVIDENCE_SEARCH: _handle_evidence_search,
    COMMAND_INCIDENT_CONTEXT: _handle_incident_context,
}
