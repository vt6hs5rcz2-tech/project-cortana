"""Local slash-command handlers for Milestone 12 incident AI analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.ai_service import OpenAIClient
from src.config import (
    AI_INCIDENT_ANALYSIS_ENABLED,
    AI_INCIDENT_ANALYSIS_NOTE_AUTHOR,
    AI_INCIDENT_ANALYSIS_NOTE_TAG,
)
from src.incident_analysis_audit import InMemoryIncidentAnalysisAuditLog
from src.incident_analysis_common import (
    ANALYSIS_NONE_SELECTION,
    IncidentAnalysisAuthorizationError,
    IncidentAnalysisConflictError,
    IncidentAnalysisDisabledError,
    IncidentAnalysisLimitError,
    IncidentAnalysisNoteSaveDisabledError,
    IncidentAnalysisValidationError,
    validate_analysis_kind,
)
from src.incident_analysis_repository import InMemoryIncidentAnalysisRepository
from src.incident_analysis_service import IncidentAnalysisService
from src.incident_repository import IncidentRepository, IncidentStorageError
from src.command_argument_utils import extract_command_argument
from src.user_facing import cortana_domain_message
from src.security_commands import split_delimited_fields
from src.settings import Settings
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolIdError,
    ToolAuthorizationError,
    validate_uuid,
)
from src.tool_repository import ToolControlRepository
from src.workflow_repository import WorkflowRunRepository

logger = logging.getLogger("ProjectCortana")

COMMAND_INCIDENT_ANALYSIS_PREPARE = "incident-analysis-prepare"
COMMAND_INCIDENT_ANALYSIS_RUN = "incident-analysis-run"
COMMAND_INCIDENT_ANALYSIS_SHOW = "incident-analysis-show"
COMMAND_INCIDENT_ANALYSIS_SAVE_NOTE = "incident-analysis-save-note"

INCIDENT_ANALYSIS_COMMAND_NAMES = frozenset(
    {
        COMMAND_INCIDENT_ANALYSIS_PREPARE,
        COMMAND_INCIDENT_ANALYSIS_RUN,
        COMMAND_INCIDENT_ANALYSIS_SHOW,
        COMMAND_INCIDENT_ANALYSIS_SAVE_NOTE,
    }
)

ANALYSIS_DISABLED_MESSAGE = (
    "Cortana: Incident AI analysis is disabled. "
    "Set AI_INCIDENT_ANALYSIS_ENABLED to enable this capability."
)
NOTE_SAVE_DISABLED_MESSAGE = (
    "Cortana: Incident AI analysis note saving is disabled. "
    "Set AI_INCIDENT_ANALYSIS_ENABLED and AI_INCIDENT_NOTE_SAVE_ENABLED "
    "to enable this capability."
)
PREPARE_USAGE = (
    "Cortana: Usage: /incident-analysis-prepare <kind> | <incident-id> | "
    "<scope-id> | <event-ids> | <indicator-ids> | <note-ids> | "
    "<workflow-run-ids>"
)
RUN_USAGE = "Cortana: Usage: /incident-analysis-run <analysis-id>"
SHOW_USAGE = "Cortana: Usage: /incident-analysis-show <analysis-id>"
SAVE_NOTE_USAGE = (
    "Cortana: Usage: /incident-analysis-save-note <analysis-id>"
)


@dataclass(frozen=True)
class IncidentAnalysisCommandContext:
    """Inputs available to Milestone 12 incident analysis command handlers."""

    message: str
    settings: Settings
    incident_repository: IncidentRepository
    tool_repository: ToolControlRepository
    workflow_run_repository: WorkflowRunRepository
    analysis_repository: InMemoryIncidentAnalysisRepository
    analysis_audit_log: InMemoryIncidentAnalysisAuditLog
    client: OpenAIClient | None


@dataclass(frozen=True)
class IncidentAnalysisCommandResult:
    """Local message returned by an incident analysis command handler."""

    message: str


IncidentAnalysisCommandHandler = Callable[
    [IncidentAnalysisCommandContext],
    IncidentAnalysisCommandResult,
]


def handle_incident_analysis_command(
    command_name: str,
    context: IncidentAnalysisCommandContext,
) -> IncidentAnalysisCommandResult | None:
    """Dispatch an incident analysis command, or return None when unknown."""
    handler = INCIDENT_ANALYSIS_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    return handler(context)


def _service(context: IncidentAnalysisCommandContext) -> IncidentAnalysisService:
    return IncidentAnalysisService(
        incident_repository=context.incident_repository,
        tool_repository=context.tool_repository,
        workflow_run_repository=context.workflow_run_repository,
        analysis_repository=context.analysis_repository,
        audit_log=context.analysis_audit_log,
        settings=context.settings,
        client=context.client,
    )


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


def _handle_prepare(
    context: IncidentAnalysisCommandContext,
) -> IncidentAnalysisCommandResult:
    try:
        argument = extract_command_argument(context.message).strip()
        fields = split_delimited_fields(argument, 7)
        if fields is None:
            return IncidentAnalysisCommandResult(message=PREPARE_USAGE)
        (
            kind_text,
            incident_id,
            scope_id,
            event_ids_text,
            indicator_ids_text,
            note_ids_text,
            workflow_ids_text,
        ) = fields
        kind = validate_analysis_kind(kind_text)
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
        prepared = _service(context).prepare(
            analysis_kind=kind,
            incident_id=incident_id,
            scope_id=scope_id,
            selected_event_ids=selected_event_ids,
            selected_indicator_ids=selected_indicator_ids,
            selected_note_ids=selected_note_ids,
            selected_workflow_run_ids=selected_workflow_run_ids,
        )
    except IncidentAnalysisDisabledError:
        return IncidentAnalysisCommandResult(message=ANALYSIS_DISABLED_MESSAGE)
    except (
        IncidentAnalysisAuthorizationError,
        ToolAuthorizationError,
    ):
        return IncidentAnalysisCommandResult(
            message=(
                "Cortana: That incident or scope could not be authorized "
                "for analysis."
            )
        )
    except IncidentAnalysisLimitError:
        return IncidentAnalysisCommandResult(
            message=(
                "Cortana: The incident analysis packet or selection exceeds "
                "configured limits."
            )
        )
    except (
        IncidentAnalysisValidationError,
        BlankToolFieldError,
        InvalidToolIdError,
    ):
        return IncidentAnalysisCommandResult(message=PREPARE_USAGE)
    except IncidentStorageError as error:
        return IncidentAnalysisCommandResult(message=error.user_message)

    packet = prepared.packet
    return IncidentAnalysisCommandResult(
        message=(
            "Cortana: Incident analysis prepared "
            f"({prepared.analysis_id}).\n"
            f"  Incident ID: {prepared.incident_id}\n"
            f"  Analysis kind: {prepared.analysis_kind}\n"
            f"  Selected events: {packet.selected_event_count}\n"
            f"  Selected indicators: {packet.selected_indicator_count}\n"
            f"  Selected notes: {packet.selected_note_count}\n"
            f"  Selected workflow summaries: {packet.selected_workflow_count}\n"
            f"  Selected tool summaries: {packet.selected_tool_summary_count}\n"
            f"  Packet characters: {packet.character_count}\n"
            "  WARNING: Selected free text will be sent to the configured "
            "AI service when you run this analysis.\n"
            "Use /incident-analysis-run <analysis-id> to confirm and send."
        )
    )


def _handle_run(
    context: IncidentAnalysisCommandContext,
) -> IncidentAnalysisCommandResult:
    try:
        analysis_id = extract_command_argument(context.message).strip()
        if not analysis_id:
            return IncidentAnalysisCommandResult(message=RUN_USAGE)
        result = _service(context).run(analysis_id)
    except IncidentAnalysisDisabledError:
        return IncidentAnalysisCommandResult(message=ANALYSIS_DISABLED_MESSAGE)
    except IncidentAnalysisAuthorizationError:
        return IncidentAnalysisCommandResult(
            message=(
                "Cortana: That incident or scope could not be authorized "
                "for analysis."
            )
        )
    except IncidentAnalysisConflictError as error:
        return IncidentAnalysisCommandResult(
            message=cortana_domain_message(
                error,
                fallback="Cortana: I couldn't run that incident analysis.",
            )
        )
    except IncidentAnalysisValidationError as error:
        message = str(error)
        if "No prepared analysis" in message:
            return IncidentAnalysisCommandResult(
                message=cortana_domain_message(
                    error,
                    fallback="Cortana: I couldn't run that incident analysis.",
                )
            )
        return IncidentAnalysisCommandResult(message=RUN_USAGE)
    except IncidentStorageError as error:
        return IncidentAnalysisCommandResult(message=error.user_message)
    except Exception as error:
        logger.error(
            "Incident analysis run failed error_type=%s",
            type(error).__name__,
        )
        return IncidentAnalysisCommandResult(
            message="Cortana: Incident AI analysis could not be completed."
        )

    label = "completed" if result.status == "completed" else "failed"
    return IncidentAnalysisCommandResult(
        message=(
            f"Cortana: Incident analysis {label} ({result.analysis_id}). "
            f"Kind: {result.analysis_kind}. "
            "Output is advisory and untrusted.\n\n"
            f"{result.advisory_output}"
        )
    )


def _handle_show(
    context: IncidentAnalysisCommandContext,
) -> IncidentAnalysisCommandResult:
    try:
        if not AI_INCIDENT_ANALYSIS_ENABLED:
            raise IncidentAnalysisDisabledError(ANALYSIS_DISABLED_MESSAGE)
        analysis_id = extract_command_argument(context.message).strip()
        if not analysis_id:
            return IncidentAnalysisCommandResult(message=SHOW_USAGE)
        service = _service(context)
        result = service.get_result(analysis_id)
        prepared = service.get_prepared(analysis_id)
        if result is None and prepared is None:
            return IncidentAnalysisCommandResult(
                message=(
                    f"Cortana: No incident analysis found with ID "
                    f"'{analysis_id}'."
                )
            )
    except IncidentAnalysisDisabledError:
        return IncidentAnalysisCommandResult(message=ANALYSIS_DISABLED_MESSAGE)

    if result is not None:
        note_state = "saved" if result.note_saved else "not saved"
        return IncidentAnalysisCommandResult(
            message=(
                "Cortana: Incident analysis\n"
                f"  Analysis ID: {result.analysis_id}\n"
                f"  Incident ID: {result.incident_id}\n"
                f"  Scope ID: {result.scope_id}\n"
                f"  Kind: {result.analysis_kind}\n"
                f"  Status: {result.status}\n"
                f"  Note status: {note_state}\n"
                f"  Packet fingerprint: {result.packet_fingerprint[:16]}...\n\n"
                "Advisory output (untrusted):\n"
                f"{result.advisory_output}"
            )
        )
    assert prepared is not None
    packet = prepared.packet
    return IncidentAnalysisCommandResult(
        message=(
            "Cortana: Incident analysis (prepared, not yet run)\n"
            f"  Analysis ID: {prepared.analysis_id}\n"
            f"  Incident ID: {prepared.incident_id}\n"
            f"  Scope ID: {prepared.scope_id}\n"
            f"  Kind: {prepared.analysis_kind}\n"
            f"  Selected events: {packet.selected_event_count}\n"
            f"  Selected indicators: {packet.selected_indicator_count}\n"
            f"  Selected notes: {packet.selected_note_count}\n"
            f"  Selected workflow summaries: {packet.selected_workflow_count}\n"
            f"  Packet characters: {packet.character_count}\n"
            "  WARNING: Selected free text will be sent to the configured "
            "AI service when you run this analysis."
        )
    )


def _handle_save_note(
    context: IncidentAnalysisCommandContext,
) -> IncidentAnalysisCommandResult:
    try:
        analysis_id = extract_command_argument(context.message).strip()
        if not analysis_id:
            return IncidentAnalysisCommandResult(message=SAVE_NOTE_USAGE)
        updated = _service(context).save_note(analysis_id)
    except IncidentAnalysisDisabledError:
        return IncidentAnalysisCommandResult(message=ANALYSIS_DISABLED_MESSAGE)
    except IncidentAnalysisNoteSaveDisabledError:
        return IncidentAnalysisCommandResult(message=NOTE_SAVE_DISABLED_MESSAGE)
    except IncidentAnalysisAuthorizationError:
        return IncidentAnalysisCommandResult(
            message=(
                "Cortana: That incident or scope could not be authorized "
                "for note saving."
            )
        )
    except IncidentAnalysisConflictError as error:
        return IncidentAnalysisCommandResult(
            message=cortana_domain_message(
                error,
                fallback="Cortana: I couldn't save that analysis note.",
            )
        )
    except IncidentAnalysisValidationError as error:
        message = str(error)
        if "No completed analysis" in message or "maximum note length" in message:
            return IncidentAnalysisCommandResult(
                message=cortana_domain_message(
                    error,
                    fallback="Cortana: I couldn't save that analysis note.",
                )
            )
        return IncidentAnalysisCommandResult(message=SAVE_NOTE_USAGE)
    except IncidentStorageError as error:
        return IncidentAnalysisCommandResult(message=error.user_message)

    return IncidentAnalysisCommandResult(
        message=(
            f"Cortana: Saved AI-assisted analysis note ({updated.saved_note_id}) "
            f"on incident {updated.incident_id}. "
            f"Author: {AI_INCIDENT_ANALYSIS_NOTE_AUTHOR}. "
            f"Tag: {AI_INCIDENT_ANALYSIS_NOTE_TAG}."
        )
    )


INCIDENT_ANALYSIS_COMMAND_HANDLERS: dict[
    str,
    IncidentAnalysisCommandHandler,
] = {
    COMMAND_INCIDENT_ANALYSIS_PREPARE: _handle_prepare,
    COMMAND_INCIDENT_ANALYSIS_RUN: _handle_run,
    COMMAND_INCIDENT_ANALYSIS_SHOW: _handle_show,
    COMMAND_INCIDENT_ANALYSIS_SAVE_NOTE: _handle_save_note,
}
