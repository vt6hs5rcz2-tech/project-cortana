"""Local slash-command handlers for Milestone 10 workflow orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.config import MAX_WORKFLOW_LIST_PREVIEW_CHARS
from src.incident_repository import IncidentRepository
from src.security_commands import extract_command_argument, split_delimited_fields
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolIdError,
    ToolAuthorizationError,
    ToolPolicyError,
    ToolValidationError,
)
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry
from src.tool_repository import ToolControlRepository, ToolStorageError
from src.workflow_builtins import build_default_workflow_registry
from src.workflow_common import (
    WorkflowAuthorizationError,
    WorkflowPolicyError,
    WorkflowStorageError,
    WorkflowValidationError,
)
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import (
    InMemoryWorkflowRunRepository,
    WorkflowRunRepository,
)
from src.workflow_request import create_workflow_run_request

logger = logging.getLogger("ProjectCortana")

COMMAND_PLAYBOOKS = "playbooks"
COMMAND_PLAYBOOK_SHOW = "playbook-show"
COMMAND_PLAYBOOK_RUN = "playbook-run"
COMMAND_PLAYBOOK_STATUS = "playbook-status"

WORKFLOW_COMMAND_NAMES = frozenset(
    {
        COMMAND_PLAYBOOKS,
        COMMAND_PLAYBOOK_SHOW,
        COMMAND_PLAYBOOK_RUN,
        COMMAND_PLAYBOOK_STATUS,
    }
)

PLAYBOOK_RUN_USAGE = (
    "Cortana: Usage: /playbook-run <name> | <scope-id> "
    "or /playbook-run <name> --execute | <scope-id>"
)
PLAYBOOKS_EMPTY = "Cortana: No defensive playbooks are registered."
EXECUTE_TOKEN = "--execute"


@dataclass(frozen=True)
class WorkflowCommandContext:
    """Inputs available to Milestone 10 workflow command handlers."""

    message: str
    tool_registry: ToolRegistry
    tool_repository: ToolControlRepository
    tool_executor: DefensiveToolExecutor
    incident_repository: IncidentRepository
    workflow_registry: WorkflowRegistry
    workflow_run_repository: WorkflowRunRepository
    workflow_executor: WorkflowExecutor


@dataclass(frozen=True)
class WorkflowCommandResult:
    """Local message returned by a workflow command handler."""

    message: str


WorkflowCommandHandler = Callable[[WorkflowCommandContext], WorkflowCommandResult]


def handle_workflow_command(
    command_name: str,
    context: WorkflowCommandContext,
) -> WorkflowCommandResult | None:
    """Dispatch a Milestone 10 workflow command, or return None when unknown."""
    handler = WORKFLOW_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    return handler(context)


def create_default_workflow_services(
    *,
    tool_registry: ToolRegistry,
    tool_repository: ToolControlRepository,
    tool_executor: DefensiveToolExecutor,
) -> tuple[WorkflowRegistry, WorkflowRunRepository, WorkflowExecutor]:
    """Create default in-memory workflow services with built-in playbooks."""
    workflow_registry = build_default_workflow_registry(tool_registry=tool_registry)
    run_repository: WorkflowRunRepository = InMemoryWorkflowRunRepository()
    workflow_executor = WorkflowExecutor(
        workflow_registry=workflow_registry,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        tool_repository=tool_repository,
        run_repository=run_repository,
    )
    return workflow_registry, run_repository, workflow_executor


def bounded_preview(text: str) -> str:
    """Return a bounded single-line preview for list command output."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= MAX_WORKFLOW_LIST_PREVIEW_CHARS:
        return cleaned
    return f"{cleaned[:MAX_WORKFLOW_LIST_PREVIEW_CHARS]}..."


def _handle_playbooks(context: WorkflowCommandContext) -> WorkflowCommandResult:
    playbooks = context.workflow_registry.list_enabled()
    if not playbooks:
        return WorkflowCommandResult(message=PLAYBOOKS_EMPTY)
    lines = ["Cortana: Enabled defensive playbooks:"]
    for playbook in playbooks:
        lines.append(
            f"  - {playbook.name} v{playbook.version} "
            f"steps={len(playbook.steps)} "
            f"{bounded_preview(playbook.description)}"
        )
    return WorkflowCommandResult(message="\n".join(lines))


def _handle_playbook_show(context: WorkflowCommandContext) -> WorkflowCommandResult:
    name = extract_command_argument(context.message).strip()
    if not name:
        return WorkflowCommandResult(
            message=(
                "Cortana: Please provide a playbook name. "
                "Usage: /playbook-show <name>"
            )
        )
    try:
        playbook = context.workflow_registry.require(name)
    except (WorkflowValidationError, InvalidToolIdError, ToolValidationError):
        return WorkflowCommandResult(
            message=f"Cortana: No registered playbook found with name '{name}'."
        )

    lines = [
        "Cortana: Playbook details",
        f"  Name: {playbook.name}",
        f"  Version: {playbook.version}",
        f"  Enabled: {playbook.enabled}",
        f"  Steps: {len(playbook.steps)}",
        f"  Description: {bounded_preview(playbook.description)}",
        "  Step list:",
    ]
    for step in playbook.steps:
        parameter_names = ", ".join(sorted(step.static_parameters)) or "(none)"
        lines.append(
            f"    - [{step.position}] {step.step_id} -> {step.tool_id} "
            f"params={parameter_names}"
        )
    return WorkflowCommandResult(message="\n".join(lines))


def _handle_playbook_run(context: WorkflowCommandContext) -> WorkflowCommandResult:
    argument = extract_command_argument(context.message).strip()
    if not argument:
        return WorkflowCommandResult(message=PLAYBOOK_RUN_USAGE)

    fields = split_delimited_fields(argument, 2)
    if fields is None:
        return WorkflowCommandResult(message=PLAYBOOK_RUN_USAGE)

    name_field, scope_id = fields
    name_tokens = name_field.split()
    dry_run = True
    if len(name_tokens) == 1:
        playbook_name = name_tokens[0]
    elif len(name_tokens) == 2 and name_tokens[1] == EXECUTE_TOKEN:
        playbook_name = name_tokens[0]
        dry_run = False
    else:
        return WorkflowCommandResult(message=PLAYBOOK_RUN_USAGE)

    try:
        context.workflow_registry.require(playbook_name)
        scope = context.tool_repository.get_scope(scope_id.strip())
        if scope is None:
            return WorkflowCommandResult(
                message=(
                    f"Cortana: No saved scope found with ID '{scope_id.strip()}'."
                )
            )
        run_request = create_workflow_run_request(
            playbook_name=playbook_name,
            scope_id=scope.scope_id,
            dry_run=dry_run,
        )
        result = context.workflow_executor.run(run_request)
    except (
        BlankToolFieldError,
        InvalidToolIdError,
        ToolAuthorizationError,
        ToolPolicyError,
        ToolValidationError,
        WorkflowAuthorizationError,
        WorkflowPolicyError,
        WorkflowValidationError,
    ):
        return WorkflowCommandResult(message=PLAYBOOK_RUN_USAGE)
    except (ToolStorageError, WorkflowStorageError) as error:
        return WorkflowCommandResult(message=error.user_message)

    mode = "dry-run" if result.dry_run else "execute"
    logger.info(
        "Workflow run finished run_id=%s playbook=%s status=%s dry_run=%s",
        result.run_id,
        result.playbook_name,
        result.status,
        result.dry_run,
    )
    return WorkflowCommandResult(
        message=(
            f"Cortana: Playbook run {result.status} ({result.run_id}). "
            f"Playbook: {result.playbook_name} v{result.playbook_version}. "
            f"Mode: {mode}. Steps recorded: {len(result.step_results)}."
        )
    )


def _handle_playbook_status(context: WorkflowCommandContext) -> WorkflowCommandResult:
    run_id = extract_command_argument(context.message).strip()
    if not run_id:
        return WorkflowCommandResult(
            message=(
                "Cortana: Please provide a run ID. "
                "Usage: /playbook-status <run-id>"
            )
        )
    try:
        run = context.workflow_run_repository.get_run(run_id)
    except WorkflowStorageError as error:
        return WorkflowCommandResult(message=error.user_message)
    if run is None:
        return WorkflowCommandResult(
            message=f"Cortana: No workflow run found with ID '{run_id}'."
        )

    lines = [
        "Cortana: Playbook run status",
        f"  Run ID: {run.run_id}",
        f"  Playbook: {run.playbook_name}",
        f"  Version: {run.playbook_version}",
        f"  Status: {run.status}",
        f"  Dry-run: {run.dry_run}",
        f"  Scope ID: {run.scope_id}",
        f"  Steps recorded: {len(run.step_results)}",
    ]
    if run.error_code is not None:
        lines.append(f"  Error code: {run.error_code}")
    for step in run.step_results:
        outcome = (
            step.tool_result.outcome
            if step.tool_result is not None
            else step.status
        )
        lines.append(
            f"    - [{step.position}] {step.step_id} "
            f"tool={step.tool_id} status={step.status} outcome={outcome}"
        )
    return WorkflowCommandResult(message="\n".join(lines))


WORKFLOW_COMMAND_HANDLERS: dict[str, WorkflowCommandHandler] = {
    COMMAND_PLAYBOOKS: _handle_playbooks,
    COMMAND_PLAYBOOK_SHOW: _handle_playbook_show,
    COMMAND_PLAYBOOK_RUN: _handle_playbook_run,
    COMMAND_PLAYBOOK_STATUS: _handle_playbook_status,
}
