"""Local slash-command handlers for the Milestone 9 defensive tool framework."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.config import MAX_TOOL_LIST_PREVIEW_CHARS
from src.incident_repository import IncidentRepository
from src.tool_approval import create_tool_approval
from src.tool_audit import create_tool_audit_entry
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolEnumError,
    InvalidToolIdError,
    ToolAuthorizationError,
    ToolParameterError,
    ToolPolicyError,
    ToolValidationError,
    fingerprint_display_prefix,
    is_expired,
)
from src.tool_definition import (
    redact_parameters_for_display,
    validate_parameters_against_schema,
)
from src.tool_executor import DefensiveToolExecutor
from src.tool_policy import (
    approval_required,
    assert_requestable,
    initial_request_status,
)
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_repository import ToolControlRepository, ToolStorageError
from src.tool_request import (
    create_tool_execution_request,
    replace_tool_execution_request,
)
from src.tool_scope import (
    assert_incident_authorized,
    assert_tool_authorized,
    create_authorized_scope,
    disable_authorized_scope,
)
from src.security_commands import extract_command_argument, split_delimited_fields

logger = logging.getLogger("ProjectCortana")

COMMAND_TOOLS = "tools"
COMMAND_TOOL = "tool"
COMMAND_SCOPE_NEW = "scope-new"
COMMAND_SCOPES = "scopes"
COMMAND_SCOPE = "scope"
COMMAND_SCOPE_DISABLE = "scope-disable"
COMMAND_TOOL_REQUEST = "tool-request"
COMMAND_TOOL_REQUESTS = "tool-requests"
COMMAND_TOOL_REQUEST_SHOW = "tool-request-show"
COMMAND_TOOL_DRY_RUN = "tool-dry-run"
COMMAND_TOOL_APPROVE = "tool-approve"
COMMAND_TOOL_REJECT = "tool-reject"
COMMAND_TOOL_CANCEL = "tool-cancel"
COMMAND_TOOL_RUN = "tool-run"
COMMAND_TOOL_RESULT = "tool-result"
COMMAND_TOOL_AUDIT = "tool-audit"

TOOL_COMMAND_NAMES = frozenset(
    {
        COMMAND_TOOLS,
        COMMAND_TOOL,
        COMMAND_SCOPE_NEW,
        COMMAND_SCOPES,
        COMMAND_SCOPE,
        COMMAND_SCOPE_DISABLE,
        COMMAND_TOOL_REQUEST,
        COMMAND_TOOL_REQUESTS,
        COMMAND_TOOL_REQUEST_SHOW,
        COMMAND_TOOL_DRY_RUN,
        COMMAND_TOOL_APPROVE,
        COMMAND_TOOL_REJECT,
        COMMAND_TOOL_CANCEL,
        COMMAND_TOOL_RUN,
        COMMAND_TOOL_RESULT,
        COMMAND_TOOL_AUDIT,
    }
)

SCOPE_NEW_USAGE = (
    "Cortana: Usage: /scope-new <name> | <tool-id-list> | "
    "<allowed-root-or-none> | <justification>"
)
TOOL_REQUEST_USAGE = (
    "Cortana: Usage: /tool-request <tool-id> | <scope-id> | "
    "<parameter-json> | <justification>"
)
TOOL_APPROVE_USAGE = (
    "Cortana: Usage: /tool-approve <request-id> | <reason>"
)
TOOL_REJECT_USAGE = (
    "Cortana: Usage: /tool-reject <request-id> | <reason>"
)
TOOLS_EMPTY = "Cortana: No defensive tools are registered."
SCOPES_EMPTY = "Cortana: No authorized scopes are saved."
REQUESTS_EMPTY = "Cortana: No tool execution requests are saved."
AUDIT_EMPTY = "Cortana: No tool audit entries are saved."


@dataclass(frozen=True)
class ToolCommandContext:
    """Inputs available to Milestone 9 tool command handlers."""

    message: str
    tool_registry: ToolRegistry
    tool_repository: ToolControlRepository
    tool_executor: DefensiveToolExecutor
    incident_repository: IncidentRepository


@dataclass(frozen=True)
class ToolCommandResult:
    """Local message returned by a tool command handler."""

    message: str


ToolCommandHandler = Callable[[ToolCommandContext], ToolCommandResult]


def handle_tool_command(
    command_name: str,
    context: ToolCommandContext,
) -> ToolCommandResult | None:
    """Dispatch a Milestone 9 tool command, or return None when unknown."""
    handler = TOOL_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    return handler(context)


def bounded_preview(text: str) -> str:
    """Return a bounded single-line preview for list command output."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= MAX_TOOL_LIST_PREVIEW_CHARS:
        return cleaned
    return f"{cleaned[:MAX_TOOL_LIST_PREVIEW_CHARS]}..."


def _handle_tools(context: ToolCommandContext) -> ToolCommandResult:
    tools = context.tool_registry.list_enabled()
    if not tools:
        return ToolCommandResult(message=TOOLS_EMPTY)
    lines = ["Cortana: Enabled defensive tools:"]
    for tool in tools:
        lines.append(
            f"  - {tool.tool_id} [{tool.risk_level}/{tool.category}] "
            f"{bounded_preview(tool.description)}"
        )
    return ToolCommandResult(message="\n".join(lines))


def _handle_tool(context: ToolCommandContext) -> ToolCommandResult:
    tool_id = extract_command_argument(context.message).strip()
    if not tool_id:
        return ToolCommandResult(
            message="Cortana: Please provide a tool ID. Usage: /tool <tool-id>"
        )
    try:
        definition = context.tool_registry.require(tool_id)
    except ToolValidationError:
        return ToolCommandResult(
            message=f"Cortana: No registered tool found with ID '{tool_id}'."
        )

    params = ", ".join(
        f"{item.name}:{'required' if item.required else 'optional'}"
        for item in definition.parameter_schema
    ) or "(none)"
    message = (
        "Cortana: Tool details\n"
        f"  Tool ID: {definition.tool_id}\n"
        f"  Name: {definition.name}\n"
        f"  Category: {definition.category}\n"
        f"  Risk: {definition.risk_level}\n"
        f"  Mode: {definition.execution_mode}\n"
        f"  Version: {definition.version}\n"
        f"  Enabled: {definition.enabled}\n"
        f"  Requires approval: {definition.requires_approval}\n"
        f"  Supports dry run: {definition.supports_dry_run}\n"
        f"  Timeout seconds: {definition.timeout_seconds}\n"
        f"  Parameters: {params}\n"
        f"  Description: {bounded_preview(definition.description)}"
    )
    return ToolCommandResult(message=message)


def _handle_scope_new(context: ToolCommandContext) -> ToolCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 4)
    if fields is None:
        return ToolCommandResult(message=SCOPE_NEW_USAGE)

    name, tool_id_list, root_or_none, justification = fields
    tool_ids = [
        item.strip()
        for item in tool_id_list.split(",")
        if item.strip()
    ]
    roots: list[str] = []
    target_types = ["none", "system-summary", "incident", "mock-log"]
    if root_or_none.strip().lower() != "none":
        roots = [root_or_none]
        target_types.append("local-file")

    # Justification is required by the command contract but never logged.
    if not justification.strip():
        return ToolCommandResult(message=SCOPE_NEW_USAGE)

    try:
        for tool_id in tool_ids:
            definition = context.tool_registry.require(tool_id)
            assert_requestable(definition)
        scope = create_authorized_scope(
            scope_name=name,
            allowed_tool_ids=tool_ids,
            allowed_target_types=target_types,
            allowed_local_path_roots=roots,
            notes=justification,
            prohibited_tool_ids=context.tool_registry.prohibited_tool_ids(),
        )
        saved = context.tool_repository.add_scope(scope)
    except (
        BlankToolFieldError,
        InvalidToolIdError,
        ToolAuthorizationError,
        ToolPolicyError,
        ToolValidationError,
    ):
        return ToolCommandResult(message=SCOPE_NEW_USAGE)
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info(
        "Tool scope created scope_id=%s allowed_tool_count=%s",
        saved.scope_id,
        len(saved.allowed_tool_ids),
    )
    return ToolCommandResult(
        message=(
            f"Cortana: Authorized scope created ({saved.scope_id}). "
            f"Allowed tools: {len(saved.allowed_tool_ids)}."
        )
    )


def _handle_scopes(context: ToolCommandContext) -> ToolCommandResult:
    try:
        scopes = context.tool_repository.list_scopes()
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)
    if not scopes:
        return ToolCommandResult(message=SCOPES_EMPTY)
    lines = ["Cortana: Authorized scopes:"]
    for scope in scopes:
        status = "active" if scope.active and not is_expired(scope.expires_at) else "inactive"
        lines.append(
            f"  - {scope.scope_id} [{status}] tools={len(scope.allowed_tool_ids)} "
            f"roots={len(scope.allowed_local_path_roots)} "
            f"{bounded_preview(scope.scope_name)}"
        )
    return ToolCommandResult(message="\n".join(lines))


def _handle_scope(context: ToolCommandContext) -> ToolCommandResult:
    scope_id = extract_command_argument(context.message).strip()
    if not scope_id:
        return ToolCommandResult(
            message="Cortana: Please provide a scope ID. Usage: /scope <scope-id>"
        )
    try:
        scope = context.tool_repository.get_scope(scope_id)
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)
    if scope is None:
        return ToolCommandResult(
            message=f"Cortana: No saved scope found with ID '{scope_id}'."
        )

    message = (
        "Cortana: Scope details\n"
        f"  Scope ID: {scope.scope_id}\n"
        f"  Name: {bounded_preview(scope.scope_name)}\n"
        f"  Active: {scope.active}\n"
        f"  Expired: {is_expired(scope.expires_at)}\n"
        f"  Allowed tools: {', '.join(scope.allowed_tool_ids) or '(none)'}\n"
        f"  Allowed target types: {', '.join(scope.allowed_target_types)}\n"
        f"  Allowed root count: {len(scope.allowed_local_path_roots)}\n"
        f"  Allowed incident count: {len(scope.allowed_incident_ids)}\n"
        f"  Created at: {scope.created_at}"
    )
    return ToolCommandResult(message=message)


def _handle_scope_disable(context: ToolCommandContext) -> ToolCommandResult:
    scope_id = extract_command_argument(context.message).strip()
    if not scope_id:
        return ToolCommandResult(
            message=(
                "Cortana: Please provide a scope ID. "
                "Usage: /scope-disable <scope-id>"
            )
        )
    try:
        scope = context.tool_repository.get_scope(scope_id)
        if scope is None:
            return ToolCommandResult(
                message=f"Cortana: No saved scope found with ID '{scope_id}'."
            )
        updated = context.tool_repository.update_scope(disable_authorized_scope(scope))
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info("Tool scope disabled scope_id=%s", updated.scope_id)
    return ToolCommandResult(
        message=f"Cortana: Scope '{updated.scope_id}' has been disabled."
    )


def _handle_tool_request(context: ToolCommandContext) -> ToolCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 4)
    if fields is None:
        return ToolCommandResult(message=TOOL_REQUEST_USAGE)

    tool_id, scope_id, parameter_json, justification = fields
    try:
        definition = context.tool_registry.require(tool_id)
        assert_requestable(definition)
        scope = context.tool_repository.get_scope(scope_id.strip())
        if scope is None:
            return ToolCommandResult(
                message=f"Cortana: No saved scope found with ID '{scope_id.strip()}'."
            )
        assert_tool_authorized(scope, definition.tool_id)

        raw_parameters = _parse_parameter_json(parameter_json)
        normalized = validate_parameters_against_schema(
            raw_parameters,
            definition.parameter_schema,
        )

        incident_id = normalized.get("incident_id")
        if isinstance(incident_id, str):
            incident = context.incident_repository.get_incident(incident_id)
            if incident is None:
                return ToolCommandResult(
                    message="Cortana: The specified incident was not found."
                )
            assert_incident_authorized(scope, incident_id)

        status = initial_request_status(definition)
        request = create_tool_execution_request(
            tool_id=definition.tool_id,
            normalized_parameters=normalized,
            scope_id=scope.scope_id,
            justification=justification,
            request_status=status,
            incident_id=incident_id if isinstance(incident_id, str) else None,
        )
        saved = context.tool_repository.add_request(request)
    except json.JSONDecodeError:
        return ToolCommandResult(
            message="Cortana: parameter-json must be valid strict JSON."
        )
    except (
        BlankToolFieldError,
        InvalidToolEnumError,
        InvalidToolIdError,
        ToolAuthorizationError,
        ToolParameterError,
        ToolPolicyError,
        ToolValidationError,
    ):
        return ToolCommandResult(message=TOOL_REQUEST_USAGE)
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info(
        "Tool request created request_id=%s tool_id=%s status=%s",
        saved.request_id,
        saved.tool_id,
        saved.request_status,
    )
    return ToolCommandResult(
        message=(
            f"Cortana: Tool request created ({saved.request_id}). "
            f"Tool: {saved.tool_id}. Status: {saved.request_status}. "
            f"Fingerprint: {fingerprint_display_prefix(saved.fingerprint)}."
        )
    )


def _handle_tool_requests(context: ToolCommandContext) -> ToolCommandResult:
    try:
        requests = context.tool_repository.list_requests()
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)
    if not requests:
        return ToolCommandResult(message=REQUESTS_EMPTY)
    lines = ["Cortana: Tool execution requests:"]
    for request in requests:
        lines.append(
            f"  - {request.request_id} [{request.request_status}] "
            f"tool={request.tool_id} "
            f"fp={fingerprint_display_prefix(request.fingerprint)}"
        )
    return ToolCommandResult(message="\n".join(lines))


def _handle_tool_request_show(context: ToolCommandContext) -> ToolCommandResult:
    request_id = extract_command_argument(context.message).strip()
    if not request_id:
        return ToolCommandResult(
            message=(
                "Cortana: Please provide a request ID. "
                "Usage: /tool-request-show <request-id>"
            )
        )
    try:
        request = context.tool_repository.get_request(request_id)
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)
    if request is None:
        return ToolCommandResult(
            message=f"Cortana: No saved request found with ID '{request_id}'."
        )

    definition = context.tool_registry.get(request.tool_id)
    redacted = (
        redact_parameters_for_display(
            request.normalized_parameters,
            definition.parameter_schema,
        )
        if definition is not None
        else {"redacted": True}
    )
    message = (
        "Cortana: Tool request details\n"
        f"  Request ID: {request.request_id}\n"
        f"  Tool ID: {request.tool_id}\n"
        f"  Scope ID: {request.scope_id}\n"
        f"  Status: {request.request_status}\n"
        f"  Dry-run completed: {request.dry_run_completed}\n"
        f"  Incident linked: {request.incident_id is not None}\n"
        f"  Fingerprint: {fingerprint_display_prefix(request.fingerprint)}\n"
        f"  Parameters: {json.dumps(redacted, ensure_ascii=False, sort_keys=True)}"
    )
    return ToolCommandResult(message=message)


def _handle_tool_dry_run(context: ToolCommandContext) -> ToolCommandResult:
    request_id = extract_command_argument(context.message).strip()
    if not request_id:
        return ToolCommandResult(
            message=(
                "Cortana: Please provide a request ID. "
                "Usage: /tool-dry-run <request-id>"
            )
        )
    try:
        request = context.tool_repository.get_request(request_id)
        if request is None:
            return ToolCommandResult(
                message=f"Cortana: No saved request found with ID '{request_id}'."
            )
        if request.request_status in {"cancelled", "rejected", "expired"}:
            return ToolCommandResult(
                message=(
                    "Cortana: Dry run is not available for "
                    f"'{request.request_status}' requests."
                )
            )
        definition = context.tool_registry.require(request.tool_id)
        scope = context.tool_repository.get_scope(request.scope_id)
        if scope is None:
            return ToolCommandResult(message="Cortana: Request scope was not found.")

        result = context.tool_executor.plan_dry_run(
            definition=definition,
            request=request,
            scope=scope,
        )
        saved_result = context.tool_repository.add_result(result)
        updated = replace_tool_execution_request(request, dry_run_completed=True)
        context.tool_repository.update_request(updated)
    except (
        ToolAuthorizationError,
        ToolPolicyError,
        ToolValidationError,
    ):
        return ToolCommandResult(
            message="Cortana: Dry run could not be completed for that request."
        )
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info(
        "Tool dry-run generated request_id=%s result_id=%s",
        request.request_id,
        saved_result.result_id,
    )
    return ToolCommandResult(
        message=(
            f"Cortana: Dry-run plan recorded ({saved_result.result_id}). "
            "No action was executed. "
            f"Approval required: "
            f"{saved_result.structured_data.get('approval_required', False)}."
        )
    )


def _handle_tool_approve(context: ToolCommandContext) -> ToolCommandResult:
    return _record_decision(context, decision="approved", usage=TOOL_APPROVE_USAGE)


def _handle_tool_reject(context: ToolCommandContext) -> ToolCommandResult:
    return _record_decision(context, decision="rejected", usage=TOOL_REJECT_USAGE)


def _record_decision(
    context: ToolCommandContext,
    *,
    decision: str,
    usage: str,
) -> ToolCommandResult:
    fields = split_delimited_fields(extract_command_argument(context.message), 2)
    if fields is None:
        return ToolCommandResult(message=usage)
    request_id, reason = fields
    try:
        request = context.tool_repository.get_request(request_id.strip())
        if request is None:
            return ToolCommandResult(
                message=(
                    f"Cortana: No saved request found with ID '{request_id.strip()}'."
                )
            )
        if request.request_status not in {"awaiting-approval", "drafted"}:
            return ToolCommandResult(
                message=(
                    "Cortana: Only awaiting-approval or drafted requests "
                    "can receive a decision."
                )
            )
        definition = context.tool_registry.require(request.tool_id)
        if decision == "approved" and not approval_required(definition):
            # Allow explicit approval even when optional.
            pass
        approval = create_tool_approval(
            request=request,
            decision=decision,
            reason=reason,
        )
        saved_approval = context.tool_repository.add_approval(approval)
        next_status = "approved" if decision == "approved" else "rejected"
        updated = replace_tool_execution_request(request, request_status=next_status)
        context.tool_repository.update_request(updated)
    except (ToolValidationError, ToolPolicyError):
        return ToolCommandResult(message=usage)
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info(
        "Tool approval recorded request_id=%s decision=%s",
        request.request_id,
        decision,
    )
    return ToolCommandResult(
        message=(
            f"Cortana: Request '{request.request_id}' {decision} "
            f"({saved_approval.approval_id})."
        )
    )


def _handle_tool_cancel(context: ToolCommandContext) -> ToolCommandResult:
    request_id = extract_command_argument(context.message).strip()
    if not request_id:
        return ToolCommandResult(
            message=(
                "Cortana: Please provide a request ID. "
                "Usage: /tool-cancel <request-id>"
            )
        )
    try:
        request = context.tool_repository.get_request(request_id)
        if request is None:
            return ToolCommandResult(
                message=f"Cortana: No saved request found with ID '{request_id}'."
            )
        if request.request_status in {
            "succeeded",
            "failed",
            "cancelled",
            "running",
        }:
            return ToolCommandResult(
                message=(
                    "Cortana: That request can no longer be cancelled "
                    f"(status: {request.request_status})."
                )
            )
        updated = replace_tool_execution_request(
            request,
            request_status="cancelled",
        )
        context.tool_repository.update_request(updated)
        context.tool_repository.append_audit_entry(
            create_tool_audit_entry(
                action="request-cancelled",
                request_id=updated.request_id,
                tool_id=updated.tool_id,
                scope_id=updated.scope_id,
                safe_details={"status": "cancelled"},
            )
        )
    except ToolPolicyError:
        return ToolCommandResult(
            message="Cortana: That request can no longer be cancelled."
        )
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info("Tool request cancelled request_id=%s", request_id)
    return ToolCommandResult(
        message=f"Cortana: Tool request '{request_id}' has been cancelled."
    )


def _handle_tool_run(context: ToolCommandContext) -> ToolCommandResult:
    request_id = extract_command_argument(context.message).strip()
    if not request_id:
        return ToolCommandResult(
            message=(
                "Cortana: Please provide a request ID. Usage: /tool-run <request-id>"
            )
        )
    try:
        request = context.tool_repository.get_request(request_id)
        if request is None:
            return ToolCommandResult(
                message=f"Cortana: No saved request found with ID '{request_id}'."
            )
        definition = context.tool_registry.require(request.tool_id)
        scope = context.tool_repository.get_scope(request.scope_id)
        if scope is None:
            return ToolCommandResult(message="Cortana: Request scope was not found.")
        approval = context.tool_repository.get_approval_for_request(request.request_id)

        if request.request_status not in {"drafted", "approved"}:
            return ToolCommandResult(
                message=(
                    "Cortana: Tool execution was denied by policy or authorization."
                )
            )

        running = replace_tool_execution_request(request, request_status="running")
        context.tool_repository.update_request(running)
        context.tool_repository.append_audit_entry(
            create_tool_audit_entry(
                action="execution-started",
                request_id=running.request_id,
                tool_id=running.tool_id,
                scope_id=running.scope_id,
                safe_details={"status": "running"},
            )
        )

        result = context.tool_executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=approval,
        )
        saved_result = context.tool_repository.add_result(result)

        if result.outcome == "succeeded":
            final_status = "succeeded"
        else:
            final_status = "failed"

        context.tool_repository.update_request(
            replace_tool_execution_request(
                running,
                request_status=final_status,
            )
        )
    except (
        ToolAuthorizationError,
        ToolPolicyError,
        ToolValidationError,
    ):
        return ToolCommandResult(
            message="Cortana: Tool execution was denied by policy or authorization."
        )
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)

    logger.info(
        "Tool execution finished request_id=%s outcome=%s",
        request_id,
        saved_result.outcome,
    )
    return ToolCommandResult(
        message=(
            f"Cortana: Tool result recorded ({saved_result.result_id}). "
            f"Outcome: {saved_result.outcome}. {saved_result.safe_summary}"
        )
    )


def _handle_tool_result(context: ToolCommandContext) -> ToolCommandResult:
    result_id = extract_command_argument(context.message).strip()
    if not result_id:
        return ToolCommandResult(
            message=(
                "Cortana: Please provide a result ID. "
                "Usage: /tool-result <result-id>"
            )
        )
    try:
        result = context.tool_repository.get_result(result_id)
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)
    if result is None:
        return ToolCommandResult(
            message=f"Cortana: No saved result found with ID '{result_id}'."
        )

    # Show safe metadata and structured data already validated as JSON-safe.
    safe_data = {
        key: value
        for key, value in result.structured_data.items()
        if key
        not in {
            "validated_parameters",
        }
    }
    # Include redacted parameters if present.
    if "validated_parameters" in result.structured_data:
        safe_data["validated_parameters"] = result.structured_data[
            "validated_parameters"
        ]

    message = (
        "Cortana: Tool result details\n"
        f"  Result ID: {result.result_id}\n"
        f"  Request ID: {result.request_id}\n"
        f"  Tool ID: {result.tool_id}\n"
        f"  Outcome: {result.outcome}\n"
        f"  Dry run: {result.dry_run}\n"
        f"  Truncated: {result.output_truncated}\n"
        f"  Summary: {bounded_preview(result.safe_summary)}\n"
        f"  Data: {json.dumps(safe_data, ensure_ascii=False, sort_keys=True)}"
    )
    return ToolCommandResult(message=message)


def _handle_tool_audit(context: ToolCommandContext) -> ToolCommandResult:
    try:
        entries = context.tool_repository.list_audit_entries()
    except ToolStorageError as error:
        return ToolCommandResult(message=error.user_message)
    if not entries:
        return ToolCommandResult(message=AUDIT_EMPTY)
    lines = ["Cortana: Tool audit entries:"]
    for entry in entries[-50:]:
        lines.append(
            f"  - {entry.timestamp} {entry.action} "
            f"tool={entry.tool_id or '-'} "
            f"request={entry.request_id or '-'}"
        )
    return ToolCommandResult(message="\n".join(lines))


def _parse_parameter_json(raw: str) -> dict[str, Any]:
    """Parse strict JSON object parameters and reject non-objects."""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ToolParameterError("parameter-json must be a JSON object.")
    if any(key in {"command", "shell", "cmd", "powershell", "bash"} for key in parsed):
        raise ToolParameterError("Command string parameters are not permitted.")
    return parsed


def create_default_tool_services(
    *,
    tool_repository: ToolControlRepository,
    incident_repository: IncidentRepository,
) -> tuple[ToolRegistry, DefensiveToolExecutor]:
    """Create the default registry and executor for application wiring."""
    from src.tool_process_adapter import ToolProcessAdapter, bind_audit_appender

    registry = build_default_tool_registry()
    executor = DefensiveToolExecutor(
        incident_repository=incident_repository,
        process_adapter=ToolProcessAdapter(
            audit_appender=bind_audit_appender(tool_repository),
        ),
    )
    return registry, executor


TOOL_COMMAND_HANDLERS: dict[str, ToolCommandHandler] = {
    COMMAND_TOOLS: _handle_tools,
    COMMAND_TOOL: _handle_tool,
    COMMAND_SCOPE_NEW: _handle_scope_new,
    COMMAND_SCOPES: _handle_scopes,
    COMMAND_SCOPE: _handle_scope,
    COMMAND_SCOPE_DISABLE: _handle_scope_disable,
    COMMAND_TOOL_REQUEST: _handle_tool_request,
    COMMAND_TOOL_REQUESTS: _handle_tool_requests,
    COMMAND_TOOL_REQUEST_SHOW: _handle_tool_request_show,
    COMMAND_TOOL_DRY_RUN: _handle_tool_dry_run,
    COMMAND_TOOL_APPROVE: _handle_tool_approve,
    COMMAND_TOOL_REJECT: _handle_tool_reject,
    COMMAND_TOOL_CANCEL: _handle_tool_cancel,
    COMMAND_TOOL_RUN: _handle_tool_run,
    COMMAND_TOOL_RESULT: _handle_tool_result,
    COMMAND_TOOL_AUDIT: _handle_tool_audit,
}
