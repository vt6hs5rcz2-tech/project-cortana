"""Sequential defensive workflow executor composed over DefensiveToolExecutor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.config import (
    MAX_NOTE_TEXT_LENGTH,
    MAX_WORKFLOW_RUNTIME_SECONDS,
    MAX_WORKFLOW_STEPS,
    WORKFLOW_INCIDENT_LINKAGE_ENABLED,
)
from src.incident_repository import IncidentRepository
from src.security_note import create_incident_note
from src.tool_approval import ToolApprovalRecord, assert_approval_allows_execution
from src.tool_common import (
    ToolAuthorizationError,
    ToolPolicyError,
    ToolValidationError,
    is_expired,
)
from src.tool_definition import (
    DefensiveToolDefinition,
    validate_parameters_against_schema,
)
from src.tool_executor import DefensiveToolExecutor
from src.tool_policy import (
    approval_required,
    assert_executable,
    assert_requestable,
    initial_request_status,
    tool_is_side_effecting,
)
from src.tool_registry import ToolRegistry
from src.tool_repository import ToolControlRepository
from src.tool_request import (
    ToolExecutionRequest,
    create_tool_execution_request,
    replace_tool_execution_request,
)
from src.tool_result import ToolExecutionResult
from src.tool_scope import (
    AuthorizedScope,
    assert_incident_authorized,
    assert_path_authorized,
    assert_scope_usable,
    assert_target_type_authorized,
    assert_tool_authorized,
)
from src.workflow_audit import create_workflow_audit_entry
from src.workflow_common import (
    SIDE_EFFECT_CLAIM_FAILED_ERROR_CODE,
    SIDE_EFFECT_CLAIM_FAILED_MESSAGE,
    SIDE_EFFECT_REEXECUTION_ERROR_CODE,
    SIDE_EFFECT_REEXECUTION_MESSAGE,
    SystemWorkflowClock,
    WorkflowAuthorizationError,
    WorkflowClock,
    WorkflowPolicyError,
    WorkflowStorageError,
    WorkflowValidationError,
    compute_workflow_side_effect_key,
)
from src.workflow_definition import WorkflowDefinition, WorkflowStepDefinition
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import WorkflowRunRepository
from src.workflow_request import WorkflowRunRequest
from src.workflow_result import (
    WorkflowRunResult,
    WorkflowStepResult,
    append_step_result,
    create_workflow_run_result,
    create_workflow_step_result,
    transition_workflow_run_result,
)

logger = logging.getLogger("ProjectCortana")

CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class _PreflightContext:
    workflow: WorkflowDefinition
    scope: AuthorizedScope
    tool_definitions: tuple[DefensiveToolDefinition, ...]


class WorkflowExecutor:
    """Coordinate trusted playbooks through DefensiveToolExecutor only.

    Workflow code never invokes tool implementations directly. Dry-run is the
    default path and calls ``plan_dry_run`` exclusively. Explicit execution calls
    ``execute`` only after live revalidation and approval checks.
    """

    def __init__(
        self,
        *,
        workflow_registry: WorkflowRegistry,
        tool_registry: ToolRegistry,
        tool_executor: DefensiveToolExecutor,
        tool_repository: ToolControlRepository,
        run_repository: WorkflowRunRepository,
        incident_repository: IncidentRepository | None = None,
        clock: WorkflowClock | None = None,
        max_runtime_seconds: int = MAX_WORKFLOW_RUNTIME_SECONDS,
        max_steps: int = MAX_WORKFLOW_STEPS,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._tool_repository = tool_repository
        self._run_repository = run_repository
        self._incident_repository = incident_repository
        self._clock = clock or SystemWorkflowClock()
        self._max_runtime_seconds = max_runtime_seconds
        self._max_steps = max_steps

    def run(
        self,
        request: WorkflowRunRequest,
        *,
        cancellation_check: CancellationCheck | None = None,
    ) -> WorkflowRunResult:
        """Execute or dry-run one workflow request sequentially."""
        created_at = self._clock.utc_now_iso()
        started_monotonic = self._clock.monotonic()

        resolved_version = request.playbook_version
        try:
            early = self._workflow_registry.get(
                request.playbook_name,
                version=request.playbook_version,
            )
            if early is not None:
                resolved_version = early.version
        except (WorkflowValidationError, ToolValidationError):
            resolved_version = request.playbook_version

        run = create_workflow_run_result(
            run_id=request.run_id,
            playbook_name=request.playbook_name,
            playbook_version=resolved_version or "0.0.0",
            dry_run=request.dry_run,
            status="pending",
            scope_id=request.scope_id,
            incident_id=request.incident_id,
            created_timestamp=created_at,
            cancellation_requested=request.cancellation_requested,
        )
        run = self._run_repository.save_run(run)
        self._audit(
            action="workflow-run-created",
            request=request,
            run=run,
            safe_details={
                "dry_run": request.dry_run,
                "status": run.status,
                "incident_linked": request.incident_id is not None,
            },
        )

        try:
            preflight = self._preflight(request)
        except (
            WorkflowValidationError,
            WorkflowAuthorizationError,
            WorkflowPolicyError,
            ToolValidationError,
            ToolAuthorizationError,
            ToolPolicyError,
        ) as error:
            # Ensure denied runs carry the best-known playbook version.
            if run.playbook_version == "0.0.0":
                run = create_workflow_run_result(
                    run_id=run.run_id,
                    playbook_name=run.playbook_name,
                    playbook_version="0.0.0",
                    dry_run=run.dry_run,
                    status="pending",
                    scope_id=run.scope_id,
                    incident_id=run.incident_id,
                    step_results=run.step_results,
                    created_timestamp=run.created_timestamp,
                    cancellation_requested=run.cancellation_requested,
                )
            terminal = transition_workflow_run_result(
                run,
                status="preflight_failed",
                completed_timestamp=self._clock.utc_now_iso(),
                error_code=type(error).__name__,
                error_message=_safe_failure_message(error),
            )
            saved = self._run_repository.save_run(terminal)
            self._audit(
                action="workflow-preflight-denied",
                request=request,
                run=saved,
                safe_details={
                    "dry_run": request.dry_run,
                    "status": saved.status,
                    "error_code": saved.error_code,
                },
            )
            self._audit(
                action="workflow-failed",
                request=request,
                run=saved,
                safe_details={
                    "dry_run": request.dry_run,
                    "status": saved.status,
                    "error_code": saved.error_code,
                },
            )
            return saved

        run = create_workflow_run_result(
            run_id=run.run_id,
            playbook_name=preflight.workflow.name,
            playbook_version=preflight.workflow.version,
            dry_run=request.dry_run,
            status="pending",
            scope_id=request.scope_id,
            incident_id=request.incident_id,
            step_results=(),
            created_timestamp=run.created_timestamp,
            cancellation_requested=request.cancellation_requested,
        )
        run = transition_workflow_run_result(
            run,
            status="running",
            started_timestamp=self._clock.utc_now_iso(),
        )
        run = self._run_repository.save_run(run)
        self._audit(
            action="workflow-preflight-accepted",
            request=request,
            run=run,
            safe_details={
                "dry_run": request.dry_run,
                "status": run.status,
                "step_count": len(preflight.workflow.steps),
            },
        )

        for step, tool_definition in zip(
            preflight.workflow.steps,
            preflight.tool_definitions,
            strict=True,
        ):
            if self._is_cancelled(request, cancellation_check):
                return self._terminate(
                    run,
                    request,
                    status="cancelled",
                    error_code="Cancelled",
                    error_message="Workflow cancelled before the next step.",
                    audit_action="workflow-cancelled",
                )

            if self._budget_exhausted(started_monotonic):
                return self._terminate(
                    run,
                    request,
                    status="timed_out",
                    error_code="WorkflowRuntimeBudgetExceeded",
                    error_message=(
                        "Workflow total runtime budget was exhausted before the "
                        "next step started."
                    ),
                    audit_action="workflow-failed",
                )

            step_result, terminal_status = self._run_step(
                request=request,
                run=run,
                workflow=preflight.workflow,
                step=step,
                started_monotonic=started_monotonic,
                cancellation_check=cancellation_check,
            )
            run = append_step_result(run, step_result)
            run = self._run_repository.save_run(run)

            if terminal_status is not None:
                return self._terminate(
                    run,
                    request,
                    status=terminal_status,
                    error_code=step_result.error_code,
                    error_message=step_result.error_message,
                    audit_action=_terminal_audit_action(terminal_status),
                )

        completed = transition_workflow_run_result(
            run,
            status="completed",
            completed_timestamp=self._clock.utc_now_iso(),
        )
        saved = self._run_repository.save_run(completed)
        self._audit(
            action="workflow-completed",
            request=request,
            run=saved,
            safe_details={
                "dry_run": request.dry_run,
                "status": saved.status,
                "step_count": len(saved.step_results),
            },
        )
        self._maybe_append_incident_note(request=request, run=saved)
        return saved

    def _preflight(self, request: WorkflowRunRequest) -> _PreflightContext:
        workflow = self._workflow_registry.require(
            request.playbook_name,
            version=request.playbook_version,
        )
        if not workflow.enabled:
            raise WorkflowPolicyError("Disabled playbooks cannot run.")
        if not workflow.steps:
            raise WorkflowValidationError("Workflow must contain at least one step.")
        if len(workflow.steps) > self._max_steps:
            raise WorkflowPolicyError(
                f"Workflow exceeds the maximum of {self._max_steps} steps."
            )

        scope = self._require_scope(request.scope_id)
        assert_scope_usable(scope)
        self._assert_incident_link_authorized(request, scope)

        tool_definitions: list[DefensiveToolDefinition] = []
        for step in workflow.steps:
            if step.tool_id in self._workflow_registry.playbook_names():
                raise WorkflowValidationError(
                    "Workflow steps cannot reference playbooks."
                )
            definition = self._tool_registry.require(step.tool_id)
            self._assert_tool_currently_allowed(definition, scope)
            # Static parameters were validated at registration; re-check schema.
            validate_parameters_against_schema(
                step.static_parameters,
                definition.parameter_schema,
            )
            self._assert_static_targets_in_scope(definition, step, scope)
            tool_definitions.append(definition)

        return _PreflightContext(
            workflow=workflow,
            scope=scope,
            tool_definitions=tuple(tool_definitions),
        )

    def _run_step(
        self,
        *,
        request: WorkflowRunRequest,
        run: WorkflowRunResult,
        workflow: WorkflowDefinition,
        step: WorkflowStepDefinition,
        started_monotonic: float,
        cancellation_check: CancellationCheck | None,
    ) -> tuple[WorkflowStepResult, str | None]:
        started_at = self._clock.utc_now_iso()
        self._audit(
            action="workflow-step-attempt",
            request=request,
            run=run,
            step=step,
            safe_details={
                "dry_run": request.dry_run,
                "position": step.position,
                "status": "running",
            },
        )

        if self._is_cancelled(request, cancellation_check):
            step_result = create_workflow_step_result(
                step_id=step.step_id,
                tool_id=step.tool_id,
                position=step.position,
                status="cancelled",
                started_timestamp=started_at,
                completed_timestamp=self._clock.utc_now_iso(),
                error_code="Cancelled",
                error_message="Workflow cancelled before step execution.",
                dry_run=request.dry_run,
            )
            return step_result, "cancelled"

        if self._budget_exhausted(started_monotonic):
            step_result = create_workflow_step_result(
                step_id=step.step_id,
                tool_id=step.tool_id,
                position=step.position,
                status="timed_out",
                started_timestamp=started_at,
                completed_timestamp=self._clock.utc_now_iso(),
                error_code="WorkflowRuntimeBudgetExceeded",
                error_message=(
                    "Workflow total runtime budget was exhausted before the "
                    "step started."
                ),
                dry_run=request.dry_run,
            )
            self._audit(
                action="workflow-step-timed-out",
                request=request,
                run=run,
                step=step,
                safe_details={
                    "dry_run": request.dry_run,
                    "position": step.position,
                    "status": "timed_out",
                    "error_code": step_result.error_code,
                },
            )
            return step_result, "timed_out"

        try:
            scope = self._require_scope(request.scope_id)
            assert_scope_usable(scope)
            definition = self._tool_registry.require(step.tool_id)
            self._assert_tool_currently_allowed(definition, scope)
            self._assert_static_targets_in_scope(definition, step, scope)

            tool_request = self._resolve_tool_request(
                request=request,
                step=step,
                definition=definition,
            )
            approval = None
            if not request.dry_run and approval_required(definition):
                approval = self._resolve_approval(
                    request=request,
                    run=run,
                    step=step,
                    tool_request=tool_request,
                )
                # Revalidate immediately before execution.
                assert_approval_allows_execution(approval, tool_request)

            if not request.dry_run and tool_is_side_effecting(definition):
                self._claim_side_effecting_step(
                    request=request,
                    workflow=workflow,
                    step=step,
                )

            if request.dry_run:
                tool_result = self._tool_executor.plan_dry_run(
                    definition=definition,
                    request=tool_request,
                    scope=scope,
                )
                self._audit(
                    action="workflow-step-dry-run",
                    request=request,
                    run=run,
                    step=step,
                    tool_result=tool_result,
                    safe_details={
                        "dry_run": True,
                        "position": step.position,
                        "outcome": tool_result.outcome,
                    },
                )
                step_status = _map_dry_run_step_status(tool_result)
            else:
                if approval_required(definition):
                    if tool_request.request_status not in {"approved", "running"}:
                        tool_request = replace_tool_execution_request(
                            tool_request,
                            request_status="approved",
                        )
                    assert_approval_allows_execution(approval, tool_request)
                elif tool_request.request_status not in {
                    "drafted",
                    "approved",
                    "running",
                }:
                    tool_request = replace_tool_execution_request(
                        tool_request,
                        request_status="drafted",
                    )

                if not tool_request.dry_run_completed:
                    tool_request = replace_tool_execution_request(
                        tool_request,
                        dry_run_completed=True,
                    )
                if tool_request.request_status != "running":
                    tool_request = replace_tool_execution_request(
                        tool_request,
                        request_status="running",
                    )
                tool_result = self._tool_executor.execute(
                    definition=definition,
                    request=tool_request,
                    scope=scope,
                    approval=approval,
                )
                step_status = _map_execute_step_status(tool_result)
                self._audit(
                    action="workflow-step-result",
                    request=request,
                    run=run,
                    step=step,
                    tool_result=tool_result,
                    approval=approval,
                    safe_details={
                        "dry_run": False,
                        "position": step.position,
                        "outcome": tool_result.outcome,
                    },
                )

            step_result = create_workflow_step_result(
                step_id=step.step_id,
                tool_id=step.tool_id,
                position=step.position,
                status=step_status,
                tool_result=tool_result,
                tool_request_id=tool_request.request_id,
                started_timestamp=started_at,
                completed_timestamp=self._clock.utc_now_iso(),
                error_code=(
                    tool_result.error_class
                    if tool_result.outcome
                    in {
                        "failed",
                        "denied",
                        "expired",
                        "cancelled",
                        "timed_out_terminated",
                        "termination_unconfirmed",
                        "resource_limit_exceeded",
                    }
                    else None
                ),
                error_message=(
                    tool_result.safe_summary
                    if tool_result.outcome
                    in {
                        "failed",
                        "denied",
                        "expired",
                        "cancelled",
                        "timed_out_terminated",
                        "termination_unconfirmed",
                        "resource_limit_exceeded",
                    }
                    else None
                ),
                dry_run=request.dry_run,
            )
            terminal = _terminal_status_for_step(step_result)
            if terminal in {"denied", "failed", "timed_out", "cancelled"}:
                action = {
                    "denied": "workflow-step-denied",
                    "failed": "workflow-step-failed",
                    "timed_out": "workflow-step-timed-out",
                    "cancelled": "workflow-cancelled",
                }[terminal]
                self._audit(
                    action=action,
                    request=request,
                    run=run,
                    step=step,
                    tool_result=tool_result,
                    safe_details={
                        "dry_run": request.dry_run,
                        "position": step.position,
                        "status": terminal,
                        "error_code": step_result.error_code,
                    },
                )
            return step_result, terminal
        except (
            WorkflowValidationError,
            WorkflowAuthorizationError,
            WorkflowPolicyError,
            ToolValidationError,
            ToolAuthorizationError,
            ToolPolicyError,
        ) as error:
            error_code = type(error).__name__
            error_message = _safe_failure_message(error)
            if isinstance(error, WorkflowPolicyError):
                if str(error) == SIDE_EFFECT_REEXECUTION_MESSAGE:
                    error_code = SIDE_EFFECT_REEXECUTION_ERROR_CODE
                    error_message = SIDE_EFFECT_REEXECUTION_MESSAGE
                elif str(error) == SIDE_EFFECT_CLAIM_FAILED_MESSAGE:
                    error_code = SIDE_EFFECT_CLAIM_FAILED_ERROR_CODE
                    error_message = SIDE_EFFECT_CLAIM_FAILED_MESSAGE
            step_result = create_workflow_step_result(
                step_id=step.step_id,
                tool_id=step.tool_id,
                position=step.position,
                status="denied",
                started_timestamp=started_at,
                completed_timestamp=self._clock.utc_now_iso(),
                error_code=error_code,
                error_message=error_message,
                dry_run=request.dry_run,
            )
            self._audit(
                action="workflow-step-denied",
                request=request,
                run=run,
                step=step,
                safe_details={
                    "dry_run": request.dry_run,
                    "position": step.position,
                    "status": "denied",
                    "error_code": error_code,
                },
            )
            return step_result, "denied"

    def _resolve_tool_request(
        self,
        *,
        request: WorkflowRunRequest,
        step: WorkflowStepDefinition,
        definition: DefensiveToolDefinition,
    ) -> ToolExecutionRequest:
        prepared = request.step_tool_requests.get(step.step_id)
        if prepared is not None:
            if prepared.tool_id != step.tool_id:
                raise WorkflowValidationError(
                    "Prepared tool request tool_id does not match the step."
                )
            if prepared.scope_id != request.scope_id:
                raise WorkflowValidationError(
                    "Prepared tool request scope_id does not match the workflow."
                )
            if dict(prepared.normalized_parameters) != dict(step.static_parameters):
                raise WorkflowValidationError(
                    "Prepared tool request parameters must match static step parameters."
                )
            return prepared

        if request.dry_run:
            status = "drafted"
            dry_run_completed = False
        else:
            # Workflow dry-run is the operator-facing default planning path.
            # Explicit --execute sets the Milestone 9 completion gate so
            # DefensiveToolExecutor.execute remains the sole execution boundary.
            status = (
                "approved"
                if approval_required(definition)
                else initial_request_status(definition)
            )
            dry_run_completed = True
        return create_tool_execution_request(
            tool_id=step.tool_id,
            normalized_parameters=dict(step.static_parameters),
            scope_id=request.scope_id,
            requested_by=request.requested_by,
            dry_run_requested=False,
            justification=(
                f"Workflow '{request.playbook_name}' step '{step.step_id}'"
            ),
            request_status=status,
            dry_run_completed=dry_run_completed,
            incident_id=(
                step.static_parameters.get("incident_id")
                if isinstance(step.static_parameters.get("incident_id"), str)
                else None
            ),
        )

    def _resolve_approval(
        self,
        *,
        request: WorkflowRunRequest,
        run: WorkflowRunResult,
        step: WorkflowStepDefinition,
        tool_request: ToolExecutionRequest,
    ) -> ToolApprovalRecord:
        self._audit(
            action="workflow-approval-requested",
            request=request,
            run=run,
            step=step,
            safe_details={
                "dry_run": False,
                "position": step.position,
                "request_id_present": True,
            },
        )
        approval = request.step_approvals.get(step.step_id)
        if approval is None:
            approval = self._tool_repository.get_approval_for_request(
                tool_request.request_id
            )
        if approval is None:
            self._audit(
                action="workflow-approval-decision",
                request=request,
                run=run,
                step=step,
                safe_details={
                    "dry_run": False,
                    "position": step.position,
                    "decision": "missing",
                },
            )
            raise WorkflowAuthorizationError(
                "Explicit step approval is required for this workflow step."
            )
        assert_approval_allows_execution(approval, tool_request)
        self._audit(
            action="workflow-approval-decision",
            request=request,
            run=run,
            step=step,
            approval=approval,
            safe_details={
                "dry_run": False,
                "position": step.position,
                "decision": approval.decision,
            },
        )
        return approval

    def _claim_side_effecting_step(
        self,
        *,
        request: WorkflowRunRequest,
        workflow: WorkflowDefinition,
        step: WorkflowStepDefinition,
    ) -> None:
        """Record a side-effect claim before live execution.

        The claim is persisted before ``execute`` so a later crash or failed
        run-completion write cannot silently replay the same operation.
        Presence of the marker means the step was already attempted; it does
        not assert that the prior side effect succeeded.
        """
        effect_key = compute_workflow_side_effect_key(
            playbook_name=workflow.name,
            playbook_version=workflow.version,
            step_id=step.step_id,
            tool_id=step.tool_id,
            static_parameters=step.static_parameters,
            operation_id=request.operation_id,
        )
        if self._run_repository.has_side_effect_key(effect_key):
            raise WorkflowPolicyError(SIDE_EFFECT_REEXECUTION_MESSAGE)
        try:
            self._run_repository.claim_side_effect_key(
                effect_key,
                playbook_name=workflow.name,
                step_id=step.step_id,
            )
        except WorkflowStorageError as error:
            raise WorkflowPolicyError(SIDE_EFFECT_CLAIM_FAILED_MESSAGE) from error

    def _assert_tool_currently_allowed(
        self,
        definition: DefensiveToolDefinition,
        scope: AuthorizedScope,
    ) -> None:
        assert_requestable(definition)
        assert_executable(definition)
        assert_tool_authorized(scope, definition.tool_id)
        target_type = (
            definition.supported_target_types[0]
            if definition.supported_target_types
            else "none"
        )
        assert_target_type_authorized(scope, target_type)

    def _assert_static_targets_in_scope(
        self,
        definition: DefensiveToolDefinition,
        step: WorkflowStepDefinition,
        scope: AuthorizedScope,
    ) -> None:
        incident_id = step.static_parameters.get("incident_id")
        if isinstance(incident_id, str):
            assert_incident_authorized(scope, incident_id)
        for parameter in definition.parameter_schema:
            if parameter.parameter_type != "file-path":
                continue
            value = step.static_parameters.get(parameter.name)
            if isinstance(value, str):
                assert_path_authorized(scope, value)

    def _require_scope(self, scope_id: str) -> AuthorizedScope:
        scope = self._tool_repository.get_scope(scope_id)
        if scope is None:
            raise WorkflowAuthorizationError("Authorized scope was not found.")
        if is_expired(scope.expires_at):
            raise WorkflowAuthorizationError("Authorized scope has expired.")
        return scope

    def _assert_incident_link_authorized(
        self,
        request: WorkflowRunRequest,
        scope: AuthorizedScope,
    ) -> None:
        """Validate optional incident linkage before step one."""
        if request.incident_id is None:
            return
        if not WORKFLOW_INCIDENT_LINKAGE_ENABLED:
            raise WorkflowPolicyError("Workflow incident linkage is disabled.")
        if self._incident_repository is None:
            raise WorkflowPolicyError(
                "Incident repository is required for workflow incident linkage."
            )
        incident = self._incident_repository.get_incident(request.incident_id)
        if incident is None:
            raise WorkflowValidationError("Linked incident was not found.")
        assert_incident_authorized(scope, request.incident_id)

    def _maybe_append_incident_note(
        self,
        *,
        request: WorkflowRunRequest,
        run: WorkflowRunResult,
    ) -> None:
        """Append exactly one safe summary note for a completed linked run."""
        if not WORKFLOW_INCIDENT_LINKAGE_ENABLED:
            return
        if run.incident_id is None or request.incident_id is None:
            return
        if run.status != "completed":
            return
        if self._incident_repository is None:
            return

        # Revalidate existence and scope authorization before appending.
        incident = self._incident_repository.get_incident(run.incident_id)
        if incident is None:
            logger.error(
                "Workflow incident note skipped; incident missing run_id=%s",
                run.run_id,
            )
            return
        try:
            scope = self._require_scope(run.scope_id)
            assert_incident_authorized(scope, run.incident_id)
        except (ToolAuthorizationError, WorkflowAuthorizationError):
            logger.error(
                "Workflow incident note skipped; authorization failed run_id=%s",
                run.run_id,
            )
            return

        completed_at = run.completed_timestamp or self._clock.utc_now_iso()
        note_text = (
            f"Workflow playbook '{run.playbook_name}' v{run.playbook_version} "
            f"run {run.run_id} finished with status {run.status} at {completed_at}."
        )
        if len(note_text) > MAX_NOTE_TEXT_LENGTH:
            note_text = f"{note_text[:MAX_NOTE_TEXT_LENGTH]}..."
        note = create_incident_note(
            incident_id=run.incident_id,
            author="workflow-orchestration",
            text=note_text,
            note_type="summary",
        )
        self._incident_repository.add_note(note)

    def _budget_exhausted(self, started_monotonic: float) -> bool:
        elapsed = self._clock.monotonic() - started_monotonic
        return elapsed >= self._max_runtime_seconds

    def _is_cancelled(
        self,
        request: WorkflowRunRequest,
        cancellation_check: CancellationCheck | None,
    ) -> bool:
        if request.cancellation_requested:
            return True
        if cancellation_check is not None and cancellation_check():
            return True
        return False

    def _terminate(
        self,
        run: WorkflowRunResult,
        request: WorkflowRunRequest,
        *,
        status: str,
        error_code: str | None,
        error_message: str | None,
        audit_action: str,
    ) -> WorkflowRunResult:
        terminal = transition_workflow_run_result(
            run,
            status=status,
            completed_timestamp=self._clock.utc_now_iso(),
            error_code=error_code,
            error_message=error_message,
            cancellation_requested=(
                True if status == "cancelled" else run.cancellation_requested
            ),
        )
        saved = self._run_repository.save_run(terminal)
        self._audit(
            action=audit_action,
            request=request,
            run=saved,
            safe_details={
                "dry_run": request.dry_run,
                "status": saved.status,
                "error_code": saved.error_code,
                "step_count": len(saved.step_results),
            },
        )
        if audit_action != "workflow-failed" and status in {
            "failed",
            "denied",
            "timed_out",
            "preflight_failed",
        }:
            self._audit(
                action="workflow-failed",
                request=request,
                run=saved,
                safe_details={
                    "dry_run": request.dry_run,
                    "status": saved.status,
                    "error_code": saved.error_code,
                },
            )
        return saved

    def _audit(
        self,
        *,
        action: str,
        request: WorkflowRunRequest,
        run: WorkflowRunResult,
        step: WorkflowStepDefinition | None = None,
        tool_result: ToolExecutionResult | None = None,
        approval: ToolApprovalRecord | None = None,
        safe_details: dict[str, object] | None = None,
    ) -> None:
        entry = create_workflow_audit_entry(
            action=action,
            actor=request.requested_by,
            run_id=run.run_id,
            playbook_name=run.playbook_name,
            step_id=None if step is None else step.step_id,
            tool_id=None if step is None else step.tool_id,
            scope_id=request.scope_id,
            approval_id=None if approval is None else approval.approval_id,
            result_id=None if tool_result is None else tool_result.result_id,
            safe_details=dict(safe_details or {}),
        )
        self._run_repository.append_audit_entry(entry)


def _map_dry_run_step_status(result: ToolExecutionResult) -> str:
    if result.outcome == "planned":
        return "planned"
    if result.outcome == "denied":
        return "denied"
    if result.outcome == "expired":
        return "denied"
    if result.outcome == "cancelled":
        return "cancelled"
    if result.outcome == "failed":
        return "failed"
    raise WorkflowValidationError(
        "Dry-run tool results cannot report successful execution."
    )


def _map_execute_step_status(result: ToolExecutionResult) -> str:
    if result.outcome == "succeeded":
        return "completed"
    if result.outcome == "denied":
        return "denied"
    if result.outcome == "cancelled":
        return "cancelled"
    if result.outcome == "expired":
        return "denied"
    if result.outcome in {"timed_out_terminated", "termination_unconfirmed"}:
        return "timed_out"
    if result.outcome == "resource_limit_exceeded":
        return "failed"
    if result.outcome == "failed":
        if result.error_class == "TimeoutError":
            return "timed_out"
        return "failed"
    raise WorkflowValidationError(
        f"Unexpected tool execution outcome '{result.outcome}'."
    )


def _terminal_status_for_step(step_result: WorkflowStepResult) -> str | None:
    if step_result.status in {"planned", "completed"}:
        return None
    if step_result.status == "denied":
        return "denied"
    if step_result.status == "failed":
        return "failed"
    if step_result.status == "timed_out":
        return "timed_out"
    if step_result.status == "cancelled":
        return "cancelled"
    return "failed"


def _terminal_audit_action(status: str) -> str:
    if status == "cancelled":
        return "workflow-cancelled"
    if status == "completed":
        return "workflow-completed"
    return "workflow-failed"


def _safe_failure_message(error: Exception) -> str:
    name = type(error).__name__
    return f"Workflow stopped ({name})."
