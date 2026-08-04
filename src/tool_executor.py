"""Defensive tool execution engine with dry-run planning and approval gates."""

from __future__ import annotations

import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from src.incident_repository import IncidentRepository
from src.tool_approval import ToolApprovalRecord, assert_approval_allows_execution
from src.tool_common import (
    ToolAuthorizationError,
    ToolPolicyError,
    ToolValidationError,
    fingerprint_display_prefix,
    utc_timestamp,
)
from src.tool_definition import (
    DefensiveToolDefinition,
    redact_parameters_for_display,
)
from src.tool_implementations import (
    ToolCallable,
    build_implementation_dispatch,
    dry_run_filename_hint,
)
from src.tool_policy import (
    approval_required,
    assert_executable,
    assert_ready_for_execution,
    dry_run_required,
)
from src.tool_request import ToolExecutionRequest
from src.tool_result import ToolExecutionResult, create_tool_execution_result
from src.tool_scope import (
    AuthorizedScope,
    assert_incident_authorized,
    assert_path_authorized,
    assert_target_type_authorized,
    assert_tool_authorized,
)

logger = logging.getLogger("ProjectCortana")

# Timed-out workers are abandoned, not killed. Keep a small pool so one abandoned
# hang does not permanently serialize every later tool call.
#
# Milestone 9 intentionally uses in-process threads (no subprocess/process isolation).
# shutdown(wait=False) returns to the caller promptly, but the Python interpreter can
# still wait for running ThreadPoolExecutor workers before process exit. Hard
# cancellation of a hung worker would require process isolation, which is out of scope.
_TOOL_POOL_WORKERS = 4


def _shutdown_pool(pool: ThreadPoolExecutor) -> None:
    """Call ThreadPoolExecutor.shutdown(wait=False, cancel_futures=True).

    This returns control to the caller without joining running workers. It does
    **not** guarantee that the Python interpreter can exit while a worker is still
    running.
    """
    pool.shutdown(wait=False, cancel_futures=True)


def _discard_abandoned_future(future: Future[Any]) -> None:
    """Retrieve and discard a late future outcome without publishing it."""
    try:
        if future.cancelled():
            return
        # exception() retrieves completion state; the payload is intentionally ignored.
        _ = future.exception()
    except Exception:
        return


class DefensiveToolExecutor:
    """Execute allowlisted defensive tools through trusted internal callables only.

    Timeouts stop the command caller from waiting after ``timeout_seconds``.
    Python threads cannot be forcibly terminated safely, so a timed-out worker may
    still finish later (or hang). Late completion is discarded and must never
    publish a result, change request status, mutate approvals/scopes/registry
    state, or append additional audit records.

    ``shutdown(wait=False)`` returns promptly to the caller, but a permanently
    hung worker may still prevent interpreter shutdown. Process-isolated hard
    cancellation is intentionally out of scope for Milestone 9.
    """

    def __init__(
        self,
        *,
        implementations: dict[str, ToolCallable] | None = None,
        incident_repository: IncidentRepository | None = None,
    ) -> None:
        self._implementations = (
            implementations
            if implementations is not None
            else build_implementation_dispatch(
                incident_repository=incident_repository
            )
        )
        self._pool = ThreadPoolExecutor(
            max_workers=_TOOL_POOL_WORKERS,
            thread_name_prefix="cortana-defensive-tool",
        )
        self._abandon_lock = threading.Lock()
        self._abandoned_futures: set[Future[Any]] = set()
        self._finalize = weakref.finalize(self, _shutdown_pool, self._pool)

    def shutdown(self) -> None:
        """Request pool shutdown without joining running workers.

        Returns promptly to the caller. Does not claim that a hung worker has
        been terminated, and does not guarantee interpreter exit while workers
        remain alive.
        """
        if self._finalize.detach():
            _shutdown_pool(self._pool)

    def plan_dry_run(
        self,
        *,
        definition: DefensiveToolDefinition,
        request: ToolExecutionRequest,
        scope: AuthorizedScope,
    ) -> ToolExecutionResult:
        """Return a dry-run planning result without performing the tool operation."""
        assert_tool_authorized(scope, request.tool_id)
        target_type = _primary_target_type(definition)
        assert_target_type_authorized(scope, target_type)
        assert_incident_authorized(scope, request.incident_id)
        _assert_path_params_in_scope(definition, request, scope)

        # Dry run must not read the target file or perform the primary operation.
        redacted = redact_parameters_for_display(
            request.normalized_parameters,
            definition.parameter_schema,
        )
        filename_hint = dry_run_filename_hint(request.normalized_parameters)
        structured: dict[str, Any] = {
            "tool_name": definition.name,
            "validated_parameters": redacted,
            "target_type": target_type,
            "scope_id": scope.scope_id,
            "risk_level": definition.risk_level,
            "approval_required": approval_required(definition),
            "maximum_runtime_seconds": definition.timeout_seconds,
            "expected_output_category": definition.category,
            "no_action_executed": True,
            "dry_run_required_before_execution": dry_run_required(definition),
            "request_fingerprint_prefix": fingerprint_display_prefix(
                request.fingerprint
            ),
        }
        if filename_hint is not None:
            structured["target_filename"] = filename_hint

        return create_tool_execution_result(
            request_id=request.request_id,
            tool_id=request.tool_id,
            outcome="planned",
            safe_summary=(
                f"Dry-run plan for '{definition.tool_id}'. No action was executed."
            ),
            structured_data=structured,
            dry_run=True,
            incident_id=request.incident_id,
        )

    def execute(
        self,
        *,
        definition: DefensiveToolDefinition,
        request: ToolExecutionRequest,
        scope: AuthorizedScope,
        approval: ToolApprovalRecord | None,
    ) -> ToolExecutionResult:
        """Execute one approved, scoped, allowlisted defensive tool request."""
        started = utc_timestamp()
        try:
            assert_ready_for_execution(definition, request)
            assert_tool_authorized(scope, request.tool_id)
            target_type = _primary_target_type(definition)
            assert_target_type_authorized(scope, target_type)
            assert_incident_authorized(scope, request.incident_id)
            _assert_path_params_in_scope(definition, request, scope)

            if approval_required(definition):
                assert_approval_allows_execution(approval, request)

            assert_executable(definition)
            callable_impl = self._implementations.get(
                definition.implementation_identifier
            )
            if callable_impl is None:
                raise ToolValidationError(
                    "Unknown implementation identifier."
                )

            raw_data = self._invoke_with_timeout(
                callable_impl,
                request.normalized_parameters,
                {"scope_id": scope.scope_id},
                timeout_seconds=definition.timeout_seconds,
            )
            structured, truncated = _bound_structured_output(
                raw_data,
                maximum_characters=definition.maximum_output_characters,
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="succeeded",
                safe_summary=f"Tool '{definition.tool_id}' completed successfully.",
                structured_data=structured,
                output_truncated=truncated,
                dry_run=False,
                incident_id=request.incident_id,
            )
        except (
            ToolAuthorizationError,
            ToolPolicyError,
            ToolValidationError,
        ) as error:
            logger.info(
                "Tool execution denied tool_id=%s request_id=%s error_class=%s",
                definition.tool_id,
                request.request_id,
                type(error).__name__,
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="denied",
                safe_summary="Tool execution was denied by policy or authorization.",
                structured_data={"denied": True},
                error_class=type(error).__name__,
                dry_run=False,
                incident_id=request.incident_id,
            )
        except FuturesTimeoutError:
            logger.info(
                "Tool execution timed out tool_id=%s request_id=%s",
                definition.tool_id,
                request.request_id,
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="failed",
                safe_summary=(
                    "Tool execution exceeded the configured timeout. "
                    "The caller stopped waiting; the worker was not forcibly "
                    "terminated."
                ),
                structured_data={
                    "timeout": True,
                    "caller_stopped_waiting": True,
                    "worker_forcibly_terminated": False,
                },
                error_class="TimeoutError",
                dry_run=False,
                incident_id=request.incident_id,
            )
        except Exception as error:  # noqa: BLE001 - must contain unexpected failures
            logger.error(
                "Tool execution failed tool_id=%s request_id=%s error_class=%s",
                definition.tool_id,
                request.request_id,
                type(error).__name__,
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="failed",
                safe_summary="Tool execution failed.",
                structured_data={"failed": True},
                error_class=type(error).__name__,
                dry_run=False,
                incident_id=request.incident_id,
            )

    def _invoke_with_timeout(
        self,
        callable_impl: ToolCallable,
        parameters: dict[str, Any],
        context: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Invoke a trusted callable and bound caller wait time.

        On timeout, control returns immediately. The worker thread is not killed;
        its eventual completion is abandoned and never published as a tool result.
        """
        future = self._pool.submit(callable_impl, parameters, context)
        try:
            result = future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            with self._abandon_lock:
                self._abandoned_futures.add(future)
            future.add_done_callback(self._on_abandoned_future_done)
            raise

        if not isinstance(result, dict):
            raise ToolValidationError("Tool implementation returned invalid data.")
        return result

    def _on_abandoned_future_done(self, future: Future[Any]) -> None:
        """Discard late completion from a timed-out invocation without publishing."""
        with self._abandon_lock:
            self._abandoned_futures.discard(future)
        _discard_abandoned_future(future)


def _primary_target_type(definition: DefensiveToolDefinition) -> str:
    """Return the primary supported target type for scope checks."""
    if not definition.supported_target_types:
        return "none"
    return definition.supported_target_types[0]


def _assert_path_params_in_scope(
    definition: DefensiveToolDefinition,
    request: ToolExecutionRequest,
    scope: AuthorizedScope,
) -> None:
    """Ensure file-path parameters remain under authorized roots."""
    path_names = {
        item.name
        for item in definition.parameter_schema
        if item.parameter_type == "file-path"
    }
    for name in path_names:
        value = request.normalized_parameters.get(name)
        if isinstance(value, str):
            assert_path_authorized(scope, value)


def _bound_structured_output(
    data: dict[str, Any],
    *,
    maximum_characters: int,
) -> tuple[dict[str, Any], bool]:
    """Bound structured output size without exposing raw exceptions."""
    rendered = str(data)
    if len(rendered) <= maximum_characters:
        return dict(data), False

    # Prefer preserving compact scalar fields when truncating.
    compact: dict[str, Any] = {"output_truncated": True}
    used = len(str(compact))
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            candidate = {**compact, key: value}
            if len(str(candidate)) > maximum_characters:
                continue
            compact = candidate
            used = len(str(compact))
        elif isinstance(value, list):
            bounded_list: list[Any] = []
            for item in value:
                trial = {**compact, key: [*bounded_list, item]}
                if len(str(trial)) > maximum_characters:
                    break
                bounded_list.append(item)
            compact[key] = bounded_list
            used = len(str(compact))
        if used >= maximum_characters:
            break
    return compact, True
