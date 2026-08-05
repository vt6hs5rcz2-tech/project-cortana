"""Parent-side process isolation adapter using subprocess.Popen and JSON IPC.

This is the only module that may import subprocess for tool execution.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from src.config import (
    MAX_PROCESS_DIAGNOSTIC_STDERR_BYTES,
    MAX_PROCESS_DIAGNOSTIC_STDOUT_BYTES,
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
    PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED,
    PROCESS_TERMINATION_CONFIRMATION_TIMEOUT_SECONDS,
    get_default_tool_process_scratch_dir_path,
)
from src.tool_audit import ToolAuditEntry, create_tool_audit_entry
from src.tool_common import ToolPolicyError, utc_timestamp
from src.tool_definition import DefensiveToolDefinition
from src.tool_process_common import (
    ToolProcessError,
    ToolProcessResultRejectedError,
    build_child_environment,
    ensure_tool_process_scratch_dir,
    python_executable,
)
from src.tool_process_envelope import (
    ProcessExecutionResponse,
    create_process_execution_request,
    decode_process_response,
    encode_process_request,
)
from src.tool_request import ToolExecutionRequest
from src.tool_result import ToolExecutionResult, create_tool_execution_result

logger = logging.getLogger("ProjectCortana")

AuditAppender = Callable[[ToolAuditEntry], object]


class SupportsAppendAudit(Protocol):
    """Minimal protocol for appending process-isolation audit entries."""

    def append_audit_entry(self, entry: ToolAuditEntry) -> ToolAuditEntry:
        """Append one validated audit entry."""


@dataclass(frozen=True)
class _DiagnosticCapture:
    stdout_present: bool
    stderr_present: bool
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_chars: int
    stderr_chars: int


@dataclass
class _TerminationDecision:
    """First-observed parent state for timeout/cancellation races."""

    kind: str  # "completed" | "timeout" | "cancelled"
    termination_confirmed: bool | None = None
    exit_code: int | None = None
    response: ProcessExecutionResponse | None = None
    error_class: str | None = None
    safe_summary: str | None = None


class ToolProcessAdapter:
    """Launch one process-isolated eligible/required tool execution."""

    def __init__(
        self,
        *,
        scratch_dir: Path | None = None,
        python_executable_path: str | None = None,
        audit_appender: AuditAppender | None = None,
        popen_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._scratch_dir = scratch_dir
        self._python = python_executable_path or python_executable()
        self._audit_appender = audit_appender
        self._popen_factory = popen_factory or subprocess.Popen

    def execute(
        self,
        *,
        definition: DefensiveToolDefinition,
        request: ToolExecutionRequest,
        started_timestamp: str | None = None,
    ) -> ToolExecutionResult:
        """Execute one already-authorized tool in a child process."""
        started = started_timestamp or utc_timestamp()
        if not PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED:
            raise ToolPolicyError(
                "Process-isolated tool execution is disabled."
            )
        if definition.process_isolation == "prohibited":
            raise ToolPolicyError(
                "This tool is prohibited from process-isolated execution."
            )
        if request.request_status == "cancelled":
            self._audit(
                action="process_cancellation_requested",
                request=request,
                definition=definition,
                safe_details={"status": "pre_launch_cancelled"},
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="cancelled",
                safe_summary="Tool execution was cancelled before process launch.",
                structured_data={
                    "cancelled": True,
                    "process_isolated": True,
                    "pre_launch": True,
                },
                error_class="CancelledError",
                dry_run=False,
                incident_id=request.incident_id,
            )

        correlation_id = str(uuid4())
        envelope = create_process_execution_request(
            correlation_id=correlation_id,
            implementation_identifier=definition.implementation_identifier,
            normalized_parameters=dict(request.normalized_parameters),
            execution_timeout_seconds=definition.timeout_seconds,
            max_output_characters=definition.maximum_output_characters,
        )
        request_bytes = encode_process_request(envelope)

        scratch = ensure_tool_process_scratch_dir(
            self._scratch_dir or get_default_tool_process_scratch_dir_path()
        )
        result_path = scratch / f"result-{correlation_id}.json"
        # Ensure the result channel starts empty and parent-owned.
        result_path.write_bytes(b"")

        env = build_child_environment()
        command = [
            self._python,
            "-m",
            "src.tool_process_runner",
            str(result_path),
        ]
        process: Any | None = None
        decision: _TerminationDecision | None = None
        diagnostics = _DiagnosticCapture(False, False, False, False, 0, 0)
        pid: int | None = None
        began = time.monotonic()

        self._audit(
            action="process_execution_started",
            request=request,
            definition=definition,
            safe_details={
                "correlation_id": correlation_id,
                "implementation_identifier": definition.implementation_identifier,
            },
        )

        try:
            process = self._popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(scratch),
                close_fds=True,
                shell=False,
            )
            pid = int(getattr(process, "pid", 0) or 0) or None
        except Exception as error:  # noqa: BLE001 - launch failures stay contained
            self._audit(
                action="process_launch_failed",
                request=request,
                definition=definition,
                safe_details={
                    "correlation_id": correlation_id,
                    "error_class": type(error).__name__,
                },
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="failed",
                safe_summary="Process-isolated tool launch failed.",
                structured_data={
                    "failed": True,
                    "process_isolated": True,
                    "launch_failed": True,
                },
                error_class=type(error).__name__,
                dry_run=False,
                incident_id=request.incident_id,
            )

        try:
            try:
                stdout_raw, stderr_raw = process.communicate(
                    input=request_bytes,
                    timeout=definition.timeout_seconds,
                )
                diagnostics = _capture_diagnostics(stdout_raw, stderr_raw)
                exit_code = (
                    int(process.returncode) if process.returncode is not None else -1
                )
                try:
                    response = self._read_result_channel(
                        result_path,
                        expected_correlation_id=correlation_id,
                    )
                    decision = _TerminationDecision(
                        kind="completed",
                        termination_confirmed=True,
                        exit_code=exit_code,
                        response=response,
                    )
                except (ToolProcessError, ToolProcessResultRejectedError) as error:
                    self._audit(
                        action="process_result_rejected",
                        request=request,
                        definition=definition,
                        safe_details={
                            "correlation_id": correlation_id,
                            "pid": pid,
                            "exit_code": exit_code,
                            "error_class": type(error).__name__,
                        },
                    )
                    decision = _TerminationDecision(
                        kind="rejected",
                        termination_confirmed=True,
                        exit_code=exit_code,
                        error_class=type(error).__name__,
                        safe_summary="Process-isolated tool result was rejected.",
                    )
            except subprocess.TimeoutExpired:
                decision = self._handle_timeout(
                    process=process,
                    request=request,
                    definition=definition,
                    correlation_id=correlation_id,
                    pid=pid,
                )
                stdout_raw, stderr_raw = self._collect_after_terminate(process)
                diagnostics = _capture_diagnostics(stdout_raw, stderr_raw)
            except KeyboardInterrupt:
                decision = self._handle_cancellation(
                    process=process,
                    request=request,
                    definition=definition,
                    correlation_id=correlation_id,
                    pid=pid,
                )
                stdout_raw, stderr_raw = self._collect_after_terminate(process)
                diagnostics = _capture_diagnostics(stdout_raw, stderr_raw)
        finally:
            self._close_process_streams(process)
            self._unlink_quietly(result_path)

        assert decision is not None
        elapsed = max(0.0, time.monotonic() - began)
        return self._result_from_decision(
            decision=decision,
            definition=definition,
            request=request,
            started=started,
            correlation_id=correlation_id,
            pid=pid,
            elapsed_seconds=elapsed,
            diagnostics=diagnostics,
        )

    def _handle_timeout(
        self,
        *,
        process: Any,
        request: ToolExecutionRequest,
        definition: DefensiveToolDefinition,
        correlation_id: str,
        pid: int | None,
    ) -> _TerminationDecision:
        """Timeout observed first: terminate and never accept a late result."""
        self._audit(
            action="process_timeout_observed",
            request=request,
            definition=definition,
            safe_details={
                "correlation_id": correlation_id,
                "pid": pid,
            },
        )
        confirmed = self._terminate_child(process)
        if confirmed:
            self._audit(
                action="process_timeout_terminated",
                request=request,
                definition=definition,
                safe_details={
                    "correlation_id": correlation_id,
                    "pid": pid,
                    "exit_code": process.returncode,
                    "termination_confirmed": True,
                },
            )
            return _TerminationDecision(
                kind="timeout",
                termination_confirmed=True,
                exit_code=process.returncode,
                error_class="TimeoutError",
                safe_summary=(
                    "Process-isolated tool execution timed out and the child "
                    "process was terminated."
                ),
            )

        self._audit(
            action="process_termination_unconfirmed",
            request=request,
            definition=definition,
            safe_details={
                "correlation_id": correlation_id,
                "pid": pid,
                "termination_confirmed": False,
                "reason": "timeout",
            },
        )
        return _TerminationDecision(
            kind="timeout",
            termination_confirmed=False,
            exit_code=process.returncode,
            error_class="TerminationUnconfirmed",
            safe_summary=(
                "Process-isolated tool execution timed out and child "
                "termination could not be confirmed."
            ),
        )

    def _handle_cancellation(
        self,
        *,
        process: Any,
        request: ToolExecutionRequest,
        definition: DefensiveToolDefinition,
        correlation_id: str,
        pid: int | None,
    ) -> _TerminationDecision:
        """KeyboardInterrupt while a child is active."""
        self._audit(
            action="process_cancellation_requested",
            request=request,
            definition=definition,
            safe_details={
                "correlation_id": correlation_id,
                "pid": pid,
                "status": "keyboard_interrupt",
            },
        )
        confirmed = self._terminate_child(process)
        if confirmed:
            self._audit(
                action="process_cancellation_terminated",
                request=request,
                definition=definition,
                safe_details={
                    "correlation_id": correlation_id,
                    "pid": pid,
                    "exit_code": process.returncode,
                    "termination_confirmed": True,
                },
            )
            return _TerminationDecision(
                kind="cancelled",
                termination_confirmed=True,
                exit_code=process.returncode,
                error_class="CancelledError",
                safe_summary=(
                    "Process-isolated tool execution was cancelled and the child "
                    "process was terminated."
                ),
            )
        self._audit(
            action="process_termination_unconfirmed",
            request=request,
            definition=definition,
            safe_details={
                "correlation_id": correlation_id,
                "pid": pid,
                "termination_confirmed": False,
                "reason": "cancellation",
            },
        )
        return _TerminationDecision(
            kind="cancelled",
            termination_confirmed=False,
            exit_code=process.returncode,
            error_class="TerminationUnconfirmed",
            safe_summary=(
                "Process-isolated tool execution was cancelled and child "
                "termination could not be confirmed."
            ),
        )

    def _terminate_child(self, process: Any) -> bool:
        """Hard-terminate the child and wait for confirmation.

        On Windows, ``kill()`` uses TerminateProcess. On POSIX, ``kill()`` sends
        SIGKILL. There is no graceful shutdown protocol for v1 tools.
        Termination is only attempted when the termination feature flag is enabled
        or when execution isolation is enabled (execution implies termination for
        timeout correctness). The termination flag cannot independently enable
        process execution.
        """
        if (
            not PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED
            and not PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED
        ):
            return False
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=PROCESS_TERMINATION_CONFIRMATION_TIMEOUT_SECONDS)
            return process.poll() is not None
        except Exception:
            try:
                return process.poll() is not None
            except Exception:
                return False

    def _collect_after_terminate(self, process: Any) -> tuple[bytes, bytes]:
        try:
            stdout_raw, stderr_raw = process.communicate(timeout=0.2)
            return stdout_raw or b"", stderr_raw or b""
        except (Exception, KeyboardInterrupt):
            return b"", b""

    def _read_result_channel(
        self,
        result_path: Path,
        *,
        expected_correlation_id: str,
    ) -> ProcessExecutionResponse:
        try:
            raw = result_path.read_bytes()
        except OSError as error:
            raise ToolProcessResultRejectedError(
                "Process result channel could not be read."
            ) from error
        return decode_process_response(
            raw,
            expected_correlation_id=expected_correlation_id,
        )

    def _result_from_decision(
        self,
        *,
        decision: _TerminationDecision,
        definition: DefensiveToolDefinition,
        request: ToolExecutionRequest,
        started: str,
        correlation_id: str,
        pid: int | None,
        elapsed_seconds: float,
        diagnostics: _DiagnosticCapture,
    ) -> ToolExecutionResult:
        diagnostic_details = {
            "diagnostic_stdout_present": diagnostics.stdout_present,
            "diagnostic_stderr_present": diagnostics.stderr_present,
            "diagnostic_output_truncated": (
                diagnostics.stdout_truncated or diagnostics.stderr_truncated
            ),
            "diagnostic_stdout_chars": diagnostics.stdout_chars,
            "diagnostic_stderr_chars": diagnostics.stderr_chars,
            "correlation_id": correlation_id,
            "pid": pid,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "exit_code": decision.exit_code,
            "termination_confirmed": decision.termination_confirmed,
        }

        if decision.kind == "completed":
            assert decision.response is not None
            try:
                response = decision.response
            except ToolProcessResultRejectedError:
                raise
            self._audit(
                action="process_execution_completed",
                request=request,
                definition=definition,
                safe_details={
                    **diagnostic_details,
                    "status": response.outcome,
                },
            )
            if response.outcome == "succeeded":
                return create_tool_execution_result(
                    request_id=request.request_id,
                    tool_id=request.tool_id,
                    started_timestamp=started,
                    outcome="succeeded",
                    safe_summary=(
                        f"Tool '{definition.tool_id}' completed successfully."
                    ),
                    structured_data={
                        **response.structured_data,
                        "process_isolated": True,
                    },
                    output_truncated=response.output_truncated,
                    dry_run=False,
                    incident_id=request.incident_id,
                )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="failed",
                safe_summary="Tool execution failed.",
                structured_data={
                    "failed": True,
                    "process_isolated": True,
                },
                error_class=response.error_class or "ToolProcessError",
                dry_run=False,
                incident_id=request.incident_id,
            )

        if decision.kind == "rejected":
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome="failed",
                safe_summary=decision.safe_summary
                or "Process-isolated tool result was rejected.",
                structured_data={
                    "failed": True,
                    "process_isolated": True,
                    "result_rejected": True,
                },
                error_class=decision.error_class or "ToolProcessResultRejectedError",
                dry_run=False,
                incident_id=request.incident_id,
            )

        if decision.kind == "timeout":
            confirmed = bool(decision.termination_confirmed)
            outcome = (
                "timed_out_terminated" if confirmed else "termination_unconfirmed"
            )
            return create_tool_execution_result(
                request_id=request.request_id,
                tool_id=request.tool_id,
                started_timestamp=started,
                outcome=outcome,
                safe_summary=decision.safe_summary or "Tool execution timed out.",
                structured_data={
                    "timeout": True,
                    "process_isolated": True,
                    "caller_stopped_waiting": True,
                    "worker_forcibly_terminated": confirmed,
                    "termination_confirmed": confirmed,
                },
                error_class=decision.error_class or "TimeoutError",
                dry_run=False,
                incident_id=request.incident_id,
            )

        # cancelled
        confirmed = bool(decision.termination_confirmed)
        return create_tool_execution_result(
            request_id=request.request_id,
            tool_id=request.tool_id,
            started_timestamp=started,
            outcome="cancelled",
            safe_summary=decision.safe_summary or "Tool execution was cancelled.",
            structured_data={
                "cancelled": True,
                "process_isolated": True,
                "termination_confirmed": confirmed,
                "worker_forcibly_terminated": confirmed,
            },
            error_class=decision.error_class or "CancelledError",
            dry_run=False,
            incident_id=request.incident_id,
        )

    def _audit(
        self,
        *,
        action: str,
        request: ToolExecutionRequest,
        definition: DefensiveToolDefinition,
        safe_details: dict[str, Any],
    ) -> None:
        if self._audit_appender is None:
            return
        # Never include parameter values, env, stdout/stderr content, or JSON bodies.
        cleaned = {
            key: value
            for key, value in safe_details.items()
            if value is not None
        }
        entry = create_tool_audit_entry(
            action=action,
            request_id=request.request_id,
            tool_id=definition.tool_id,
            scope_id=request.scope_id,
            safe_details=cleaned,
        )
        try:
            self._audit_appender(entry)
        except Exception:
            logger.info(
                "Process audit append failed action=%s request_id=%s",
                action,
                request.request_id,
            )

    @staticmethod
    def _close_process_streams(process: Any | None) -> None:
        if process is None:
            return
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _capture_diagnostics(stdout_raw: bytes | None, stderr_raw: bytes | None) -> _DiagnosticCapture:
    stdout = stdout_raw or b""
    stderr = stderr_raw or b""
    stdout_truncated = len(stdout) > MAX_PROCESS_DIAGNOSTIC_STDOUT_BYTES
    stderr_truncated = len(stderr) > MAX_PROCESS_DIAGNOSTIC_STDERR_BYTES
    stdout = stdout[:MAX_PROCESS_DIAGNOSTIC_STDOUT_BYTES]
    stderr = stderr[:MAX_PROCESS_DIAGNOSTIC_STDERR_BYTES]
    # Content is intentionally discarded after measuring presence/size.
    return _DiagnosticCapture(
        stdout_present=bool(stdout),
        stderr_present=bool(stderr),
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        stdout_chars=len(stdout),
        stderr_chars=len(stderr),
    )


def bind_audit_appender(
    repository: SupportsAppendAudit | None,
) -> AuditAppender | None:
    """Return an appender bound to a tool-control repository, if provided."""
    if repository is None:
        return None

    def _append(entry: ToolAuditEntry) -> None:
        repository.append_audit_entry(entry)

    return _append
