"""Workflow run repository abstractions for in-memory and JSON persistence.

One application instance should access a workflow repository file at a time.
Atomic writes protect against partial-write corruption within a single writer.
They do not provide full concurrent multi-process coordination; last-writer-wins
behavior is not a supported operating mode. Cross-process locking is not
implemented in this milestone.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from src.config import (
    MAX_WORKFLOW_AUDIT_ENTRIES_RETAINED,
    MAX_WORKFLOW_RUNS_RETAINED,
    WORKFLOW_REPOSITORY_SCHEMA_VERSION,
)
from src.tool_common import ToolExecutionOutcome, utc_timestamp, validate_uuid
from src.tool_result import ToolExecutionResult, validate_tool_execution_result
from src.workflow_audit import WorkflowAuditEntry, validate_workflow_audit_entry
from src.workflow_common import (
    ABANDONED_AFTER_RESTART_ERROR_CODE,
    ABANDONED_AFTER_RESTART_MESSAGE,
    WORKFLOW_TERMINAL_STATUSES,
    WorkflowAuditAction,
    WorkflowRunStatus,
    WorkflowStorageError,
    WorkflowValidationError,
)
from src.workflow_result import (
    WorkflowRunResult,
    WorkflowStepResult,
    create_workflow_run_result,
    create_workflow_step_result,
    validate_workflow_run_result,
)

logger = logging.getLogger("ProjectCortana")

WORKFLOW_LOAD_ERROR_MESSAGE = (
    "Cortana: Workflow repository data could not be loaded safely. "
    "The existing repository file was left unchanged for inspection."
)
WORKFLOW_SAVE_ERROR_MESSAGE = (
    "Cortana: Workflow repository data could not be updated safely."
)

# Explicit safe persistence allowlists. Unexpected keys fail closed.
PERSISTED_ROOT_KEYS: frozenset[str] = frozenset(
    {"version", "runs", "audit_entries"}
)
PERSISTED_RUN_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "playbook_name",
        "playbook_version",
        "dry_run",
        "status",
        "scope_id",
        "incident_id",
        "step_results",
        "created_timestamp",
        "started_timestamp",
        "completed_timestamp",
        "error_code",
        "error_message",
        "cancellation_requested",
    }
)
PERSISTED_STEP_KEYS: frozenset[str] = frozenset(
    {
        "step_id",
        "tool_id",
        "position",
        "status",
        "dry_run",
        "started_timestamp",
        "completed_timestamp",
        "error_code",
        "error_message",
        "tool_request_id",
        "tool_result_id",
        "tool_outcome",
        "tool_safe_summary",
        "tool_output_truncated",
        "tool_started_timestamp",
        "tool_completed_timestamp",
        "tool_error_class",
    }
)
PERSISTED_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "audit_id",
        "timestamp",
        "action",
        "actor",
        "run_id",
        "playbook_name",
        "step_id",
        "tool_id",
        "scope_id",
        "approval_id",
        "result_id",
        "safe_details",
    }
)


class WorkflowRunRepository(Protocol):
    """Persistence boundary for workflow runs and workflow audit entries."""

    def save_run(self, run: WorkflowRunResult) -> WorkflowRunResult:
        """Insert or replace one workflow run record."""

    def get_run(self, run_id: str) -> WorkflowRunResult | None:
        """Return one run by ID, or None when missing."""

    def list_runs(self) -> list[WorkflowRunResult]:
        """Return retained runs in deterministic insertion order."""

    def append_audit_entry(self, entry: WorkflowAuditEntry) -> WorkflowAuditEntry:
        """Append one workflow audit entry."""

    def list_audit_entries(self) -> list[WorkflowAuditEntry]:
        """Return audit entries in append order."""


@dataclass
class _WorkflowRepositoryState:
    """In-memory snapshot of the workflow repository."""

    runs: dict[str, WorkflowRunResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    audit_entries: list[WorkflowAuditEntry] = field(default_factory=list)


class InMemoryWorkflowRunRepository:
    """Bounded in-memory workflow run store with append-only audit history."""

    def __init__(
        self,
        *,
        max_runs: int = MAX_WORKFLOW_RUNS_RETAINED,
        max_audit_entries: int = MAX_WORKFLOW_AUDIT_ENTRIES_RETAINED,
    ) -> None:
        if max_runs < 1:
            raise WorkflowValidationError("max_runs must be at least 1.")
        if max_audit_entries < 1:
            raise WorkflowValidationError("max_audit_entries must be at least 1.")
        self._max_runs = max_runs
        self._max_audit_entries = max_audit_entries
        self._runs: dict[str, WorkflowRunResult] = {}
        self._order: list[str] = []
        self._audit: list[WorkflowAuditEntry] = []

    def save_run(self, run: WorkflowRunResult) -> WorkflowRunResult:
        """Insert or update one validated run without rewriting step history."""
        validated = validate_workflow_run_result(run)
        existing = self._runs.get(validated.run_id)
        if existing is not None:
            _assert_immutable_run_identity(existing, validated)
            _assert_terminal_monotonicity(existing, validated)
            _assert_prefix_immutable_step_history(existing, validated)
            self._runs[validated.run_id] = validated
            return validated

        self._runs[validated.run_id] = validated
        self._order.append(validated.run_id)
        self._enforce_retention()
        return validated

    def get_run(self, run_id: str) -> WorkflowRunResult | None:
        """Return one run by ID, or None when missing."""
        try:
            cleaned = validate_uuid(run_id, field_name="Run ID")
        except Exception:
            return None
        return self._runs.get(cleaned)

    def list_runs(self) -> list[WorkflowRunResult]:
        """Return retained runs in deterministic insertion order."""
        return [self._runs[run_id] for run_id in self._order if run_id in self._runs]

    def append_audit_entry(self, entry: WorkflowAuditEntry) -> WorkflowAuditEntry:
        """Append one validated workflow audit entry."""
        validated = validate_workflow_audit_entry(entry)
        self._audit.append(validated)
        self._enforce_audit_retention()
        return validated

    def list_audit_entries(self) -> list[WorkflowAuditEntry]:
        """Return audit entries in append order."""
        return list(self._audit)

    def _enforce_retention(self) -> None:
        """Drop oldest terminal runs when the retention bound is hit.

        Active/non-terminal runs are preserved preferentially. When only
        non-terminal runs remain beyond the bound, fail closed.
        """
        while len(self._order) > self._max_runs:
            removable_index = None
            for index, run_id in enumerate(self._order):
                run = self._runs.get(run_id)
                if run is not None and run.status in WORKFLOW_TERMINAL_STATUSES:
                    removable_index = index
                    break
            if removable_index is None:
                raise WorkflowStorageError(
                    "Workflow run retention limit reached with active runs present.",
                    user_message=(
                        "Cortana: Workflow run retention limit reached."
                    ),
                )
            removed_id = self._order.pop(removable_index)
            self._runs.pop(removed_id, None)

    def _enforce_audit_retention(self) -> None:
        """Drop oldest audit entries when the audit retention bound is hit."""
        overflow = len(self._audit) - self._max_audit_entries
        if overflow > 0:
            del self._audit[:overflow]


class JsonWorkflowRunRepository:
    """JSON-backed workflow run store with atomic writes and safe projections."""

    def __init__(
        self,
        file_path: Path,
        *,
        max_runs: int = MAX_WORKFLOW_RUNS_RETAINED,
        max_audit_entries: int = MAX_WORKFLOW_AUDIT_ENTRIES_RETAINED,
    ) -> None:
        if max_runs < 1:
            raise WorkflowValidationError("max_runs must be at least 1.")
        if max_audit_entries < 1:
            raise WorkflowValidationError("max_audit_entries must be at least 1.")
        self._file_path = file_path
        self._max_runs = max_runs
        self._max_audit_entries = max_audit_entries
        self._state: _WorkflowRepositoryState | None = None
        self._load_error: WorkflowStorageError | None = None

    @property
    def file_path(self) -> Path:
        """Return the workflow repository JSON file path."""
        return self._file_path

    def save_run(self, run: WorkflowRunResult) -> WorkflowRunResult:
        """Insert or update one validated run without rewriting step history."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_workflow_run_result(run)
        existing = state.runs.get(validated.run_id)
        if existing is not None:
            _assert_immutable_run_identity(existing, validated)
            _assert_terminal_monotonicity(existing, validated)
            _assert_prefix_immutable_step_history(existing, validated)
            state.runs[validated.run_id] = validated
        else:
            state.runs[validated.run_id] = validated
            state.order.append(validated.run_id)
            _enforce_run_retention(state, self._max_runs)
        self._persist(state)
        return validated

    def get_run(self, run_id: str) -> WorkflowRunResult | None:
        """Return one run by ID, or None when missing."""
        try:
            cleaned = validate_uuid(run_id, field_name="Run ID")
        except Exception:
            return None
        return self._ensure_loaded().runs.get(cleaned)

    def list_runs(self) -> list[WorkflowRunResult]:
        """Return retained runs in deterministic insertion order."""
        state = self._ensure_loaded()
        return [state.runs[run_id] for run_id in state.order if run_id in state.runs]

    def append_audit_entry(self, entry: WorkflowAuditEntry) -> WorkflowAuditEntry:
        """Append one validated workflow audit entry."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_workflow_audit_entry(entry)
        if any(item.audit_id == validated.audit_id for item in state.audit_entries):
            raise WorkflowStorageError(
                "Duplicate workflow audit ID rejected.",
                user_message=WORKFLOW_SAVE_ERROR_MESSAGE,
            )
        state.audit_entries.append(validated)
        _enforce_audit_retention(state, self._max_audit_entries)
        self._persist(state)
        return validated

    def list_audit_entries(self) -> list[WorkflowAuditEntry]:
        """Return audit entries in append order."""
        return list(self._ensure_loaded().audit_entries)

    def _ensure_loaded(self) -> _WorkflowRepositoryState:
        if self._load_error is not None:
            raise self._load_error
        if self._state is not None:
            return self._state

        if not self._file_path.exists():
            self._state = _WorkflowRepositoryState()
            return self._state

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                raise WorkflowStorageError(
                    "Workflow repository file is empty.",
                    user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
                )
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise WorkflowStorageError(
                    "Workflow repository root must be an object.",
                    user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
                )
            state = self._deserialize_state(payload)
            abandoned_any = _abandon_stale_nonterminal_runs(state)
            self._state = state
            if abandoned_any:
                # Persist abandonment so restart never leaves stale running status.
                self._persist(state)
            return self._state
        except WorkflowStorageError as error:
            self._load_error = WorkflowStorageError(
                str(error),
                user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
            )
            raise self._load_error from error
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            WorkflowValidationError,
        ) as error:
            logger.error(
                "Workflow repository load failed error_type=%s",
                type(error).__name__,
            )
            self._load_error = WorkflowStorageError(
                "Workflow repository load failed.",
                user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
            )
            raise self._load_error from error

    def _deserialize_state(self, payload: dict[str, Any]) -> _WorkflowRepositoryState:
        _assert_exact_keys(payload, PERSISTED_ROOT_KEYS, context="repository root")
        version = payload.get("version")
        if version != WORKFLOW_REPOSITORY_SCHEMA_VERSION:
            logger.error("Workflow repository has unsupported schema version.")
            raise WorkflowStorageError(
                "Unsupported workflow repository schema version.",
                user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
            )

        runs_payload = payload.get("runs")
        if not isinstance(runs_payload, list):
            raise WorkflowStorageError(
                "Workflow repository runs must be a list.",
                user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
            )
        audit_payload = payload.get("audit_entries")
        if not isinstance(audit_payload, list):
            raise WorkflowStorageError(
                "Workflow repository audit_entries must be a list.",
                user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
            )

        runs: dict[str, WorkflowRunResult] = {}
        order: list[str] = []
        for item in runs_payload:
            if not isinstance(item, dict):
                raise WorkflowStorageError(
                    "Workflow run record must be an object.",
                    user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
                )
            run = _deserialize_run(item)
            if run.run_id in runs:
                raise WorkflowStorageError(
                    "Duplicate workflow run ID rejected.",
                    user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
                )
            runs[run.run_id] = run
            order.append(run.run_id)

        audit_entries: list[WorkflowAuditEntry] = []
        seen_audit_ids: set[str] = set()
        for item in audit_payload:
            if not isinstance(item, dict):
                raise WorkflowStorageError(
                    "Workflow audit record must be an object.",
                    user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
                )
            entry = _deserialize_audit(item)
            if entry.audit_id in seen_audit_ids:
                raise WorkflowStorageError(
                    "Duplicate workflow audit ID rejected.",
                    user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
                )
            seen_audit_ids.add(entry.audit_id)
            audit_entries.append(entry)

        return _WorkflowRepositoryState(
            runs=runs,
            order=order,
            audit_entries=audit_entries,
        )

    def _persist(self, state: _WorkflowRepositoryState) -> None:
        payload = {
            "version": WORKFLOW_REPOSITORY_SCHEMA_VERSION,
            "runs": [_serialize_run(state.runs[run_id]) for run_id in state.order],
            "audit_entries": [
                _serialize_audit(entry) for entry in state.audit_entries
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._state = self._clone_state(state)

    def _atomic_write(self, content: str) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".workflow-runs-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._file_path)
            temporary_path = None
        except OSError as error:
            logger.error(
                "Unable to write workflow repository due to OS error type: %s",
                type(error).__name__,
            )
            raise WorkflowStorageError(
                "Workflow repository save failed.",
                user_message=WORKFLOW_SAVE_ERROR_MESSAGE,
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _clone_state(self, state: _WorkflowRepositoryState) -> _WorkflowRepositoryState:
        return _WorkflowRepositoryState(
            runs=dict(state.runs),
            order=list(state.order),
            audit_entries=list(state.audit_entries),
        )


def assert_no_filesystem_paths(paths: Sequence[object]) -> None:
    """Test helper marker: the in-memory repository never accepts file paths."""
    if paths:
        raise WorkflowValidationError(
            "In-memory workflow repository does not use filesystem paths."
        )


def _assert_immutable_run_identity(
    existing: WorkflowRunResult,
    validated: WorkflowRunResult,
) -> None:
    if existing.incident_id != validated.incident_id:
        raise WorkflowStorageError(
            "Workflow run incident_id is immutable.",
            user_message="Cortana: Workflow run history cannot be rewritten.",
        )
    if existing.scope_id != validated.scope_id:
        raise WorkflowStorageError(
            "Workflow run scope_id is immutable.",
            user_message="Cortana: Workflow run history cannot be rewritten.",
        )
    if existing.playbook_name != validated.playbook_name:
        raise WorkflowStorageError(
            "Workflow run playbook_name is immutable.",
            user_message="Cortana: Workflow run history cannot be rewritten.",
        )
    if existing.dry_run != validated.dry_run:
        raise WorkflowStorageError(
            "Workflow run dry_run flag is immutable.",
            user_message="Cortana: Workflow run history cannot be rewritten.",
        )


def _assert_terminal_monotonicity(
    existing: WorkflowRunResult,
    validated: WorkflowRunResult,
) -> None:
    if existing.status in WORKFLOW_TERMINAL_STATUSES:
        if validated.status != existing.status:
            raise WorkflowStorageError(
                "Terminal workflow runs cannot resume or change status.",
                user_message=(
                    "Cortana: That workflow run has already finished and "
                    "cannot be resumed."
                ),
            )


def _assert_prefix_immutable_step_history(
    existing: WorkflowRunResult,
    validated: WorkflowRunResult,
) -> None:
    """Reject removals, reorders, or replacements of prior step results."""
    if len(validated.step_results) < len(existing.step_results):
        raise WorkflowStorageError(
            "Prior workflow step results cannot be removed.",
            user_message="Cortana: Workflow run history cannot be rewritten.",
        )
    for index, prior in enumerate(existing.step_results):
        if validated.step_results[index] != prior:
            raise WorkflowStorageError(
                "Prior workflow step results cannot be rewritten.",
                user_message="Cortana: Workflow run history cannot be rewritten.",
            )


def _enforce_run_retention(state: _WorkflowRepositoryState, max_runs: int) -> None:
    while len(state.order) > max_runs:
        removable_index = None
        for index, run_id in enumerate(state.order):
            run = state.runs.get(run_id)
            if run is not None and run.status in WORKFLOW_TERMINAL_STATUSES:
                removable_index = index
                break
        if removable_index is None:
            raise WorkflowStorageError(
                "Workflow run retention limit reached with active runs present.",
                user_message="Cortana: Workflow run retention limit reached.",
            )
        removed_id = state.order.pop(removable_index)
        state.runs.pop(removed_id, None)


def _enforce_audit_retention(
    state: _WorkflowRepositoryState,
    max_audit_entries: int,
) -> None:
    overflow = len(state.audit_entries) - max_audit_entries
    if overflow > 0:
        del state.audit_entries[:overflow]


def _abandon_stale_nonterminal_runs(state: _WorkflowRepositoryState) -> bool:
    """Convert non-terminal persisted runs into abandoned terminal runs.

    Never resumes execution, never recreates approvals, and never retries.
    """
    changed = False
    completed_at = utc_timestamp()
    for run_id, run in list(state.runs.items()):
        if run.status in WORKFLOW_TERMINAL_STATUSES:
            continue
        abandoned = create_workflow_run_result(
            run_id=run.run_id,
            playbook_name=run.playbook_name,
            playbook_version=run.playbook_version,
            dry_run=run.dry_run,
            status="abandoned",
            scope_id=run.scope_id,
            incident_id=run.incident_id,
            step_results=run.step_results,
            created_timestamp=run.created_timestamp,
            started_timestamp=run.started_timestamp,
            completed_timestamp=completed_at,
            error_code=ABANDONED_AFTER_RESTART_ERROR_CODE,
            error_message=ABANDONED_AFTER_RESTART_MESSAGE,
            cancellation_requested=run.cancellation_requested,
        )
        state.runs[run_id] = abandoned
        changed = True
    return changed


def _assert_exact_keys(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    keys = frozenset(payload.keys())
    unexpected = keys - allowed
    if unexpected:
        raise WorkflowStorageError(
            f"Unexpected {context} keys rejected.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )
    missing = allowed - keys
    if missing:
        raise WorkflowStorageError(
            f"Missing {context} keys rejected.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )


def _serialize_run(run: WorkflowRunResult) -> dict[str, Any]:
    payload = {
        "run_id": run.run_id,
        "playbook_name": run.playbook_name,
        "playbook_version": run.playbook_version,
        "dry_run": run.dry_run,
        "status": run.status,
        "scope_id": run.scope_id,
        "incident_id": run.incident_id,
        "step_results": [_serialize_step(step) for step in run.step_results],
        "created_timestamp": run.created_timestamp,
        "started_timestamp": run.started_timestamp,
        "completed_timestamp": run.completed_timestamp,
        "error_code": run.error_code,
        "error_message": run.error_message,
        "cancellation_requested": run.cancellation_requested,
    }
    _assert_exact_keys(payload, PERSISTED_RUN_KEYS, context="run record")
    return payload


def _serialize_step(step: WorkflowStepResult) -> dict[str, Any]:
    tool = step.tool_result
    payload = {
        "step_id": step.step_id,
        "tool_id": step.tool_id,
        "position": step.position,
        "status": step.status,
        "dry_run": step.dry_run,
        "started_timestamp": step.started_timestamp,
        "completed_timestamp": step.completed_timestamp,
        "error_code": step.error_code,
        "error_message": step.error_message,
        "tool_request_id": step.tool_request_id,
        "tool_result_id": None if tool is None else tool.result_id,
        "tool_outcome": None if tool is None else tool.outcome,
        "tool_safe_summary": None if tool is None else tool.safe_summary,
        "tool_output_truncated": None if tool is None else tool.output_truncated,
        "tool_started_timestamp": None if tool is None else tool.started_timestamp,
        "tool_completed_timestamp": None if tool is None else tool.completed_timestamp,
        "tool_error_class": None if tool is None else tool.error_class,
    }
    _assert_exact_keys(payload, PERSISTED_STEP_KEYS, context="step record")
    return payload


def _serialize_audit(entry: WorkflowAuditEntry) -> dict[str, Any]:
    payload = {
        "audit_id": entry.audit_id,
        "timestamp": entry.timestamp,
        "action": entry.action,
        "actor": entry.actor,
        "run_id": entry.run_id,
        "playbook_name": entry.playbook_name,
        "step_id": entry.step_id,
        "tool_id": entry.tool_id,
        "scope_id": entry.scope_id,
        "approval_id": entry.approval_id,
        "result_id": entry.result_id,
        "safe_details": dict(entry.safe_details),
    }
    _assert_exact_keys(payload, PERSISTED_AUDIT_KEYS, context="audit record")
    return payload


def _deserialize_run(payload: dict[str, Any]) -> WorkflowRunResult:
    _assert_exact_keys(payload, PERSISTED_RUN_KEYS, context="run record")
    steps_payload = payload.get("step_results")
    if not isinstance(steps_payload, list):
        raise WorkflowStorageError(
            "Workflow step_results must be a list.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )
    steps = tuple(_deserialize_step(item) for item in steps_payload)
    for index, step in enumerate(steps):
        if step.position != index:
            raise WorkflowStorageError(
                "Workflow step ordering is invalid.",
                user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
            )
    return validate_workflow_run_result(
        WorkflowRunResult(
            run_id=str(payload["run_id"]),
            playbook_name=str(payload["playbook_name"]),
            playbook_version=str(payload["playbook_version"]),
            dry_run=bool(payload["dry_run"]),
            status=cast(WorkflowRunStatus, str(payload["status"])),
            scope_id=str(payload["scope_id"]),
            incident_id=payload["incident_id"],
            step_results=steps,
            created_timestamp=str(payload["created_timestamp"]),
            started_timestamp=payload["started_timestamp"],
            completed_timestamp=payload["completed_timestamp"],
            error_code=payload["error_code"],
            error_message=payload["error_message"],
            cancellation_requested=bool(payload["cancellation_requested"]),
        )
    )


def _deserialize_step(payload: object) -> WorkflowStepResult:
    if not isinstance(payload, dict):
        raise WorkflowStorageError(
            "Workflow step record must be an object.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )
    _assert_exact_keys(payload, PERSISTED_STEP_KEYS, context="step record")
    tool_result = _deserialize_tool_projection(payload)
    return create_workflow_step_result(
        step_id=str(payload["step_id"]),
        tool_id=str(payload["tool_id"]),
        position=int(payload["position"]),
        status=str(payload["status"]),
        tool_result=tool_result,
        tool_request_id=payload["tool_request_id"],
        started_timestamp=payload["started_timestamp"],
        completed_timestamp=payload["completed_timestamp"],
        error_code=payload["error_code"],
        error_message=payload["error_message"],
        dry_run=bool(payload["dry_run"]),
    )


def _deserialize_tool_projection(
    payload: dict[str, Any],
) -> ToolExecutionResult | None:
    outcome = payload.get("tool_outcome")
    if outcome is None:
        return None
    request_id = payload.get("tool_request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        request_id = str(uuid4())
    result_id = payload.get("tool_result_id")
    if not isinstance(result_id, str) or not result_id.strip():
        result_id = str(uuid4())
    started = payload.get("tool_started_timestamp")
    completed = payload.get("tool_completed_timestamp")
    if not isinstance(started, str) or not isinstance(completed, str):
        raise WorkflowStorageError(
            "Persisted tool projection timestamps are invalid.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )
    summary = payload.get("tool_safe_summary")
    if not isinstance(summary, str):
        raise WorkflowStorageError(
            "Persisted tool projection summary is invalid.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )
    # Reconstruct only the allowlisted projection. structured_data is always empty.
    return validate_tool_execution_result(
        ToolExecutionResult(
            result_id=result_id,
            request_id=request_id,
            tool_id=str(payload["tool_id"]),
            started_timestamp=started,
            completed_timestamp=completed,
            outcome=cast(ToolExecutionOutcome, outcome),
            safe_summary=summary,
            structured_data={},
            output_truncated=bool(payload["tool_output_truncated"]),
            error_class=payload["tool_error_class"],
            dry_run=bool(payload["dry_run"]),
            incident_id=None,
        )
    )


def _deserialize_audit(payload: dict[str, Any]) -> WorkflowAuditEntry:
    _assert_exact_keys(payload, PERSISTED_AUDIT_KEYS, context="audit record")
    details = payload.get("safe_details")
    if not isinstance(details, dict):
        raise WorkflowStorageError(
            "Workflow audit safe_details must be an object.",
            user_message=WORKFLOW_LOAD_ERROR_MESSAGE,
        )
    return validate_workflow_audit_entry(
        WorkflowAuditEntry(
            audit_id=str(payload["audit_id"]),
            timestamp=str(payload["timestamp"]),
            action=cast(WorkflowAuditAction, payload["action"]),
            actor=str(payload["actor"]),
            run_id=payload["run_id"],
            playbook_name=payload["playbook_name"],
            step_id=payload["step_id"],
            tool_id=payload["tool_id"],
            scope_id=payload["scope_id"],
            approval_id=payload["approval_id"],
            result_id=payload["result_id"],
            safe_details=dict(details),
        )
    )
