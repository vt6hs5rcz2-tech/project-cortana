"""In-memory workflow run repository abstraction for Milestone 10."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.config import MAX_WORKFLOW_RUNS_RETAINED
from src.workflow_audit import WorkflowAuditEntry, validate_workflow_audit_entry
from src.workflow_common import (
    WORKFLOW_TERMINAL_STATUSES,
    WorkflowStorageError,
    WorkflowValidationError,
)
from src.workflow_result import WorkflowRunResult, validate_workflow_run_result
from src.tool_common import validate_uuid


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


class InMemoryWorkflowRunRepository:
    """Bounded in-memory workflow run store with append-only audit history."""

    def __init__(self, *, max_runs: int = MAX_WORKFLOW_RUNS_RETAINED) -> None:
        if max_runs < 1:
            raise WorkflowValidationError("max_runs must be at least 1.")
        self._max_runs = max_runs
        self._runs: dict[str, WorkflowRunResult] = {}
        self._order: list[str] = []
        self._audit: list[WorkflowAuditEntry] = []

    def save_run(self, run: WorkflowRunResult) -> WorkflowRunResult:
        """Insert or update one validated run without rewriting step history."""
        validated = validate_workflow_run_result(run)
        existing = self._runs.get(validated.run_id)
        if existing is not None:
            if existing.status in WORKFLOW_TERMINAL_STATUSES:
                if validated.status != existing.status:
                    raise WorkflowStorageError(
                        "Terminal workflow runs cannot resume or change status.",
                        user_message=(
                            "Cortana: That workflow run has already finished and "
                            "cannot be resumed."
                        ),
                    )
            self._assert_prefix_immutable_step_history(existing, validated)
            self._runs[validated.run_id] = validated
            return validated

        self._runs[validated.run_id] = validated
        self._order.append(validated.run_id)
        self._enforce_retention()
        return validated

    def _assert_prefix_immutable_step_history(
        self,
        existing: WorkflowRunResult,
        validated: WorkflowRunResult,
    ) -> None:
        """Reject removals, reorders, or replacements of prior step results."""
        if len(validated.step_results) < len(existing.step_results):
            raise WorkflowStorageError(
                "Prior workflow step results cannot be removed.",
                user_message=(
                    "Cortana: Workflow run history cannot be rewritten."
                ),
            )
        for index, prior in enumerate(existing.step_results):
            if validated.step_results[index] != prior:
                raise WorkflowStorageError(
                    "Prior workflow step results cannot be rewritten.",
                    user_message=(
                        "Cortana: Workflow run history cannot be rewritten."
                    ),
                )

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
        return validated

    def list_audit_entries(self) -> list[WorkflowAuditEntry]:
        """Return audit entries in append order."""
        return list(self._audit)

    def _enforce_retention(self) -> None:
        """Drop oldest non-essential completed runs when the retention bound is hit.

        Active/non-terminal runs are preserved preferentially. When only terminal
        runs remain beyond the bound, the oldest terminal runs are removed.
        """
        while len(self._order) > self._max_runs:
            removable_index = None
            for index, run_id in enumerate(self._order):
                run = self._runs.get(run_id)
                if run is not None and run.status in WORKFLOW_TERMINAL_STATUSES:
                    removable_index = index
                    break
            if removable_index is None:
                # Prefer failing closed over silently deleting an active run.
                raise WorkflowStorageError(
                    "Workflow run retention limit reached with active runs present.",
                    user_message=(
                        "Cortana: Workflow run retention limit reached."
                    ),
                )
            removed_id = self._order.pop(removable_index)
            self._runs.pop(removed_id, None)


def assert_no_filesystem_paths(paths: Sequence[object]) -> None:
    """Test helper marker: the in-memory repository never accepts file paths."""
    if paths:
        raise WorkflowValidationError(
            "In-memory workflow repository does not use filesystem paths."
        )
