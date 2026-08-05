"""In-memory workflow run repository tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.workflow_audit import create_workflow_audit_entry
from src.workflow_common import WorkflowStorageError
from src.workflow_repository import InMemoryWorkflowRunRepository
from src.workflow_result import (
    WorkflowRunResult,
    create_workflow_run_result,
    create_workflow_step_result,
    transition_workflow_run_result,
)


def _run(run_id: str, status: str = "pending") -> WorkflowRunResult:
    return create_workflow_run_result(
        run_id=run_id,
        playbook_name="platform-baseline",
        playbook_version="1.0.0",
        dry_run=True,
        status=status,
        scope_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )


def test_save_get_list_and_safe_not_found() -> None:
    repo = InMemoryWorkflowRunRepository(max_runs=10)
    saved = repo.save_run(
        _run("11111111-1111-1111-1111-111111111111", "pending")
    )
    assert repo.get_run(saved.run_id) == saved
    assert repo.get_run("22222222-2222-2222-2222-222222222222") is None
    assert repo.get_run("not-a-uuid") is None
    assert [item.run_id for item in repo.list_runs()] == [saved.run_id]


def test_prior_results_not_rewritten_and_terminal_cannot_resume() -> None:
    repo = InMemoryWorkflowRunRepository()
    run_id = "11111111-1111-1111-1111-111111111111"
    pending = repo.save_run(_run(run_id, "pending"))
    running = repo.save_run(
        transition_workflow_run_result(pending, status="running")
    )
    with_step = create_workflow_run_result(
        run_id=run_id,
        playbook_name=running.playbook_name,
        playbook_version=running.playbook_version,
        dry_run=True,
        status="running",
        scope_id=running.scope_id,
        step_results=(
            create_workflow_step_result(
                step_id="one",
                tool_id="system-summary",
                position=0,
                status="planned",
                dry_run=True,
            ),
        ),
        created_timestamp=running.created_timestamp,
        started_timestamp=running.started_timestamp,
    )
    saved = repo.save_run(with_step)
    completed = repo.save_run(
        transition_workflow_run_result(saved, status="completed")
    )
    with pytest.raises(WorkflowStorageError):
        repo.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=completed.playbook_name,
                playbook_version=completed.playbook_version,
                dry_run=True,
                status="running",
                scope_id=completed.scope_id,
                step_results=completed.step_results,
                created_timestamp=completed.created_timestamp,
            )
        )
    with pytest.raises(WorkflowStorageError):
        repo.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=completed.playbook_name,
                playbook_version=completed.playbook_version,
                dry_run=True,
                status="completed",
                scope_id=completed.scope_id,
                step_results=(),
                created_timestamp=completed.created_timestamp,
                completed_timestamp=completed.completed_timestamp,
            )
        )


def test_nonterminal_save_cannot_rewrite_or_remove_step_history() -> None:
    """Prefix step history must stay immutable while a run is still running."""
    repo = InMemoryWorkflowRunRepository()
    run_id = "44444444-4444-4444-4444-444444444444"
    pending = repo.save_run(_run(run_id, "pending"))
    running = repo.save_run(
        transition_workflow_run_result(pending, status="running")
    )
    first_step = create_workflow_step_result(
        step_id="one",
        tool_id="system-summary",
        position=0,
        status="planned",
        dry_run=True,
    )
    with_first = create_workflow_run_result(
        run_id=run_id,
        playbook_name=running.playbook_name,
        playbook_version=running.playbook_version,
        dry_run=True,
        status="running",
        scope_id=running.scope_id,
        step_results=(first_step,),
        created_timestamp=running.created_timestamp,
        started_timestamp=running.started_timestamp,
    )
    saved = repo.save_run(with_first)

    second_step = create_workflow_step_result(
        step_id="two",
        tool_id="simulated-log-check",
        position=1,
        status="planned",
        dry_run=True,
    )
    appended = create_workflow_run_result(
        run_id=run_id,
        playbook_name=saved.playbook_name,
        playbook_version=saved.playbook_version,
        dry_run=True,
        status="running",
        scope_id=saved.scope_id,
        step_results=(first_step, second_step),
        created_timestamp=saved.created_timestamp,
        started_timestamp=saved.started_timestamp,
    )
    saved = repo.save_run(appended)
    assert len(saved.step_results) == 2

    replaced_first = create_workflow_step_result(
        step_id="one",
        tool_id="system-summary",
        position=0,
        status="failed",
        dry_run=True,
        error_code="Replaced",
        error_message="should not rewrite prior step",
    )
    with pytest.raises(WorkflowStorageError):
        repo.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=saved.playbook_name,
                playbook_version=saved.playbook_version,
                dry_run=True,
                status="running",
                scope_id=saved.scope_id,
                step_results=(replaced_first, second_step),
                created_timestamp=saved.created_timestamp,
                started_timestamp=saved.started_timestamp,
            )
        )

    with pytest.raises(WorkflowStorageError):
        repo.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=saved.playbook_name,
                playbook_version=saved.playbook_version,
                dry_run=True,
                status="running",
                scope_id=saved.scope_id,
                step_results=(second_step, first_step),
                created_timestamp=saved.created_timestamp,
                started_timestamp=saved.started_timestamp,
            )
        )

    with pytest.raises(WorkflowStorageError):
        repo.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=saved.playbook_name,
                playbook_version=saved.playbook_version,
                dry_run=True,
                status="running",
                scope_id=saved.scope_id,
                step_results=(first_step,),
                created_timestamp=saved.created_timestamp,
                started_timestamp=saved.started_timestamp,
            )
        )

    retained = repo.get_run(run_id)
    assert retained is not None
    assert len(retained.step_results) == 2
    assert retained.step_results[0] == first_step


def _completed(run_id: str, completed_timestamp: str) -> WorkflowRunResult:
    pending = _run(run_id, "pending")
    running = transition_workflow_run_result(pending, status="running")
    return transition_workflow_run_result(
        running,
        status="completed",
        completed_timestamp=completed_timestamp,
    )


def test_bounded_retention_and_no_file_creation(tmp_path: Path) -> None:
    repo = InMemoryWorkflowRunRepository(max_runs=2)
    first = repo.save_run(
        _completed(
            "11111111-1111-1111-1111-111111111111",
            "2026-01-01T00:00:00.000000Z",
        )
    )
    second = repo.save_run(
        _completed(
            "22222222-2222-2222-2222-222222222222",
            "2026-01-01T00:00:01.000000Z",
        )
    )
    third = repo.save_run(
        _completed(
            "33333333-3333-3333-3333-333333333333",
            "2026-01-01T00:00:02.000000Z",
        )
    )
    runs = repo.list_runs()
    assert len(runs) == 2
    assert first.run_id not in {item.run_id for item in runs}
    assert {item.run_id for item in runs} == {second.run_id, third.run_id}
    assert list(tmp_path.iterdir()) == []


def test_audit_append_order() -> None:
    repo = InMemoryWorkflowRunRepository()
    first = repo.append_audit_entry(
        create_workflow_audit_entry(
            action="workflow-run-created",
            safe_details={"status": "pending"},
        )
    )
    second = repo.append_audit_entry(
        create_workflow_audit_entry(
            action="workflow-completed",
            safe_details={"status": "completed"},
        )
    )
    entries = repo.list_audit_entries()
    assert [item.audit_id for item in entries] == [first.audit_id, second.audit_id]
