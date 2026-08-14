"""JSON persistence tests for Milestone 11 workflow repository."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.tool_result import create_tool_execution_result
from src.workflow_audit import create_workflow_audit_entry
from src.workflow_common import (
    ABANDONED_AFTER_RESTART_ERROR_CODE,
    WorkflowStorageError,
)
from src.workflow_repository import (
    PERSISTED_RUN_KEYS,
    PERSISTED_STEP_KEYS,
    JsonWorkflowRunRepository,
)
from src.workflow_result import (
    WorkflowRunResult,
    WorkflowStepResult,
    create_workflow_run_result,
    create_workflow_step_result,
    transition_workflow_run_result,
)


SCOPE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _run(
    *,
    run_id: str | None = None,
    status: str = "pending",
    incident_id: str | None = None,
    step_results: tuple[WorkflowStepResult, ...] = (),
) -> WorkflowRunResult:
    return create_workflow_run_result(
        run_id=run_id or str(uuid4()),
        playbook_name="platform-baseline",
        playbook_version="1.0.0",
        dry_run=True,
        status=status,
        scope_id=SCOPE_ID,
        incident_id=incident_id,
        step_results=step_results,
    )


def _step_with_tool_result(*, position: int = 0) -> WorkflowStepResult:
    request_id = str(uuid4())
    tool_result = create_tool_execution_result(
        request_id=request_id,
        tool_id="system-summary",
        outcome="planned",
        safe_summary="Planned system summary.",
        structured_data={"secret_blob": "should-not-persist", "path": "C:\\secret"},
        dry_run=True,
    )
    return create_workflow_step_result(
        step_id="one" if position == 0 else f"step-{position}",
        tool_id="system-summary",
        position=position,
        status="planned",
        tool_result=tool_result,
        tool_request_id=request_id,
        dry_run=True,
    )


def test_round_trip_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path)
    step = _step_with_tool_result()
    saved = repo.save_run(
        _run(status="completed", step_results=(step,), incident_id=str(uuid4()))
    )
    assert path.exists()

    reloaded = JsonWorkflowRunRepository(path)
    loaded = reloaded.get_run(saved.run_id)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.incident_id == saved.incident_id
    assert len(loaded.step_results) == 1
    assert loaded.step_results[0].tool_result is not None
    assert loaded.step_results[0].tool_result.structured_data == {}
    assert loaded.step_results[0].tool_result.safe_summary == "Planned system summary."


def test_serialized_key_allowlists_exact(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path)
    repo.save_run(_run(status="completed", step_results=(_step_with_tool_result(),)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "version",
        "runs",
        "audit_entries",
        "completed_effect_keys",
    }
    assert payload["completed_effect_keys"] == []
    run_payload = payload["runs"][0]
    assert set(run_payload.keys()) == set(PERSISTED_RUN_KEYS)
    step_payload = run_payload["step_results"][0]
    assert set(step_payload.keys()) == set(PERSISTED_STEP_KEYS)
    assert "structured_data" not in step_payload
    assert "parameters" not in step_payload
    assert "path" not in step_payload
    assert "approval" not in json.dumps(step_payload)
    assert "secret_blob" not in json.dumps(payload)
    assert "C:\\secret" not in json.dumps(payload)


def test_atomic_write_and_interrupted_temp_leaves_original(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path)
    first = repo.save_run(_run(status="completed"))
    original = path.read_text(encoding="utf-8")

    # Simulate a leftover temp file from an interrupted writer.
    temp = path.parent / ".workflow-runs-interrupted.tmp"
    temp.write_text("{broken", encoding="utf-8")
    second = repo.save_run(
        _run(status="completed", run_id=str(uuid4()))
    )
    assert second.run_id != first.run_id
    assert path.read_text(encoding="utf-8") != original
    # Original content remains valid JSON object shape after successful replace.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 2
    assert temp.exists()  # leftover temp is unrelated and unused


def test_forced_atomic_write_failure_leaves_original_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """os.replace failure must not partially replace an existing repository file."""
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path)
    saved = repo.save_run(_run(status="completed"))
    original_bytes = path.read_bytes()
    assert original_bytes

    def _failing_replace(src: object, dst: object) -> None:
        raise OSError("forced atomic replace failure")

    monkeypatch.setattr("src.workflow_repository.os.replace", _failing_replace)

    with pytest.raises(WorkflowStorageError) as error:
        repo.save_run(_run(status="completed", run_id=str(uuid4())))
    assert "could not be updated safely" in error.value.user_message
    assert path.read_bytes() == original_bytes
    assert saved.run_id in {
        item.run_id for item in JsonWorkflowRunRepository(path).list_runs()
    }
    assert not any(path.parent.glob(".workflow-runs-*.tmp"))


def test_invalid_and_truncated_json_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    path.write_text('{"version": 1, "runs": [', encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    repo = JsonWorkflowRunRepository(path)
    with pytest.raises(WorkflowStorageError) as error:
        repo.list_runs()
    assert "left unchanged" in error.value.user_message
    assert path.read_text(encoding="utf-8") == before

    path.write_text("not-json", encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    repo2 = JsonWorkflowRunRepository(path)
    with pytest.raises(WorkflowStorageError):
        repo2.list_runs()
    assert path.read_text(encoding="utf-8") == before


def test_side_effect_markers_round_trip_and_retention(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path, max_completed_effect_keys=2)
    first = "a" * 64
    second = "b" * 64
    third = "c" * 64
    repo.claim_side_effect_key(first, playbook_name="side-effect-once", step_id="mutate")
    repo.claim_side_effect_key(second, playbook_name="side-effect-once", step_id="mutate")
    repo.claim_side_effect_key(third, playbook_name="side-effect-once", step_id="mutate")
    keys = [item.effect_key for item in repo.list_side_effect_markers()]
    assert keys == [second, third]
    reloaded = JsonWorkflowRunRepository(path, max_completed_effect_keys=2)
    assert [item.effect_key for item in reloaded.list_side_effect_markers()] == [
        second,
        third,
    ]


def test_pre_idempotency_root_without_effect_keys_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    path.write_text(
        json.dumps({"version": 1, "runs": [], "audit_entries": []}),
        encoding="utf-8",
    )
    repo = JsonWorkflowRunRepository(path)
    assert repo.list_runs() == []
    assert repo.list_side_effect_markers() == []


def test_wrong_schema_unexpected_keys_and_duplicates_rejected(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    path.write_text(
        json.dumps({"version": 99, "runs": [], "audit_entries": []}),
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")
    with pytest.raises(WorkflowStorageError):
        JsonWorkflowRunRepository(path).list_runs()
    assert path.read_text(encoding="utf-8") == before

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "runs": [],
                "audit_entries": [],
                "extra": True,
            }
        ),
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")
    with pytest.raises(WorkflowStorageError):
        JsonWorkflowRunRepository(path).list_runs()
    assert path.read_text(encoding="utf-8") == before

    run_id = str(uuid4())
    step = {
        "step_id": "one",
        "tool_id": "system-summary",
        "position": 0,
        "status": "planned",
        "dry_run": True,
        "started_timestamp": None,
        "completed_timestamp": None,
        "error_code": None,
        "error_message": None,
        "tool_request_id": None,
        "tool_result_id": None,
        "tool_outcome": None,
        "tool_safe_summary": None,
        "tool_output_truncated": None,
        "tool_started_timestamp": None,
        "tool_completed_timestamp": None,
        "tool_error_class": None,
    }
    run = {
        "run_id": run_id,
        "playbook_name": "platform-baseline",
        "playbook_version": "1.0.0",
        "dry_run": True,
        "status": "completed",
        "scope_id": SCOPE_ID,
        "incident_id": None,
        "step_results": [step],
        "created_timestamp": "2026-01-01T00:00:00.000000Z",
        "started_timestamp": None,
        "completed_timestamp": "2026-01-01T00:00:01.000000Z",
        "error_code": None,
        "error_message": None,
        "cancellation_requested": False,
    }
    path.write_text(
        json.dumps({"version": 1, "runs": [run, run], "audit_entries": []}),
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")
    with pytest.raises(WorkflowStorageError):
        JsonWorkflowRunRepository(path).list_runs()
    assert path.read_text(encoding="utf-8") == before


def test_prefix_rewrite_removal_reorder_and_terminal_rules(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path)
    run_id = str(uuid4())
    pending = repo.save_run(_run(run_id=run_id, status="pending"))
    running = repo.save_run(transition_workflow_run_result(pending, status="running"))
    first = _step_with_tool_result(position=0)
    with_first = create_workflow_run_result(
        run_id=run_id,
        playbook_name=running.playbook_name,
        playbook_version=running.playbook_version,
        dry_run=True,
        status="running",
        scope_id=running.scope_id,
        step_results=(first,),
        created_timestamp=running.created_timestamp,
        started_timestamp=running.started_timestamp,
    )
    saved = repo.save_run(with_first)
    second = create_workflow_step_result(
        step_id="two",
        tool_id="simulated-log-check",
        position=1,
        status="planned",
        dry_run=True,
    )
    saved = repo.save_run(
        create_workflow_run_result(
            run_id=run_id,
            playbook_name=saved.playbook_name,
            playbook_version=saved.playbook_version,
            dry_run=True,
            status="running",
            scope_id=saved.scope_id,
            step_results=(first, second),
            created_timestamp=saved.created_timestamp,
            started_timestamp=saved.started_timestamp,
        )
    )
    completed = repo.save_run(transition_workflow_run_result(saved, status="completed"))

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
                status="failed",
                scope_id=completed.scope_id,
                step_results=completed.step_results,
                created_timestamp=completed.created_timestamp,
                completed_timestamp=completed.completed_timestamp,
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
    with pytest.raises(WorkflowStorageError):
        repo.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=completed.playbook_name,
                playbook_version=completed.playbook_version,
                dry_run=True,
                status="completed",
                scope_id=completed.scope_id,
                step_results=(second, first),
                created_timestamp=completed.created_timestamp,
                completed_timestamp=completed.completed_timestamp,
            )
        )


def test_retention_and_active_run_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path, max_runs=2)
    first = repo.save_run(_run(status="completed"))
    second = repo.save_run(_run(status="completed"))
    third = repo.save_run(_run(status="completed"))
    ids = {item.run_id for item in repo.list_runs()}
    assert first.run_id not in ids
    assert second.run_id in ids
    assert third.run_id in ids

    active_repo = JsonWorkflowRunRepository(tmp_path / "active.json", max_runs=1)
    active_repo.save_run(_run(status="running"))
    with pytest.raises(WorkflowStorageError):
        active_repo.save_run(_run(status="pending"))


def test_audit_append_only_ordered_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path, max_audit_entries=2)
    first = create_workflow_audit_entry(action="workflow-run-created")
    second = create_workflow_audit_entry(action="workflow-completed")
    third = create_workflow_audit_entry(action="workflow-failed")
    repo.append_audit_entry(first)
    repo.append_audit_entry(second)
    repo.append_audit_entry(third)
    entries = repo.list_audit_entries()
    assert len(entries) == 2
    assert entries[0].audit_id == second.audit_id
    assert entries[1].audit_id == third.audit_id


def test_restart_abandons_nonterminal_without_resume(tmp_path: Path) -> None:
    path = tmp_path / "workflow_runs.json"
    repo = JsonWorkflowRunRepository(path)
    run_id = str(uuid4())
    pending = repo.save_run(_run(run_id=run_id, status="pending"))
    running = repo.save_run(transition_workflow_run_result(pending, status="running"))
    step = _step_with_tool_result()
    repo.save_run(
        create_workflow_run_result(
            run_id=run_id,
            playbook_name=running.playbook_name,
            playbook_version=running.playbook_version,
            dry_run=True,
            status="running",
            scope_id=running.scope_id,
            step_results=(step,),
            created_timestamp=running.created_timestamp,
            started_timestamp=running.started_timestamp,
        )
    )

    reloaded = JsonWorkflowRunRepository(path)
    loaded = reloaded.get_run(run_id)
    assert loaded is not None
    assert loaded.status == "abandoned"
    assert loaded.error_code == ABANDONED_AFTER_RESTART_ERROR_CODE
    assert len(loaded.step_results) == 1
    assert loaded.step_results[0].status == "planned"

    # Abandonment is persisted; a third load still shows abandoned.
    again = JsonWorkflowRunRepository(path).get_run(run_id)
    assert again is not None
    assert again.status == "abandoned"

    with pytest.raises(WorkflowStorageError):
        reloaded.save_run(
            create_workflow_run_result(
                run_id=run_id,
                playbook_name=loaded.playbook_name,
                playbook_version=loaded.playbook_version,
                dry_run=True,
                status="running",
                scope_id=loaded.scope_id,
                step_results=loaded.step_results,
                created_timestamp=loaded.created_timestamp,
            )
        )
