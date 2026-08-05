"""Incident linkage and feature-flag tests for Milestone 11."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.incident_repository import IncidentRepository
from src.security_incident import SecurityIncident, create_security_incident
from src.tool_scope import create_authorized_scope
from src.workflow_commands import create_default_workflow_services
from src.workflow_executor import WorkflowExecutor
from src.workflow_repository import InMemoryWorkflowRunRepository, JsonWorkflowRunRepository
from src.workflow_request import WorkflowRunRequest, create_workflow_run_request
from src.workflow_result import WorkflowRunResult
from tests.workflow_helpers import (
    incident_repository,
    make_executor,
    make_scope,
    tool_registry,
    tool_repository,
)


def _create_incident(
    incidents: IncidentRepository,
    *,
    title: str = "Linked case",
) -> SecurityIncident:
    incident = create_security_incident(
        title=title,
        summary="Workflow linkage test incident",
        severity="low",
    )
    return incidents.add_incident(incident)


def test_authorized_incident_appends_exactly_one_safe_note(tmp_path: Path) -> None:
    executor, repo, _tools, _workflows, _runs, incidents = make_executor(tmp_path)
    incident = _create_incident(incidents)
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
            incident_id=incident.incident_id,
        )
    )
    assert result.status == "completed"
    assert result.incident_id == incident.incident_id

    notes = incidents.list_notes(incident.incident_id)
    assert len(notes) == 1
    note = notes[0]
    assert note.note_type == "summary"
    assert note.author == "workflow-orchestration"
    assert result.playbook_name in note.text
    assert result.playbook_version in note.text
    assert result.run_id in note.text
    assert result.status in note.text
    assert "structured_data" not in note.text
    assert "C:\\" not in note.text
    assert "approval" not in note.text.lower()
    assert incidents.list_evidence() == []


def test_nonexistent_and_unauthorized_incident_rejected_before_step_one(
    tmp_path: Path,
) -> None:
    executor, repo, _tools, _workflows, runs, incidents = make_executor(tmp_path)
    incident = _create_incident(incidents)
    other = _create_incident(incidents, title="Other")
    scope = create_authorized_scope(
        scope_name="Restricted",
        allowed_tool_ids=["system-summary", "simulated-log-check"],
        allowed_target_types=["none", "system-summary", "mock-log"],
        allowed_incident_ids=[incident.incident_id],
        notes="restricted",
    )
    repo.add_scope(scope)

    missing = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
            incident_id=str(uuid4()),
        )
    )
    assert missing.status == "preflight_failed"
    assert missing.step_results == ()
    assert incidents.list_notes(incident.incident_id) == []

    denied = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
            incident_id=other.incident_id,
        )
    )
    assert denied.status == "preflight_failed"
    assert denied.step_results == ()
    assert incidents.list_notes(other.incident_id) == []
    assert runs.list_runs()


def test_no_note_when_incident_omitted_or_linkage_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, repo, _tools, _workflows, _runs, incidents = make_executor(tmp_path)
    incident = _create_incident(incidents)
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    no_link = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert no_link.status == "completed"
    assert no_link.incident_id is None
    assert incidents.list_notes(incident.incident_id) == []

    monkeypatch.setattr(
        "src.workflow_executor.WORKFLOW_INCIDENT_LINKAGE_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_INCIDENT_LINKAGE_ENABLED",
        False,
    )
    disabled = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
            incident_id=incident.incident_id,
        )
    )
    assert disabled.status == "preflight_failed"
    assert disabled.step_results == ()
    assert incidents.list_notes(incident.incident_id) == []


def test_revalidation_before_note_skips_when_scope_disabled(tmp_path: Path) -> None:
    executor, repo, tools, workflows, runs, incidents = make_executor(tmp_path)
    incident = _create_incident(incidents)
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])

    class NoteGateExecutor(WorkflowExecutor):
        def _maybe_append_incident_note(
            self,
            *,
            request: WorkflowRunRequest,
            run: WorkflowRunResult,
        ) -> None:
            from src.tool_scope import disable_authorized_scope

            current = repo.get_scope(scope.scope_id)
            assert current is not None
            repo.update_scope(disable_authorized_scope(current))
            super()._maybe_append_incident_note(request=request, run=run)

    gated = NoteGateExecutor(
        workflow_registry=workflows,
        tool_registry=tools,
        tool_executor=executor._tool_executor,
        tool_repository=repo,
        run_repository=runs,
        incident_repository=incidents,
    )
    repo.add_scope(scope)
    result = gated.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
            incident_id=incident.incident_id,
        )
    )
    assert result.status == "completed"
    assert incidents.list_notes(incident.incident_id) == []


def test_feature_flag_matrix_persistence_and_linkage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = tool_registry()
    tool_repo = tool_repository(tmp_path)
    incidents = incident_repository(tmp_path)
    from src.tool_executor import DefensiveToolExecutor

    tool_exec = DefensiveToolExecutor(incident_repository=incidents)

    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_RUN_PERSISTENCE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_INCIDENT_LINKAGE_ENABLED",
        False,
    )
    _reg, runs, _exec = create_default_workflow_services(
        tool_registry=tools,
        tool_repository=tool_repo,
        tool_executor=tool_exec,
        incident_repository=incidents,
        workflow_repository_file_path=tmp_path / "wf.json",
    )
    assert isinstance(runs, JsonWorkflowRunRepository)

    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_RUN_PERSISTENCE_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_INCIDENT_LINKAGE_ENABLED",
        True,
    )
    _reg, runs, _exec = create_default_workflow_services(
        tool_registry=tools,
        tool_repository=tool_repo,
        tool_executor=tool_exec,
        incident_repository=incidents,
        persist_runs=False,
    )
    assert isinstance(runs, InMemoryWorkflowRunRepository)

    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_RUN_PERSISTENCE_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "src.workflow_commands.WORKFLOW_INCIDENT_LINKAGE_ENABLED",
        False,
    )
    _reg, runs, executor = create_default_workflow_services(
        tool_registry=tools,
        tool_repository=tool_repo,
        tool_executor=tool_exec,
        incident_repository=incidents,
        persist_runs=False,
    )
    assert isinstance(runs, InMemoryWorkflowRunRepository)
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    tool_repo.add_scope(scope)
    result = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "completed"
    assert isinstance(runs, InMemoryWorkflowRunRepository)


def test_no_evidence_or_custody_from_workflow_linkage(tmp_path: Path) -> None:
    executor, repo, _tools, _workflows, _runs, incidents = make_executor(tmp_path)
    incident = _create_incident(incidents)
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)
    executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
            incident_id=incident.incident_id,
        )
    )
    assert incidents.list_evidence() == []
    loaded = incidents.get_incident(incident.incident_id)
    assert loaded is not None
    assert loaded.evidence_ids == ()
