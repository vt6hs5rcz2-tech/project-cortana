"""Audit safety tests for Milestone 10 workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tool_audit import FORBIDDEN_AUDIT_DETAIL_KEYS
from src.workflow_audit import create_workflow_audit_entry
from src.tool_common import ToolValidationError
from src.workflow_request import create_workflow_run_request
from tests.workflow_helpers import make_executor, make_scope, workflow_registry


def test_lifecycle_audit_order_and_forbidden_keys(tmp_path: Path) -> None:
    executor, repo, _tools, _workflows, runs = make_executor(
        tmp_path,
        workflows=workflow_registry(),
    )
    scope = make_scope(tool_ids=["system-summary", "simulated-log-check"])
    repo.add_scope(scope)

    result = executor.run(
        create_workflow_run_request(
            playbook_name="platform-baseline",
            scope_id=scope.scope_id,
            dry_run=True,
        )
    )
    assert result.status == "completed"
    actions = [entry.action for entry in runs.list_audit_entries()]
    assert actions[0] == "workflow-run-created"
    assert "workflow-preflight-accepted" in actions
    assert actions.count("workflow-step-attempt") == 2
    assert actions.count("workflow-step-dry-run") == 2
    assert actions[-1] == "workflow-completed"

    for entry in runs.list_audit_entries():
        for key in entry.safe_details:
            assert key.lower() not in FORBIDDEN_AUDIT_DETAIL_KEYS
            assert key.lower() not in {
                "path",
                "parameters",
                "query",
                "token",
                "secret",
            }


def test_forbidden_audit_detail_keys_rejected() -> None:
    with pytest.raises(ToolValidationError):
        create_workflow_audit_entry(
            action="workflow-run-created",
            safe_details={"parameters": {"path": "/secret"}},
        )
    with pytest.raises(ToolValidationError):
        create_workflow_audit_entry(
            action="workflow-failed",
            safe_details={"exception": "traceback text"},
        )
