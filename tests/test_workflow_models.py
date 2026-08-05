"""Definition and model tests for Milestone 10 workflows."""

from __future__ import annotations

import pytest

from src.config import MAX_WORKFLOW_STEPS
from src.tool_registry import build_default_tool_registry
from src.workflow_common import WorkflowValidationError
from src.workflow_definition import (
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_result import (
    create_workflow_run_result,
    create_workflow_step_result,
    transition_workflow_run_result,
)


def test_valid_workflow_construction() -> None:
    registry = build_default_tool_registry()
    workflow = create_workflow_definition(
        name="platform-baseline",
        version="1.0.0",
        description="Valid playbook",
        steps=(
            create_workflow_step_definition(
                step_id="one",
                tool_id="system-summary",
                static_parameters={},
                position=0,
            ),
            create_workflow_step_definition(
                step_id="two",
                tool_id="simulated-log-check",
                static_parameters={"fixture": "auth-noise"},
                position=1,
            ),
        ),
        tool_registry=registry,
    )
    assert len(workflow.steps) == 2
    assert workflow.steps[0].step_id == "one"
    assert workflow.steps[1].tool_id == "simulated-log-check"


def test_empty_workflow_rejected() -> None:
    with pytest.raises(WorkflowValidationError):
        create_workflow_definition(
            name="empty-book",
            version="1.0.0",
            description="Empty",
            steps=(),
        )


def test_duplicate_step_ids_rejected() -> None:
    with pytest.raises(WorkflowValidationError):
        create_workflow_definition(
            name="dup-steps",
            version="1.0.0",
            description="Dup",
            steps=(
                create_workflow_step_definition(
                    step_id="same",
                    tool_id="system-summary",
                    position=0,
                ),
                create_workflow_step_definition(
                    step_id="same",
                    tool_id="system-summary",
                    position=1,
                ),
            ),
        )


def test_invalid_identifiers_and_version_rejected() -> None:
    with pytest.raises(Exception):
        create_workflow_definition(
            name="Bad Name",
            version="1.0.0",
            description="Bad",
            steps=(
                create_workflow_step_definition(
                    step_id="one",
                    tool_id="system-summary",
                    position=0,
                ),
            ),
        )
    with pytest.raises(WorkflowValidationError):
        create_workflow_definition(
            name="good-name",
            version="v1",
            description="Bad version",
            steps=(
                create_workflow_step_definition(
                    step_id="one",
                    tool_id="system-summary",
                    position=0,
                ),
            ),
        )


def test_too_many_steps_rejected() -> None:
    steps = [
        create_workflow_step_definition(
            step_id=f"step-{index}",
            tool_id="system-summary",
            position=index,
        )
        for index in range(MAX_WORKFLOW_STEPS + 1)
    ]
    with pytest.raises(WorkflowValidationError):
        create_workflow_definition(
            name="too-many",
            version="1.0.0",
            description="Too many steps",
            steps=steps,
        )


def test_static_parameter_schema_mismatch_rejected() -> None:
    registry = build_default_tool_registry()
    with pytest.raises(Exception):
        create_workflow_definition(
            name="bad-params",
            version="1.0.0",
            description="Bad params",
            steps=(
                create_workflow_step_definition(
                    step_id="one",
                    tool_id="simulated-log-check",
                    static_parameters={"fixture": "not-a-real-fixture"},
                    position=0,
                ),
            ),
            tool_registry=registry,
        )


def test_immutable_ordered_steps_preserved() -> None:
    workflow = create_workflow_definition(
        name="ordered",
        version="1.0.0",
        description="Order",
        steps=(
            create_workflow_step_definition(
                step_id="step-a",
                tool_id="system-summary",
                position=0,
            ),
            create_workflow_step_definition(
                step_id="step-b",
                tool_id="system-summary",
                position=1,
            ),
        ),
        tool_registry=build_default_tool_registry(),
    )
    assert [step.step_id for step in workflow.steps] == ["step-a", "step-b"]
    with pytest.raises(Exception):
        workflow.steps[0].tool_id = "text-search"  # type: ignore[misc]


def test_terminal_run_status_is_monotonic() -> None:
    run = create_workflow_run_result(
        run_id="11111111-1111-1111-1111-111111111111",
        playbook_name="ordered",
        playbook_version="1.0.0",
        dry_run=True,
        status="pending",
        scope_id="22222222-2222-2222-2222-222222222222",
    )
    running = transition_workflow_run_result(run, status="running")
    completed = transition_workflow_run_result(running, status="completed")
    with pytest.raises(Exception):
        transition_workflow_run_result(completed, status="running")


def test_dry_run_step_cannot_report_execution_success() -> None:
    from src.tool_result import create_tool_execution_result

    tool_result = create_tool_execution_result(
        request_id="33333333-3333-3333-3333-333333333333",
        tool_id="system-summary",
        outcome="succeeded",
        safe_summary="ok",
        dry_run=False,
    )
    with pytest.raises(WorkflowValidationError):
        create_workflow_step_result(
            step_id="one",
            tool_id="system-summary",
            position=0,
            status="completed",
            tool_result=tool_result,
            dry_run=True,
        )
