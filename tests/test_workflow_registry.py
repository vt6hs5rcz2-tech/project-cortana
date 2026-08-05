"""Registry tests for Milestone 10 workflow playbooks."""

from __future__ import annotations

import pytest

from src.tool_definition import create_tool_definition
from src.tool_registry import build_default_tool_registry
from src.workflow_builtins import build_default_workflow_registry
from src.workflow_common import WorkflowValidationError
from src.workflow_definition import (
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_registry import WorkflowRegistry


def test_builtin_registry_deterministic_ordering_and_get_require_list() -> None:
    registry = build_default_workflow_registry()
    names = [item.name for item in registry.list_all()]
    assert names == ["platform-baseline", "mock-log-triage"]
    assert registry.count() == 2
    assert registry.enabled_count() == 2
    assert registry.require("platform-baseline").version == "1.0.0"
    assert registry.get("missing-playbook") is None
    with pytest.raises(WorkflowValidationError):
        registry.require("missing-playbook")


def test_duplicate_name_version_rejected() -> None:
    tools = build_default_tool_registry()
    registry = WorkflowRegistry(tool_registry=tools)
    playbook = create_workflow_definition(
        name="dup-book",
        version="1.0.0",
        description="One",
        steps=(
            create_workflow_step_definition(
                step_id="one",
                tool_id="system-summary",
                position=0,
            ),
        ),
        tool_registry=tools,
    )
    registry.register(playbook)
    with pytest.raises(WorkflowValidationError):
        registry.register(playbook)


def test_unknown_disabled_prohibited_and_non_defensive_tools_rejected() -> None:
    tools = build_default_tool_registry()
    registry = WorkflowRegistry(tool_registry=tools)

    with pytest.raises(WorkflowValidationError):
        registry.register(
            create_workflow_definition(
                name="unknown-tool-book",
                version="1.0.0",
                description="Unknown",
                steps=(
                    create_workflow_step_definition(
                        step_id="one",
                        tool_id="not-a-real-tool",
                        position=0,
                    ),
                ),
            )
        )

    disabled = create_tool_definition(
        tool_id="disabled-tool",
        name="Disabled",
        description="Disabled tool",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        enabled=False,
        implementation_identifier="impl_disabled_workflow_tool",
    )
    tools.register(disabled)
    with pytest.raises(WorkflowValidationError):
        registry.register(
            create_workflow_definition(
                name="disabled-book",
                version="1.0.0",
                description="Disabled",
                steps=(
                    create_workflow_step_definition(
                        step_id="one",
                        tool_id="disabled-tool",
                        position=0,
                    ),
                ),
            )
        )

    prohibited = create_tool_definition(
        tool_id="prohibited-tool",
        name="Prohibited",
        description="Prohibited tool",
        category="diagnostics",
        version="1.0.0",
        risk_level="prohibited",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=True,
        enabled=False,
        implementation_identifier="impl_prohibited_workflow_tool",
    )
    tools.register(prohibited)
    with pytest.raises(Exception):
        registry.register(
            create_workflow_definition(
                name="prohibited-book",
                version="1.0.0",
                description="Prohibited",
                steps=(
                    create_workflow_step_definition(
                        step_id="one",
                        tool_id="prohibited-tool",
                        position=0,
                    ),
                ),
            )
        )

    future = create_tool_definition(
        tool_id="future-external-tool",
        name="Future",
        description="Future external",
        category="diagnostics",
        version="1.0.0",
        risk_level="low",
        execution_mode="future-external",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=True,
        enabled=False,
        implementation_identifier="impl_future_workflow_tool",
    )
    tools.register(future)
    with pytest.raises(Exception):
        registry.register(
            create_workflow_definition(
                name="future-book",
                version="1.0.0",
                description="Future",
                steps=(
                    create_workflow_step_definition(
                        step_id="one",
                        tool_id="future-external-tool",
                        position=0,
                    ),
                ),
            )
        )


def test_playbook_as_tool_nesting_rejected() -> None:
    tools = build_default_tool_registry()
    registry = WorkflowRegistry(tool_registry=tools)
    first = create_workflow_definition(
        name="outer-book",
        version="1.0.0",
        description="Outer",
        steps=(
            create_workflow_step_definition(
                step_id="one",
                tool_id="system-summary",
                position=0,
            ),
        ),
        tool_registry=tools,
    )
    registry.register(first)
    with pytest.raises(WorkflowValidationError):
        registry.register(
            create_workflow_definition(
                name="nested-book",
                version="1.0.0",
                description="Nested",
                steps=(
                    create_workflow_step_definition(
                        step_id="one",
                        tool_id="outer-book",
                        position=0,
                    ),
                ),
            )
        )


def test_builtin_steps_resolve_to_registered_tools() -> None:
    tools = build_default_tool_registry()
    workflows = build_default_workflow_registry(tool_registry=tools)
    for playbook in workflows.list_all():
        assert 2 <= len(playbook.steps) <= 4
        for step in playbook.steps:
            definition = tools.require(step.tool_id)
            assert definition.enabled
            assert definition.risk_level != "prohibited"
