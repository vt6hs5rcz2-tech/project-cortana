"""Trusted built-in defensive playbooks composed from Milestone 9 tools."""

from __future__ import annotations

from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.workflow_definition import (
    WorkflowDefinition,
    create_workflow_definition,
    create_workflow_step_definition,
)
from src.workflow_registry import WorkflowRegistry


def builtin_workflow_definitions(
    *,
    tool_registry: ToolRegistry | None = None,
) -> list[WorkflowDefinition]:
    """Return the immutable built-in playbook definitions.

    Playbooks use only registered Milestone 9 tools with static parameters.
    No dynamic step-output binding is used.
    """
    registry = tool_registry or build_default_tool_registry()
    return [
        create_workflow_definition(
            name="platform-baseline",
            version="1.0.0",
            description=(
                "Collect a safe local platform summary and run one controlled "
                "mock authentication-noise log check."
            ),
            steps=(
                create_workflow_step_definition(
                    step_id="collect-system-summary",
                    tool_id="system-summary",
                    static_parameters={},
                    description="Capture safe local platform information.",
                    position=0,
                ),
                create_workflow_step_definition(
                    step_id="check-auth-noise",
                    tool_id="simulated-log-check",
                    static_parameters={"fixture": "auth-noise"},
                    description="Run the auth-noise mock log simulation.",
                    position=1,
                ),
            ),
            enabled=True,
            tool_registry=registry,
        ),
        create_workflow_definition(
            name="mock-log-triage",
            version="1.0.0",
            description=(
                "Run a bounded mock-log triage sequence across controlled "
                "fixtures without touching real files or network resources."
            ),
            steps=(
                create_workflow_step_definition(
                    step_id="scan-auth-noise",
                    tool_id="simulated-log-check",
                    static_parameters={"fixture": "auth-noise"},
                    description="Inspect the auth-noise mock fixture.",
                    position=0,
                ),
                create_workflow_step_definition(
                    step_id="scan-malware-keyword",
                    tool_id="simulated-log-check",
                    static_parameters={"fixture": "malware-keyword"},
                    description="Inspect the malware-keyword mock fixture.",
                    position=1,
                ),
                create_workflow_step_definition(
                    step_id="scan-empty-fixture",
                    tool_id="simulated-log-check",
                    static_parameters={"fixture": "empty"},
                    description="Inspect the empty mock fixture.",
                    position=2,
                ),
            ),
            enabled=True,
            tool_registry=registry,
        ),
    ]


def build_default_workflow_registry(
    *,
    tool_registry: ToolRegistry | None = None,
) -> WorkflowRegistry:
    """Create the trusted Milestone 10 built-in workflow registry."""
    tools = tool_registry or build_default_tool_registry()
    workflow_registry = WorkflowRegistry(tool_registry=tools)
    for definition in builtin_workflow_definitions(tool_registry=tools):
        workflow_registry.register(definition)
    return workflow_registry
