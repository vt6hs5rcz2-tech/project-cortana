"""Immutable trusted workflow/playbook definitions for Milestone 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.config import (
    MAX_WORKFLOW_DESCRIPTION_LENGTH,
    MAX_WORKFLOW_NAME_LENGTH,
    MAX_WORKFLOW_STEP_DESCRIPTION_LENGTH,
    MAX_WORKFLOW_STEPS,
)
from src.tool_common import (
    json_safe_value,
    require_non_blank_text,
    validate_optional_text,
    validate_tool_id,
)
from src.tool_definition import validate_parameters_against_schema
from src.tool_registry import ToolRegistry
from src.workflow_common import (
    WorkflowValidationError,
    validate_playbook_name,
    validate_playbook_version,
    validate_step_id,
)


@dataclass(frozen=True)
class WorkflowStepDefinition:
    """One ordered, static, trusted workflow step referencing a registered tool."""

    step_id: str
    tool_id: str
    static_parameters: dict[str, Any]
    description: str | None
    position: int


@dataclass(frozen=True)
class WorkflowDefinition:
    """Trusted playbook definition composed of ordered Milestone 9 tool steps."""

    name: str
    version: str
    description: str
    steps: tuple[WorkflowStepDefinition, ...]
    enabled: bool


def create_workflow_step_definition(
    *,
    step_id: str,
    tool_id: str,
    static_parameters: Mapping[str, Any] | None = None,
    description: str | None = None,
    position: int,
) -> WorkflowStepDefinition:
    """Create one validated immutable workflow step definition."""
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise WorkflowValidationError("Step position must be a non-negative integer.")

    cleaned_step_id = validate_step_id(step_id)
    cleaned_tool_id = validate_tool_id(tool_id)
    if cleaned_step_id == cleaned_tool_id:
        # Allowed in principle, but keep IDs distinct for clearer audit trails.
        pass

    safe_parameters = json_safe_value(dict(static_parameters or {}))
    if not isinstance(safe_parameters, dict):
        raise WorkflowValidationError("Static parameters must be an object.")

    cleaned_description = validate_optional_text(
        description,
        field_name="Step description",
        max_length=MAX_WORKFLOW_STEP_DESCRIPTION_LENGTH,
    )
    return WorkflowStepDefinition(
        step_id=cleaned_step_id,
        tool_id=cleaned_tool_id,
        static_parameters=safe_parameters,
        description=cleaned_description,
        position=position,
    )


def create_workflow_definition(
    *,
    name: str,
    version: str,
    description: str,
    steps: list[WorkflowStepDefinition] | tuple[WorkflowStepDefinition, ...],
    enabled: bool = True,
    tool_registry: ToolRegistry | None = None,
) -> WorkflowDefinition:
    """Create one validated immutable workflow definition.

    When ``tool_registry`` is provided, static parameters are validated against
    each referenced tool schema during construction.
    """
    cleaned_name = validate_playbook_name(name)
    cleaned_version = validate_playbook_version(version)
    cleaned_description = require_non_blank_text(
        description,
        field_name="Playbook description",
        max_length=MAX_WORKFLOW_DESCRIPTION_LENGTH,
    )
    if not isinstance(steps, (list, tuple)):
        raise WorkflowValidationError("Workflow steps must be a list or tuple.")
    if not steps:
        raise WorkflowValidationError("Workflow must contain at least one step.")
    if len(steps) > MAX_WORKFLOW_STEPS:
        raise WorkflowValidationError(
            f"Workflow exceeds the maximum of {MAX_WORKFLOW_STEPS} steps."
        )

    normalized_steps: list[WorkflowStepDefinition] = []
    seen_step_ids: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, WorkflowStepDefinition):
            raise WorkflowValidationError(
                "Only WorkflowStepDefinition objects may be used as steps."
            )
        if step.position != index:
            raise WorkflowValidationError(
                "Workflow steps must be ordered with contiguous positions "
                "starting at 0."
            )
        if step.step_id in seen_step_ids:
            raise WorkflowValidationError(
                f"Duplicate workflow step ID '{step.step_id}'."
            )
        seen_step_ids.add(step.step_id)

        if tool_registry is not None:
            definition = tool_registry.require(step.tool_id)
            normalized_parameters = validate_parameters_against_schema(
                step.static_parameters,
                definition.parameter_schema,
            )
            step = WorkflowStepDefinition(
                step_id=step.step_id,
                tool_id=step.tool_id,
                static_parameters=normalized_parameters,
                description=step.description,
                position=step.position,
            )
        normalized_steps.append(step)

    return validate_workflow_definition(
        WorkflowDefinition(
            name=cleaned_name,
            version=cleaned_version,
            description=cleaned_description,
            steps=tuple(normalized_steps),
            enabled=bool(enabled),
        )
    )


def validate_workflow_definition(
    definition: WorkflowDefinition,
) -> WorkflowDefinition:
    """Validate an existing workflow definition."""
    cleaned_name = validate_playbook_name(definition.name)
    cleaned_version = validate_playbook_version(definition.version)
    cleaned_description = require_non_blank_text(
        definition.description,
        field_name="Playbook description",
        max_length=MAX_WORKFLOW_DESCRIPTION_LENGTH,
    )
    if not definition.steps:
        raise WorkflowValidationError("Workflow must contain at least one step.")
    if len(definition.steps) > MAX_WORKFLOW_STEPS:
        raise WorkflowValidationError(
            f"Workflow exceeds the maximum of {MAX_WORKFLOW_STEPS} steps."
        )

    seen: set[str] = set()
    validated_steps: list[WorkflowStepDefinition] = []
    for index, step in enumerate(definition.steps):
        if step.position != index:
            raise WorkflowValidationError(
                "Workflow steps must be ordered with contiguous positions "
                "starting at 0."
            )
        if step.step_id in seen:
            raise WorkflowValidationError(
                f"Duplicate workflow step ID '{step.step_id}'."
            )
        seen.add(step.step_id)
        safe_parameters = json_safe_value(step.static_parameters)
        if not isinstance(safe_parameters, dict):
            raise WorkflowValidationError("Static parameters must be an object.")
        validated_steps.append(
            WorkflowStepDefinition(
                step_id=validate_step_id(step.step_id),
                tool_id=validate_tool_id(step.tool_id),
                static_parameters=safe_parameters,
                description=validate_optional_text(
                    step.description,
                    field_name="Step description",
                    max_length=MAX_WORKFLOW_STEP_DESCRIPTION_LENGTH,
                ),
                position=step.position,
            )
        )

    # Keep playbook display names bounded similarly to tool names.
    _ = require_non_blank_text(
        cleaned_name,
        field_name="Playbook name",
        max_length=MAX_WORKFLOW_NAME_LENGTH,
    )
    return WorkflowDefinition(
        name=cleaned_name,
        version=cleaned_version,
        description=cleaned_description,
        steps=tuple(validated_steps),
        enabled=bool(definition.enabled),
    )
