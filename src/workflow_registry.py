"""Trusted in-code registry for defensive workflow/playbook definitions."""

from __future__ import annotations

from src.config import (
    MAX_WORKFLOW_STEPS,
    WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED,
    WORKFLOW_NESTED_PLAYBOOKS_ENABLED,
)
from src.tool_definition import validate_parameters_against_schema
from src.tool_policy import assert_requestable
from src.tool_registry import ToolRegistry
from src.workflow_common import WorkflowValidationError, validate_playbook_name
from src.workflow_definition import (
    WorkflowDefinition,
    WorkflowStepDefinition,
    validate_workflow_definition,
)


class WorkflowRegistry:
    """Typed registry of trusted WorkflowDefinition playbooks."""

    def __init__(self, *, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._by_key: dict[tuple[str, str], WorkflowDefinition] = {}
        self._order: list[tuple[str, str]] = []
        self._names: set[str] = set()

    def register(self, definition: WorkflowDefinition) -> None:
        """Register one immutable playbook from trusted application code."""
        if WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED:
            raise WorkflowValidationError(
                "External playbook loading is disabled for this milestone."
            )
        if not isinstance(definition, WorkflowDefinition):
            raise WorkflowValidationError(
                "Only WorkflowDefinition objects may be registered."
            )

        validated = validate_workflow_definition(definition)
        key = (validated.name, validated.version)
        if key in self._by_key:
            raise WorkflowValidationError(
                f"Duplicate playbook '{validated.name}' version "
                f"'{validated.version}'."
            )

        if len(validated.steps) > MAX_WORKFLOW_STEPS:
            raise WorkflowValidationError(
                f"Workflow exceeds the maximum of {MAX_WORKFLOW_STEPS} steps."
            )

        normalized_steps: list[WorkflowStepDefinition] = []
        for step in validated.steps:
            if step.tool_id in self._names or step.tool_id == validated.name:
                if not WORKFLOW_NESTED_PLAYBOOKS_ENABLED:
                    raise WorkflowValidationError(
                        "Workflow steps cannot reference playbooks."
                    )
            tool_definition = self._tool_registry.get(step.tool_id)
            if tool_definition is None:
                # Reject playbook names used as tool references.
                if self.get(step.tool_id) is not None:
                    raise WorkflowValidationError(
                        "Workflow steps cannot reference playbooks."
                    )
                raise WorkflowValidationError(
                    f"Unknown tool ID '{step.tool_id}'."
                )
            if not tool_definition.enabled:
                raise WorkflowValidationError(
                    f"Disabled tool '{step.tool_id}' cannot be used in a playbook."
                )
            if tool_definition.risk_level == "prohibited":
                raise WorkflowValidationError(
                    f"Prohibited tool '{step.tool_id}' cannot be used in a playbook."
                )
            assert_requestable(tool_definition)
            # Defensive-only: reject non-allowlisted execution modes already
            # covered by assert_requestable / assert later; keep explicit check.
            if tool_definition.execution_mode not in {"internal-python", "simulated"}:
                raise WorkflowValidationError(
                    f"Non-defensive tool '{step.tool_id}' cannot be used."
                )

            normalized_parameters = validate_parameters_against_schema(
                step.static_parameters,
                tool_definition.parameter_schema,
            )
            normalized_steps.append(
                WorkflowStepDefinition(
                    step_id=step.step_id,
                    tool_id=step.tool_id,
                    static_parameters=normalized_parameters,
                    description=step.description,
                    position=step.position,
                )
            )

        stored = WorkflowDefinition(
            name=validated.name,
            version=validated.version,
            description=validated.description,
            steps=tuple(normalized_steps),
            enabled=validated.enabled,
        )
        self._by_key[key] = stored
        self._order.append(key)
        self._names.add(stored.name)

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> WorkflowDefinition | None:
        """Return one playbook by name, optionally pinned to a version."""
        cleaned = validate_playbook_name(name)
        if version is not None and version.strip():
            return self._by_key.get((cleaned, version.strip()))
        matches = [
            self._by_key[key]
            for key in self._order
            if key[0] == cleaned
        ]
        if not matches:
            return None
        return matches[-1]

    def require(
        self,
        name: str,
        version: str | None = None,
    ) -> WorkflowDefinition:
        """Return one playbook or raise when missing."""
        definition = self.get(name, version=version)
        if definition is None:
            if version is not None and version.strip():
                raise WorkflowValidationError(
                    f"Unknown playbook '{name}' version '{version.strip()}'."
                )
            raise WorkflowValidationError(f"Unknown playbook '{name}'.")
        return definition

    def list_all(self) -> list[WorkflowDefinition]:
        """Return all registered playbooks in deterministic registration order."""
        return [self._by_key[key] for key in self._order]

    def list_enabled(self) -> list[WorkflowDefinition]:
        """Return enabled playbooks in deterministic order."""
        return [
            definition
            for definition in self.list_all()
            if definition.enabled
        ]

    def count(self) -> int:
        """Return the number of registered playbooks."""
        return len(self._order)

    def enabled_count(self) -> int:
        """Return the number of enabled playbooks."""
        return len(self.list_enabled())

    def playbook_names(self) -> frozenset[str]:
        """Return registered playbook names."""
        return frozenset(self._names)
