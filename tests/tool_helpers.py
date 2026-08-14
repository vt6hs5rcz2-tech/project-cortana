"""Shared helpers for Milestone 9 defensive tool framework tests."""

from pathlib import Path

from src.incident_repository import JsonIncidentRepository
from src.tool_definition import DefensiveToolDefinition, create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_repository import JsonToolControlRepository
from src.tool_scope import AuthorizedScope, create_authorized_scope


def tool_repository(tmp_path: Path) -> JsonToolControlRepository:
    """Return a temporary JSON tool-control repository."""
    return JsonToolControlRepository(tmp_path / "tool_control.json")


def incident_repository(tmp_path: Path) -> JsonIncidentRepository:
    """Return a temporary JSON incident repository."""
    return JsonIncidentRepository(tmp_path / "incidents.json")


def tool_registry() -> ToolRegistry:
    """Return the default built-in tool registry."""
    return build_default_tool_registry()


def tool_executor(
    incident_repo: JsonIncidentRepository | None = None,
) -> DefensiveToolExecutor:
    """Return a defensive tool executor wired to an optional incident repository."""
    return DefensiveToolExecutor(incident_repository=incident_repo)


def make_scope(
    tmp_path: Path,
    *,
    tool_ids: list[str],
    root: Path | None = None,
    name: str = "Test scope",
) -> AuthorizedScope:
    """Create a validated authorized scope for tests."""
    del tmp_path  # Reserved for future path fixtures; roots use explicit Path.
    roots = [str(root)] if root is not None else []
    target_types = ["none", "system-summary", "incident", "mock-log"]
    if root is not None:
        target_types.append("local-file")
    return create_authorized_scope(
        scope_name=name,
        allowed_tool_ids=tool_ids,
        allowed_target_types=target_types,
        allowed_local_path_roots=roots,
        notes="test-scope-notes-marker",
    )


def make_gated_test_tool(
    *,
    capability_class: str,
    tool_id: str = "future-side-effect",
    implementation_identifier: str = "impl_future_side_effect",
    requires_approval: bool = True,
    enabled: bool = True,
) -> DefensiveToolDefinition:
    """Return a test-only tool with an explicit gated or reserved capability class.

    Capability is never inferred from ``tool_id``.
    """
    return create_tool_definition(
        tool_id=tool_id,
        name="Future Side Effect",
        description=(
            "Test-only side-effecting tool used to prove capability kill-switches."
        ),
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=requires_approval,
        implementation_identifier=implementation_identifier,
        capability_class=capability_class,
        enabled=enabled,
    )
