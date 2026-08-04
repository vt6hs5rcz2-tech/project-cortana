"""Shared helpers for Milestone 9 defensive tool framework tests."""

from pathlib import Path

from src.incident_repository import JsonIncidentRepository
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
