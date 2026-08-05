"""Shared helpers for Milestone 10 workflow orchestration tests."""

from __future__ import annotations

from pathlib import Path

from src.config import MAX_WORKFLOW_RUNTIME_SECONDS
from src.incident_repository import JsonIncidentRepository
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_repository import JsonToolControlRepository
from src.tool_scope import AuthorizedScope, create_authorized_scope
from src.workflow_builtins import build_default_workflow_registry
from src.workflow_common import WorkflowClock
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import InMemoryWorkflowRunRepository


def tool_repository(tmp_path: Path) -> JsonToolControlRepository:
    return JsonToolControlRepository(tmp_path / "tool_control.json")


def incident_repository(tmp_path: Path) -> JsonIncidentRepository:
    return JsonIncidentRepository(tmp_path / "incidents.json")


def tool_registry() -> ToolRegistry:
    return build_default_tool_registry()


def workflow_registry(tools: ToolRegistry | None = None) -> WorkflowRegistry:
    return build_default_workflow_registry(tool_registry=tools or tool_registry())


def make_scope(
    *,
    tool_ids: list[str],
    root: Path | None = None,
    name: str = "Workflow test scope",
) -> AuthorizedScope:
    roots = [str(root)] if root is not None else []
    target_types = ["none", "system-summary", "incident", "mock-log"]
    if root is not None:
        target_types.append("local-file")
    return create_authorized_scope(
        scope_name=name,
        allowed_tool_ids=tool_ids,
        allowed_target_types=target_types,
        allowed_local_path_roots=roots,
        notes="workflow-test-scope",
    )


def make_executor(
    tmp_path: Path,
    *,
    tools: ToolRegistry | None = None,
    workflows: WorkflowRegistry | None = None,
    clock: WorkflowClock | None = None,
    max_runtime_seconds: int = MAX_WORKFLOW_RUNTIME_SECONDS,
) -> tuple[
    WorkflowExecutor,
    JsonToolControlRepository,
    ToolRegistry,
    WorkflowRegistry,
    InMemoryWorkflowRunRepository,
]:
    resolved_tools = tools or tool_registry()
    repo = tool_repository(tmp_path)
    incidents = incident_repository(tmp_path)
    tool_exec = DefensiveToolExecutor(incident_repository=incidents)
    resolved_workflows = workflows or workflow_registry(resolved_tools)
    runs = InMemoryWorkflowRunRepository()
    executor = WorkflowExecutor(
        workflow_registry=resolved_workflows,
        tool_registry=resolved_tools,
        tool_executor=tool_exec,
        tool_repository=repo,
        run_repository=runs,
        clock=clock,
        max_runtime_seconds=max_runtime_seconds,
    )
    return executor, repo, resolved_tools, resolved_workflows, runs
