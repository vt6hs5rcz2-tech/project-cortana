"""Security invariant tests for Milestone 10 workflow orchestration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config import (
    WORKFLOW_AI_CONTEXT_INJECTION_ENABLED,
    WORKFLOW_BACKGROUND_EXECUTION_ENABLED,
    WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED,
    WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED,
    WORKFLOW_NESTED_PLAYBOOKS_ENABLED,
    WORKFLOW_PARALLEL_EXECUTION_ENABLED,
)
from src.workflow_builtins import build_default_workflow_registry
from src.tool_registry import build_default_tool_registry


WORKFLOW_MODULES = [
    Path("src/workflow_common.py"),
    Path("src/workflow_definition.py"),
    Path("src/workflow_request.py"),
    Path("src/workflow_result.py"),
    Path("src/workflow_audit.py"),
    Path("src/workflow_registry.py"),
    Path("src/workflow_builtins.py"),
    Path("src/workflow_repository.py"),
    Path("src/workflow_executor.py"),
    Path("src/workflow_commands.py"),
]


def test_no_subprocess_shell_eval_exec_in_workflow_modules() -> None:
    for path in WORKFLOW_MODULES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] != "subprocess"
            if isinstance(node, ast.ImportFrom):
                assert node.module != "subprocess"
                if node.module is not None:
                    assert not node.module.startswith("subprocess.")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "eval",
                    "exec",
                    "__import__",
                }:
                    pytest.fail(f"{path} contains {node.func.id}()")
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "system" and isinstance(
                        node.func.value, ast.Name
                    ) and node.func.value.id == "os":
                        pytest.fail(f"{path} contains os.system()")
                    if node.func.attr in {"Popen", "run", "call", "check_output"}:
                        if isinstance(node.func.value, ast.Name) and (
                            node.func.value.id == "subprocess"
                        ):
                            pytest.fail(f"{path} contains subprocess usage")
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(
                        keyword.value, ast.Constant
                    ) and keyword.value.value is True:
                        pytest.fail(f"{path} contains shell=True")


def test_executor_only_uses_defensive_tool_executor_boundary() -> None:
    source = Path("src/workflow_executor.py").read_text(encoding="utf-8")
    assert "DefensiveToolExecutor" in source
    assert ".plan_dry_run(" in source
    assert ".execute(" in source
    assert "build_implementation_dispatch" not in source
    assert "impl_" not in source or "implementation" in source.lower()
    # Workflow executor must not import tool implementations module.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.tool_implementations":
            pytest.fail("workflow_executor imports tool implementations")


def test_no_external_playbook_loading_or_dynamic_binding_flags() -> None:
    assert WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED is False
    assert WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED is False
    assert WORKFLOW_PARALLEL_EXECUTION_ENABLED is False
    assert WORKFLOW_BACKGROUND_EXECUTION_ENABLED is False
    assert WORKFLOW_NESTED_PLAYBOOKS_ENABLED is False
    assert WORKFLOW_AI_CONTEXT_INJECTION_ENABLED is False


def test_builtin_playbooks_reference_only_registered_tools() -> None:
    tools = build_default_tool_registry()
    workflows = build_default_workflow_registry(tool_registry=tools)
    assert workflows.count() >= 2
    for playbook in workflows.list_all():
        assert playbook.enabled
        for step in playbook.steps:
            tools.require(step.tool_id)


def test_no_json_yaml_playbook_loaders() -> None:
    for path in WORKFLOW_MODULES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in {"yaml", "toml"}
            if isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert node.module.split(".", 1)[0] not in {"yaml", "toml"}
        assert "load_playbook" not in source
        assert "read_playbook" not in source
