"""Security invariant tests for the Milestone 9 tool execution path."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.tool_definition import FORBIDDEN_PARAMETER_NAMES, create_tool_definition
from src.tool_policy import assert_executable, assert_requestable
from src.tool_common import ToolPolicyError
from src.tool_registry import ToolRegistry, build_default_tool_registry


TOOL_EXECUTION_MODULES = [
    Path("src/tool_executor.py"),
    Path("src/tool_implementations.py"),
    Path("src/tool_safe_files.py"),
    Path("src/tool_commands.py"),
    Path("src/tool_policy.py"),
]


def _module_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_subprocess_shell_eval_exec_in_tool_path() -> None:
    for path in TOOL_EXECUTION_MODULES:
        source = _module_source(path)
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


def test_no_raw_command_field_in_parameter_schemas() -> None:
    registry = build_default_tool_registry()
    for definition in registry.list_all():
        for parameter in definition.parameter_schema:
            assert parameter.name not in FORBIDDEN_PARAMETER_NAMES
            assert "command" not in parameter.name


def test_prohibited_disabled_future_external_cannot_run() -> None:
    prohibited = create_tool_definition(
        tool_id="prohibited-tool",
        name="Prohibited",
        description="Cannot run",
        category="diagnostics",
        version="1.0.0",
        risk_level="prohibited",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=True,
        enabled=False,
        implementation_identifier="impl_prohibited_tool",
    )
    with pytest.raises(ToolPolicyError):
        assert_requestable(prohibited)
    with pytest.raises(ToolPolicyError):
        assert_executable(prohibited)

    disabled = create_tool_definition(
        tool_id="disabled-tool",
        name="Disabled",
        description="Disabled",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        enabled=False,
        implementation_identifier="impl_disabled_tool",
    )
    with pytest.raises(ToolPolicyError):
        assert_requestable(disabled)

    future = create_tool_definition(
        tool_id="future-external-tool",
        name="Future",
        description="Future",
        category="diagnostics",
        version="1.0.0",
        risk_level="low",
        execution_mode="future-external",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=True,
        enabled=False,
        implementation_identifier="impl_future_external_tool",
    )
    with pytest.raises(ToolPolicyError):
        assert_requestable(future)


def test_registry_does_not_load_plugins_dynamically() -> None:
    source = Path("src/tool_registry.py").read_text(encoding="utf-8")
    assert "importlib" not in source
    assert "entry_points" not in source
    assert "pkgutil" not in source
    registry = ToolRegistry()
    assert registry.count() == 0
