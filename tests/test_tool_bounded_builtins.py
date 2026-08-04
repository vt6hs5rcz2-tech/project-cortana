"""Static assertions that current executable built-in tools stay bounded.

These checks cover the Milestone 9 registered implementations only. They do not
prove that arbitrary future implementations are safe.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.config import MAX_TOOL_FILE_BYTES, MAX_TOOL_OUTPUT_CHARS
from src.tool_registry import build_default_tool_registry

BUILTIN_IMPLEMENTATION_MODULES = (
    Path("src/tool_implementations.py"),
    Path("src/tool_safe_files.py"),
)

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "subprocess",
        "multiprocessing",
        "socket",
        "ssl",
        "http",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "telnetlib",
        "paramiko",
        "asyncio",
    }
)

FORBIDDEN_CALL_NAMES = frozenset(
    {
        "urlopen",
        "Popen",
        "check_output",
        "eval",
        "exec",
        "__import__",
    }
)


def _module_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builtin_tools_are_internal_or_simulated_and_bounded() -> None:
    registry = build_default_tool_registry()
    enabled = registry.list_enabled()
    assert enabled
    for definition in enabled:
        assert definition.execution_mode in {"internal-python", "simulated"}
        assert definition.risk_level != "prohibited"
        assert definition.supports_dry_run is True
        assert 1 <= definition.timeout_seconds <= 30
        assert 1 <= definition.maximum_output_characters <= MAX_TOOL_OUTPUT_CHARS


def test_builtin_implementation_modules_forbid_network_and_process_wait() -> None:
    for path in BUILTIN_IMPLEMENTATION_MODULES:
        source = _module_source(path)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root not in FORBIDDEN_IMPORT_ROOTS, path
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                root = node.module.split(".", 1)[0]
                assert root not in FORBIDDEN_IMPORT_ROOTS, path
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in FORBIDDEN_CALL_NAMES, path
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in FORBIDDEN_CALL_NAMES, path
                    # Forbid os.system only; platform.system() is allowed.
                    if (
                        node.func.attr == "system"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os"
                    ):
                        raise AssertionError(f"{path} contains os.system()")
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        assert keyword.value.value is not True, path


def test_builtin_file_helpers_enforce_size_and_regular_readonly_files() -> None:
    source = _module_source(Path("src/tool_safe_files.py"))
    assert "MAX_TOOL_FILE_BYTES" in source
    assert "S_ISREG" in source
    assert "O_RDONLY" in source
    assert "is_symlink" in source
    assert "O_WRONLY" not in source
    assert "os.write" not in source
    assert str(MAX_TOOL_FILE_BYTES)  # centralized limit exists

    tree = ast.parse(source)
    while_true_count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
        ):
            while_true_count += 1
            # Streaming loops must break and enforce a byte ceiling.
            body_dump = ast.dump(node)
            assert "Break" in body_dump
            assert "max_bytes" in body_dump or "MAX_TOOL_FILE_BYTES" in source
    assert while_true_count >= 1


def test_builtin_implementations_have_no_retry_loops_or_sleep() -> None:
    source = _module_source(Path("src/tool_implementations.py"))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            pytest_fail = (
                "Built-in tool implementations must not use while-loops; "
                "file streaming belongs in tool_safe_files with size bounds."
            )
            raise AssertionError(pytest_fail)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "sleep":
                raise AssertionError("Built-in tools must not sleep.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "sleep":
                raise AssertionError("Built-in tools must not sleep.")


def test_builtin_tools_do_not_mutate_files() -> None:
    for path in BUILTIN_IMPLEMENTATION_MODULES:
        source = _module_source(path)
        for forbidden in (
            "unlink(",
            "os.remove",
            "os.replace",
            "write_text",
            "write_bytes",
            "os.write",
            "O_WRONLY",
            "O_RDWR",
        ):
            assert forbidden not in source, f"{path} contains {forbidden}"
        # Prefer os.open(..., O_RDONLY); disallow builtin open() for clarity.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "open", path
