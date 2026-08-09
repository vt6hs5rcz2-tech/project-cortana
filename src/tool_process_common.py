"""Shared constants and helpers for Milestone 13 process-isolated tool execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from src.config import (
    PROCESS_IPC_SCHEMA_VERSION,
    PROJECT_ROOT,
    get_default_tool_process_scratch_dir_path,
)
from src.tool_common import ToolValidationError, validate_implementation_id

# Exact set of implementation identifiers approved for the child dispatch table.
# Must match registry process_isolation eligible/required tools in both directions.
PROCESS_SAFE_FILE_IMPLEMENTATION_IDS: frozenset[str] = frozenset(
    {
        "impl_file_sha256",
        "impl_compare_sha256",
        "impl_text_search",
    }
)

PROCESS_SAFE_FILE_TOOL_IDS: frozenset[str] = frozenset(
    {
        "file-sha256",
        "compare-sha256",
        "text-search",
    }
)

PROCESS_SAFE_IMPLEMENTATION_IDS: frozenset[str] = frozenset(
    {
        "impl_system_summary",
        "impl_simulated_log_check",
        *PROCESS_SAFE_FILE_IMPLEMENTATION_IDS,
    }
)

PROCESS_SAFE_TOOL_IDS: frozenset[str] = frozenset(
    {
        "system-summary",
        "simulated-log-check",
        *PROCESS_SAFE_FILE_TOOL_IDS,
    }
)

CHILD_ALLOWED_OUTCOMES: frozenset[str] = frozenset({"succeeded", "failed"})
PARENT_ONLY_OUTCOMES: frozenset[str] = frozenset(
    {
        "denied",
        "timed_out_terminated",
        "termination_unconfirmed",
        "resource_limit_exceeded",
        "cancelled",
        "planned",
        "expired",
    }
)

# Environment keys that may be copied into the child allowlist when present.
_WINDOWS_ENV_ALLOWLIST: tuple[str, ...] = (
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "PATHEXT",
    "COMSPEC",
)
_POSIX_ENV_ALLOWLIST: tuple[str, ...] = ("TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL")


class ToolProcessError(ToolValidationError):
    """Raised for process-isolation validation or IPC failures."""


class ToolProcessResultRejectedError(ToolProcessError):
    """Raised when a child result fails schema or correlation checks."""


def ensure_tool_process_scratch_dir(
    scratch_dir: Path | None = None,
) -> Path:
    """Create and return the dedicated process-isolation scratch directory."""
    path = scratch_dir or get_default_tool_process_scratch_dir_path()
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def build_child_environment() -> dict[str, str]:
    """Build an explicit child environment allowlist.

    Never inherits the full parent environment. Secrets, API keys, repository
    paths, and application configuration must not appear here.
    """
    env: dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Required so ``python -m src.tool_process_runner`` can import the package
        # when the child working directory is the dedicated scratch directory.
        "PYTHONPATH": str(PROJECT_ROOT.resolve()),
    }

    if os.name == "nt":
        for key in _WINDOWS_ENV_ALLOWLIST:
            value = os.environ.get(key, "").strip()
            if value:
                env[key] = value
        system_root = env.get("SYSTEMROOT") or os.environ.get("SYSTEMROOT", "").strip()
        if not system_root:
            system_root = r"C:\Windows"
            env["SYSTEMROOT"] = system_root
        # Minimal PATH so the interpreter can resolve essential system DLLs/tools.
        env["PATH"] = str(Path(system_root) / "System32")
        if "TEMP" not in env:
            env["TEMP"] = str(Path(system_root) / "Temp")
        if "TMP" not in env:
            env["TMP"] = env["TEMP"]
    else:
        for key in _POSIX_ENV_ALLOWLIST:
            value = os.environ.get(key, "").strip()
            if value:
                env[key] = value
        env["PATH"] = "/usr/bin:/bin"
        if "TMPDIR" not in env:
            env["TMPDIR"] = "/tmp"

    return env


def assert_implementation_process_safe(implementation_identifier: str) -> str:
    """Validate and confirm an implementation ID is in the process-safe set."""
    cleaned = validate_implementation_id(implementation_identifier)
    if cleaned not in PROCESS_SAFE_IMPLEMENTATION_IDS:
        raise ToolProcessError(
            "Implementation identifier is not approved for process isolation."
        )
    return cleaned


def registry_process_isolation_implementation_ids(
    definitions: list[Any],
) -> frozenset[str]:
    """Return implementation IDs marked eligible or required for process isolation."""
    selected: set[str] = set()
    for definition in definitions:
        isolation = getattr(definition, "process_isolation", "prohibited")
        if isolation in {"eligible", "required"}:
            selected.add(str(definition.implementation_identifier))
    return frozenset(selected)


def assert_process_isolation_tables_match(
    *,
    registry_implementation_ids: frozenset[str],
    child_dispatch_ids: frozenset[str],
) -> None:
    """Fail closed when registry eligibility and child dispatch disagree."""
    if registry_implementation_ids != child_dispatch_ids:
        raise ToolProcessError(
            "Process-isolation registry eligibility and child dispatch "
            "table do not match exactly."
        )
    if registry_implementation_ids != PROCESS_SAFE_IMPLEMENTATION_IDS:
        raise ToolProcessError(
            "Process-isolation eligible set drifted from the approved "
            "PROCESS_SAFE_IMPLEMENTATION_IDS constant."
        )


def process_ipc_schema_version() -> int:
    """Return the supported process IPC schema version."""
    return PROCESS_IPC_SCHEMA_VERSION


def python_executable() -> str:
    """Return the absolute path of the Python interpreter used for child processes.

    On Windows, ``.venv\\Scripts\\python.exe`` is a launcher stub that creates a
    second OS process for the base interpreter. Cortana's Job Object enforces
    ``ActiveProcessLimit = 1``, so that second CreateProcess is rejected with
    ``ERROR_NOT_ENOUGH_QUOTA``. When running inside a Windows virtual
    environment, return the trusted base interpreter derived from
    ``sys.base_prefix`` so the child is a single process. Elsewhere, return
    ``sys.executable``.

    The path is never taken from user input or arbitrary environment overrides.
    """
    if os.name == "nt" and sys.prefix != sys.base_prefix:
        candidate = Path(sys.base_prefix) / "python.exe"
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ToolProcessError(
                "Trusted base Python interpreter is unavailable for "
                "process-isolated execution."
            ) from error
        if not resolved.is_file():
            raise ToolProcessError(
                "Trusted base Python interpreter is unavailable for "
                "process-isolated execution."
            )
        return str(resolved)
    return sys.executable
