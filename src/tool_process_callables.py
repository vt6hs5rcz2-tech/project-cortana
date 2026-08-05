"""Trusted process-safe tool callables for the child dispatch table.

Non-file callables require no repository objects, network access, or secrets.
Milestone 15 file-integrity callables use parent-authorized safe-open data only.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from typing import Any

from src.config import (
    ARBITRARY_SHELL_EXECUTION_ENABLED,
    AUTONOMOUS_REMEDIATION_ENABLED,
    DEFENSIVE_TOOL_FRAMEWORK_ENABLED,
    EXTERNAL_TOOL_EXECUTION_ENABLED,
    TOOL_DRY_RUN_ENFORCEMENT_ENABLED,
    TOOL_HUMAN_APPROVAL_ENABLED,
    TOOL_SCOPE_ENFORCEMENT_ENABLED,
    VERSION,
)
from src.tool_common import ToolValidationError
from src.tool_process_file_tools import run_compare_sha256, run_file_sha256


class ProcessSafeToolError(ToolValidationError):
    """Raised when a process-safe callable fails in a controlled way."""


def run_system_summary(
    _parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return safe local platform information without host/user secrets."""
    return {
        "os_family": platform.system() or "unknown",
        "python_version": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "architecture": platform.machine() or "unknown",
        "cortana_version": VERSION,
        "capability_flags": {
            "defensive_tool_framework": DEFENSIVE_TOOL_FRAMEWORK_ENABLED,
            "scope_enforcement": TOOL_SCOPE_ENFORCEMENT_ENABLED,
            "human_approval": TOOL_HUMAN_APPROVAL_ENABLED,
            "dry_run_enforcement": TOOL_DRY_RUN_ENFORCEMENT_ENABLED,
            "arbitrary_shell_execution": ARBITRARY_SHELL_EXECUTION_ENABLED,
            "external_tool_execution": EXTERNAL_TOOL_EXECUTION_ENABLED,
            "autonomous_remediation": AUTONOMOUS_REMEDIATION_ENABLED,
        },
    }


def run_simulated_log_check(
    parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the simulation-only mock log check against controlled fixtures."""
    fixture = str(parameters["fixture"])
    needle = str(parameters.get("needle") or "alert")
    mock_logs = {
        "auth-noise": (
            "SIMULATION: login ok\n"
            "SIMULATION: failed password attempt\n"
            "SIMULATION: session closed\n"
        ),
        "malware-keyword": (
            "SIMULATION: scanner idle\n"
            "SIMULATION: keyword alert detected\n"
            "SIMULATION: no host claims made\n"
        ),
        "empty": "",
    }
    content = mock_logs.get(fixture)
    if content is None:
        raise ProcessSafeToolError("Unknown simulation fixture.")

    match_count = content.count(needle) if needle else 0
    return {
        "simulation": True,
        "label": "SIMULATION ONLY — no claims about the host system",
        "fixture": fixture,
        "match_count": match_count,
        "lines_scanned": 0 if not content else len(content.splitlines()),
    }


PROCESS_SAFE_CALLABLES: dict[str, Any] = {
    "impl_system_summary": run_system_summary,
    "impl_simulated_log_check": run_simulated_log_check,
    "impl_file_sha256": run_file_sha256,
    "impl_compare_sha256": run_compare_sha256,
}
