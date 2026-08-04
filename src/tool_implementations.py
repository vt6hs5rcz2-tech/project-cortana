"""Trusted internal Python callables for built-in defensive tools."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Mapping
from typing import Any

from src.config import (
    ARBITRARY_SHELL_EXECUTION_ENABLED,
    AUTONOMOUS_REMEDIATION_ENABLED,
    DEFENSIVE_TOOL_FRAMEWORK_ENABLED,
    EXTERNAL_TOOL_EXECUTION_ENABLED,
    MAX_TOOL_TEXT_SEARCH_MATCHES,
    MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS,
    TOOL_DRY_RUN_ENFORCEMENT_ENABLED,
    TOOL_HUMAN_APPROVAL_ENABLED,
    TOOL_SCOPE_ENFORCEMENT_ENABLED,
    VERSION,
)
from src.incident_repository import IncidentRepository
from src.tool_common import ToolValidationError, validate_sha256_digest
from src.tool_safe_files import SafeFileError, filename_only, read_text_lines, stream_sha256

ToolCallable = Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]


class ToolImplementationError(ToolValidationError):
    """Raised when a trusted tool implementation fails in a controlled way."""


def build_implementation_dispatch(
    *,
    incident_repository: IncidentRepository | None = None,
) -> dict[str, ToolCallable]:
    """Return the trusted implementation-identifier to callable mapping."""

    def system_summary(
        _parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
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

    def file_sha256(
        parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        path_value = str(parameters["path"])
        digest, size_bytes, name = stream_sha256(path_value)
        return {
            "sha256": digest,
            "size_bytes": size_bytes,
            "filename": name,
        }

    def text_search(
        parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        path_value = str(parameters["path"])
        query = str(parameters["query"])
        max_matches = int(parameters.get("max_matches", MAX_TOOL_TEXT_SEARCH_MATCHES))
        max_matches = min(max(max_matches, 1), MAX_TOOL_TEXT_SEARCH_MATCHES)

        try:
            lines, name = read_text_lines(path_value)
        except SafeFileError:
            raise

        matches: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            if query not in line:
                continue
            preview = line.strip()
            if len(preview) > MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS:
                preview = preview[:MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS] + "..."
            matches.append(
                {
                    "line_number": index,
                    "preview": preview,
                }
            )
            if len(matches) >= max_matches:
                break

        return {
            "filename": name,
            "match_count": len(matches),
            "truncated": len(matches) >= max_matches,
            "matches": matches,
        }

    def compare_sha256(
        parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        path_value = str(parameters["path"])
        expected = validate_sha256_digest(str(parameters["expected_sha256"]))
        digest, size_bytes, name = stream_sha256(path_value)
        return {
            "filename": name,
            "size_bytes": size_bytes,
            "calculated_sha256": digest,
            "expected_sha256": expected,
            "comparison": "match" if digest == expected else "mismatch",
        }

    def incident_summary(
        parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if incident_repository is None:
            raise ToolImplementationError(
                "Incident repository is unavailable for incident-summary."
            )
        incident_id = str(parameters["incident_id"])
        incident = incident_repository.get_incident(incident_id)
        if incident is None:
            raise ToolImplementationError("Incident was not found.")

        timeline = incident_repository.build_timeline(incident_id)
        return {
            "incident_id": incident.incident_id,
            "severity": incident.severity,
            "status": incident.status,
            "linked_event_count": len(incident.event_ids),
            "linked_evidence_count": len(incident.evidence_ids),
            "linked_indicator_count": len(incident.indicator_ids),
            "linked_note_count": len(incident.note_ids),
            "timeline_entry_count": len(timeline),
        }

    def simulated_log_check(
        parameters: Mapping[str, Any],
        _context: Mapping[str, Any],
    ) -> dict[str, Any]:
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
            raise ToolImplementationError("Unknown simulation fixture.")

        match_count = content.count(needle) if needle else 0
        return {
            "simulation": True,
            "label": "SIMULATION ONLY — no claims about the host system",
            "fixture": fixture,
            "match_count": match_count,
            "lines_scanned": 0 if not content else len(content.splitlines()),
        }

    return {
        "impl_system_summary": system_summary,
        "impl_file_sha256": file_sha256,
        "impl_text_search": text_search,
        "impl_compare_sha256": compare_sha256,
        "impl_incident_summary": incident_summary,
        "impl_simulated_log_check": simulated_log_check,
    }


def dry_run_filename_hint(parameters: Mapping[str, Any]) -> str | None:
    """Return a filename-only hint for dry-run path parameters."""
    path_value = parameters.get("path")
    if isinstance(path_value, str) and path_value.strip():
        return filename_only(path_value)
    return None
