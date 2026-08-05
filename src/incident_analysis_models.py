"""Strictly typed models for Milestone 12 incident analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.incident_analysis_common import AnalysisKind, AnalysisStatus


@dataclass(frozen=True)
class IncidentAnalysisPacket:
    """Immutable sanitized allowlisted packet for one incident analysis."""

    incident_id: str
    payload: dict[str, Any]
    serialized_text: str
    character_count: int
    packet_fingerprint: str
    selected_event_count: int
    selected_indicator_count: int
    selected_note_count: int
    selected_workflow_count: int
    selected_tool_summary_count: int


@dataclass(frozen=True)
class IncidentAnalysisRequest:
    """Prepared analysis request awaiting explicit run confirmation."""

    analysis_id: str
    incident_id: str
    scope_id: str
    analysis_kind: AnalysisKind
    selected_event_ids: tuple[str, ...]
    selected_indicator_ids: tuple[str, ...]
    selected_note_ids: tuple[str, ...]
    selected_workflow_run_ids: tuple[str, ...]
    created_at: str
    previewed: bool
    packet_fingerprint: str
    boundary_token: str
    packet: IncidentAnalysisPacket


@dataclass(frozen=True)
class IncidentAnalysisResult:
    """In-memory AI analysis result retained for the current process only."""

    analysis_id: str
    incident_id: str
    scope_id: str
    analysis_kind: AnalysisKind
    packet_fingerprint: str
    boundary_token: str
    advisory_output: str
    created_at: str
    completed_at: str
    status: AnalysisStatus
    error_code: str | None
    note_saved: bool
    saved_note_id: str | None
    selected_event_count: int
    selected_indicator_count: int
    selected_note_count: int
    selected_workflow_count: int
    packet_character_count: int
    output_character_count: int
