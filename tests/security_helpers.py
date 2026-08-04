"""Shared helpers for Milestone 8 security foundation tests."""

from pathlib import Path

from src.evidence_store import LocalEvidenceStore
from src.incident_repository import JsonIncidentRepository


def incident_repository(tmp_path: Path) -> JsonIncidentRepository:
    """Return a temporary JSON incident repository."""
    return JsonIncidentRepository(tmp_path / "incidents.json")


def evidence_store(tmp_path: Path) -> LocalEvidenceStore:
    """Return a temporary local evidence store."""
    return LocalEvidenceStore(tmp_path / "evidence")
