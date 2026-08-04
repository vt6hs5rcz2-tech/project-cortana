"""Tests for Milestone 8 local security slash commands."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.commands import (
    HELP_TEXT,
    CommandOutcome,
    CommandResult,
    format_status,
    handle_slash_command,
    parse_slash_input,
)
from src.conversation import ConversationHistory
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.evidence_store import LocalEvidenceStore
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from tests.security_helpers import evidence_store, incident_repository

FAKE_CLIENT = cast(OpenAIClient, MagicMock())


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _run(
    message: str,
    tmp_path: Path,
    *,
    history: ConversationHistory | None = None,
    repo: JsonIncidentRepository | None = None,
    store: LocalEvidenceStore | None = None,
    client: OpenAIClient | None = FAKE_CLIENT,
) -> CommandResult:
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=history or ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=repo or incident_repository(tmp_path),
        evidence_store=store or evidence_store(tmp_path),
        client=client,
    )


def test_missing_argument_paths(tmp_path: Path) -> None:
    """Malformed Milestone 8 commands should return safe local usage messages."""
    assert "Usage: /event-new" in str(_run("/event-new", tmp_path).message)
    assert "Usage: /incident-new" in str(_run("/incident-new", tmp_path).message)
    assert "Usage: /indicator-add" in str(_run("/indicator-add", tmp_path).message)
    assert "Usage: /evidence-register" in str(
        _run("/evidence-register", tmp_path).message
    )
    assert "event id" in str(_run("/event", tmp_path).message).lower()
    assert "incident id" in str(_run("/incident", tmp_path).message).lower()


def test_event_incident_indicator_flows_case_insensitive(tmp_path: Path) -> None:
    """Create/list/show/status flows should work and match case-insensitively."""
    repo = incident_repository(tmp_path)
    created = _run(
        "/EVENT-NEW high | Phishing report | User clicked a link",
        tmp_path,
        repo=repo,
    )
    assert created.outcome == CommandOutcome.CONTINUE
    assert "Security event saved" in str(created.message)

    events = _run("/Events", tmp_path, repo=repo)
    assert "Phishing report" in str(events.message)

    event_id = repo.list_events()[0].event_id
    shown = _run(f"/event {event_id}", tmp_path, repo=repo)
    assert "User clicked a link" in str(shown.message)

    status = _run(f"/event-status {event_id} investigating", tmp_path, repo=repo)
    assert "investigating" in str(status.message)

    incident = _run(
        "/incident-new medium | Case One | Summary text",
        tmp_path,
        repo=repo,
    )
    assert "Security incident saved" in str(incident.message)
    incident_id = repo.list_incidents()[0].incident_id

    linked = _run(
        f"/incident-link-event {incident_id} {event_id}",
        tmp_path,
        repo=repo,
    )
    assert "Linked event" in str(linked.message)

    duplicate = _run(
        f"/incident-link-event {incident_id} {event_id}",
        tmp_path,
        repo=repo,
    )
    assert "already linked" in str(duplicate.message).lower()

    unlinked = _run(
        f"/incident-unlink-event {incident_id} {event_id}",
        tmp_path,
        repo=repo,
    )
    assert "Unlinked event" in str(unlinked.message)

    indicator = _run(
        "/indicator-add domain | Example.COM | 75",
        tmp_path,
        repo=repo,
    )
    assert "Indicator saved" in str(indicator.message)
    assert "example.com" in str(_run("/indicators", tmp_path, repo=repo).message)


def test_evidence_register_path_with_spaces_and_verify(tmp_path: Path) -> None:
    """Evidence registration must preserve spaces in paths and verify hashes."""
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    source = tmp_path / "folder with spaces" / "payload file.txt"
    source.parent.mkdir()
    source.write_text("evidence-body", encoding="utf-8")

    result = _run(
        f"/evidence-register {source} | Capture title | Capture description",
        tmp_path,
        repo=repo,
        store=store,
    )
    assert "Evidence registered" in str(result.message)
    assert str(source) not in str(result.message)
    assert str(store.directory_path) not in str(result.message)

    evidence_id = repo.list_evidence()[0].evidence_id
    shown = _run(f"/evidence-show {evidence_id}", tmp_path, repo=repo, store=store)
    assert "Capture description" in str(shown.message)
    assert str(store.directory_path) not in str(shown.message)

    verified = _run(
        f"/evidence-verify {evidence_id}",
        tmp_path,
        repo=repo,
        store=store,
    )
    assert "Hash match" in str(verified.message)
    assert len(repo.list_custody_entries(evidence_id)) >= 3


def test_notes_timeline_and_invalid_ids(tmp_path: Path) -> None:
    """Notes and timelines should work; invalid IDs should fail safely."""
    repo = incident_repository(tmp_path)
    _run("/incident-new low | Timeline Case | Summary", tmp_path, repo=repo)
    incident_id = repo.list_incidents()[0].incident_id
    _run(
        "/event-new low | Timeline Event | Event body",
        tmp_path,
        repo=repo,
    )
    event_id = repo.list_events()[0].event_id
    _run(f"/incident-link-event {incident_id} {event_id}", tmp_path, repo=repo)
    _run(
        f"/incident-add-note {incident_id} | observation | Saw unusual activity",
        tmp_path,
        repo=repo,
    )

    notes = _run(f"/incident-notes {incident_id}", tmp_path, repo=repo)
    assert "Saw unusual activity" in str(notes.message)

    timeline = _run(f"/incident-timeline {incident_id}", tmp_path, repo=repo)
    assert "Timeline Event" in str(timeline.message) or "event" in str(timeline.message)

    missing = _run("/event not-a-uuid", tmp_path, repo=repo)
    assert "No saved event" in str(missing.message)


def test_security_commands_never_call_ai_or_alter_history(tmp_path: Path) -> None:
    """Milestone 8 commands must remain local and leave conversation history alone."""
    history = ConversationHistory()
    history.add_user_message("hello")
    history.add_assistant_message("hi")
    client = MagicMock()

    result = _run(
        "/event-new low | Title | Description",
        tmp_path,
        history=history,
        client=cast(OpenAIClient, client),
    )

    assert result.outcome == CommandOutcome.CONTINUE
    assert client.responses.create.call_count == 0
    assert history.completed_turn_count == 1


def test_clear_preserves_incident_data(tmp_path: Path) -> None:
    """/clear should clear session history/manifest but not incident records."""
    repo = incident_repository(tmp_path)
    history = ConversationHistory()
    history.add_user_message("hello")
    history.add_assistant_message("hi")
    _run("/event-new low | Keep | Keep me", tmp_path, repo=repo, history=history)

    cleared = _run("/clear", tmp_path, repo=repo, history=history)
    assert "Incident records and evidence were left unchanged" in str(cleared.message)
    assert history.turns == []
    assert repo.event_count() == 1


def test_absolute_path_still_reaches_ordinary_chat() -> None:
    """Absolute path-like input should not be treated as a Milestone 8 command."""
    assert parse_slash_input("/etc/passwd") is None
    assert parse_slash_input("/event-new") == "event-new"


def test_help_and_status_include_milestone_eight(tmp_path: Path) -> None:
    """Help and status should expose Milestone 8 capabilities without secrets/paths."""
    assert "/event-new" in HELP_TEXT
    assert "/evidence-register" in HELP_TEXT

    repo = incident_repository(tmp_path)
    _run("/event-new low | Title | Desc", tmp_path, repo=repo)
    status = format_status(
        _settings(),
        ConversationHistory(),
        JsonMemoryStore(tmp_path / "memories.json"),
        ActiveMemoryContext(),
        JsonDocumentVault(tmp_path / "documents.json"),
        incident_repository=repo,
    ).lower()

    assert "incident repository: enabled" in status
    assert "saved events: 1" in status
    assert "automated response: disabled" in status
    assert "external threat-intelligence lookups: disabled" in status
    assert "incident ai-context injection: disabled" in status
    assert "single-instance coordination: disabled" in status
    assert "incidents.json" not in status
    assert "test-api-key" not in status
    assert str(repo.file_path).lower() not in status
