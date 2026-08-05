"""Tests for Milestone 9 defensive tool slash commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.commands import CommandResult, format_status, handle_slash_command
from src.conversation import ConversationHistory
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from src.tool_commands import TOOL_COMMAND_NAMES
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import build_default_tool_registry
from src.tool_repository import JsonToolControlRepository
from tests.tool_helpers import incident_repository, tool_repository


class CountingClient:
    """OpenAI client stand-in that records whether AI was called."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def responses(self) -> Any:
        self.calls += 1
        raise AssertionError("Milestone 9 commands must not call the AI")


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _run(
    message: str,
    tmp_path: Path,
    *,
    history: ConversationHistory | None = None,
    client: OpenAIClient | None = None,
) -> tuple[CommandResult, JsonToolControlRepository, JsonIncidentRepository]:
    repo = tool_repository(tmp_path)
    incidents = incident_repository(tmp_path)
    registry = build_default_tool_registry()
    executor = DefensiveToolExecutor(incident_repository=incidents)
    result = handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=history or ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=incidents,
        tool_registry=registry,
        tool_repository=repo,
        tool_executor=executor,
        client=client,
    )
    return result, repo, incidents


def test_missing_argument_cases(tmp_path: Path) -> None:
    cases = [
        "/tool",
        "/scope",
        "/scope-disable",
        "/tool-request-show",
        "/tool-dry-run",
        "/tool-cancel",
        "/tool-run",
        "/tool-result",
        "/scope-new incomplete",
        "/tool-request incomplete",
        "/tool-approve incomplete",
        "/tool-reject incomplete",
    ]
    for message in cases:
        result, _repo, _incidents = _run(message, tmp_path / hashlib.md5(message.encode()).hexdigest())
        assert result.message is not None
        assert result.message.startswith("Cortana:")


def test_malformed_json_and_delimiter(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    create, repo, _ = _run(
        f"/scope-new Lab | system-summary | none | justification text",
        tmp_path,
    )
    assert "scope created" in (create.message or "").lower() or "Authorized scope" in (
        create.message or ""
    )
    scope_id = repo.list_scopes()[0].scope_id

    bad_json, _, _ = _run(
        f"/tool-request system-summary | {scope_id} | [] | because",
        tmp_path,
    )
    assert "Usage" in (bad_json.message or "") or "JSON" in (bad_json.message or "")

    bad_delim, _, _ = _run(
        "/tool-request system-summary | only-one-field",
        tmp_path,
    )
    assert "Usage" in (bad_delim.message or "")


def test_approval_and_run_workflow(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    sample = root / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    client = cast(OpenAIClient, CountingClient())
    history = ConversationHistory()
    history.add_user_message("keep me")
    history.add_assistant_message("kept")

    scope_result, repo, _ = _run(
        f"/scope-new Hash lab | file-sha256 | {root} | Hash review",
        tmp_path,
        history=history,
        client=client,
    )
    assert "Authorized scope created" in (scope_result.message or "")
    scope_id = repo.list_scopes()[0].scope_id

    params = json.dumps({"path": str(sample)})
    request_result, repo, _ = _run(
        f"/tool-request file-sha256 | {scope_id} | {params} | Verify hash",
        tmp_path,
        history=history,
        client=client,
    )
    assert "Tool request created" in (request_result.message or "")
    request_id = repo.list_requests()[0].request_id

    dry, repo, _ = _run(
        f"/tool-dry-run {request_id}",
        tmp_path,
        history=history,
        client=client,
    )
    assert "Dry-run plan recorded" in (dry.message or "")
    assert repo.list_requests()[0].dry_run_completed is True

    approve, repo, _ = _run(
        f"/tool-approve {request_id} | Looks safe after dry-run",
        tmp_path,
        history=history,
        client=client,
    )
    assert "approved" in (approve.message or "")

    run_result, repo, _ = _run(
        f"/tool-run {request_id}",
        tmp_path,
        history=history,
        client=client,
    )
    assert "Tool result recorded" in (run_result.message or "")
    assert repo.list_requests()[0].request_status == "succeeded"
    result_id = repo.list_results()[-1].result_id

    show, _, _ = _run(
        f"/tool-result {result_id}",
        tmp_path,
        history=history,
        client=client,
    )
    assert "Outcome: succeeded" in (show.message or "")
    assert str(sample) not in (show.message or "")

    audit, _, _ = _run("/tool-audit", tmp_path, history=history, client=client)
    assert "Tool audit entries" in (audit.message or "")

    assert history.completed_turn_count == 1
    assert cast(CountingClient, client).calls == 0


def test_cancellation_prevents_execution(tmp_path: Path) -> None:
    _, repo, _ = _run(
        "/scope-new Cancel lab | system-summary | none | cancel test",
        tmp_path,
    )
    scope_id = repo.list_scopes()[0].scope_id
    _, repo, _ = _run(
        f'/tool-request system-summary | {scope_id} | {{}} | cancel me',
        tmp_path,
    )
    request_id = repo.list_requests()[0].request_id
    _, repo, _ = _run(f"/tool-cancel {request_id}", tmp_path)
    assert repo.list_requests()[0].request_status == "cancelled"
    denied, _, _ = _run(f"/tool-run {request_id}", tmp_path)
    assert "denied" in (denied.message or "").lower()


def test_case_insensitive_commands_and_list_show(tmp_path: Path) -> None:
    result, _, _ = _run("/TOOLS", tmp_path)
    assert "Enabled defensive tools" in (result.message or "")
    detail, _, _ = _run("/Tool system-summary", tmp_path)
    assert "Tool ID: system-summary" in (detail.message or "")
    scopes, _, _ = _run("/scopes", tmp_path)
    assert scopes.message is not None


def test_absolute_path_chat_preserved(tmp_path: Path) -> None:
    from src.commands import parse_slash_input

    assert parse_slash_input("/etc/passwd") is None
    assert parse_slash_input("/tools") == "tools"


def test_no_sensitive_output_in_scope_show(tmp_path: Path) -> None:
    root = tmp_path / "sensitive-root-name"
    root.mkdir()
    _, repo, _ = _run(
        f"/scope-new Notes | system-summary | none | SECRET_SCOPE_NOTE_MARKER",
        tmp_path,
    )
    scope_id = repo.list_scopes()[0].scope_id
    shown, _, _ = _run(f"/scope {scope_id}", tmp_path)
    assert "SECRET_SCOPE_NOTE_MARKER" not in (shown.message or "")
    assert str(root) not in (shown.message or "")


def test_status_flags_and_privacy(tmp_path: Path) -> None:
    registry = build_default_tool_registry()
    repo = tool_repository(tmp_path)
    _, repo, _ = _run(
        "/scope-new Status | system-summary | none | status note",
        tmp_path,
    )
    status = format_status(
        _settings(),
        ConversationHistory(),
        JsonMemoryStore(tmp_path / "memories2.json"),
        ActiveMemoryContext(),
        JsonDocumentVault(tmp_path / "documents2.json"),
        tool_registry=registry,
        tool_repository=repo,
    )
    assert "Defensive tool framework: enabled" in status
    assert "Arbitrary shell execution: disabled" in status
    assert "External tool execution: disabled" in status
    assert "Autonomous remediation: disabled" in status
    assert "Process-isolated tool execution: disabled" in status
    assert "Process-isolated tool termination: disabled" in status
    assert "Scope enforcement: enabled" in status
    assert "Human approval: enabled" in status
    assert "Dry-run enforcement: enabled" in status
    assert "Active scopes: 1" in status
    assert repo.list_scopes()[0].scope_id not in status
    assert "status note" not in status


def test_all_tool_commands_are_registered() -> None:
    assert "tools" in TOOL_COMMAND_NAMES
    assert "tool-run" in TOOL_COMMAND_NAMES
    assert "tool-audit" in TOOL_COMMAND_NAMES


def test_help_lists_tool_commands(tmp_path: Path) -> None:
    result, _, _ = _run("/help", tmp_path)
    assert "/tools" in (result.message or "")
    assert "/tool-run" in (result.message or "")
    assert "/tool-audit" in (result.message or "")


def test_incident_link_and_nonexistent_incident(tmp_path: Path) -> None:
    from src.security_incident import create_security_incident

    _, repo, incidents = _run(
        "/scope-new Incident lab | incident-summary | none | incident summary",
        tmp_path,
    )
    scope_id = repo.list_scopes()[0].scope_id
    missing, _, _ = _run(
        f'/tool-request incident-summary | {scope_id} | '
        '{"incident_id":"99999999-9999-9999-9999-999999999999"} | link',
        tmp_path,
    )
    assert "not found" in (missing.message or "").lower() or "Usage" in (
        missing.message or ""
    )

    incident = incidents.add_incident(
        create_security_incident(
            title="T",
            summary="S",
            severity="low",
        )
    )
    before_events = incidents.event_count()
    before_notes = len(incidents.list_notes(incident.incident_id))
    ok, repo, incidents = _run(
        f'/tool-request incident-summary | {scope_id} | '
        f'{{"incident_id":"{incident.incident_id}"}} | summarize',
        tmp_path,
    )
    assert "Tool request created" in (ok.message or "")
    request_id = repo.list_requests()[-1].request_id
    _run(f"/tool-dry-run {request_id}", tmp_path)
    _run(f"/tool-run {request_id}", tmp_path)
    assert incidents.event_count() == before_events
    assert len(incidents.list_notes(incident.incident_id)) == before_notes
    assert incidents.evidence_count() == 0
