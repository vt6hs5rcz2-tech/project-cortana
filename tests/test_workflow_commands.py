"""Command tests for Milestone 10 workflow slash commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.commands import (
    ABOUT_TEXT,
    HELP_TEXT,
    CommandResult,
    format_status,
    handle_slash_command,
)
from src.conversation import ConversationHistory
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_repository import JsonToolControlRepository
from src.workflow_commands import (
    WORKFLOW_COMMAND_NAMES,
    create_default_workflow_services,
)
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import WorkflowRunRepository
from tests.tool_helpers import incident_repository, tool_repository


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def responses(self) -> Any:
        self.calls += 1
        raise AssertionError("Milestone 10 commands must not call the AI")


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


WorkflowServices = tuple[
    JsonToolControlRepository,
    JsonIncidentRepository,
    ToolRegistry,
    DefensiveToolExecutor,
    WorkflowRegistry,
    WorkflowRunRepository,
    WorkflowExecutor,
]


def _services(tmp_path: Path) -> WorkflowServices:
    tools_repo = tool_repository(tmp_path)
    incidents = incident_repository(tmp_path)
    registry = build_default_tool_registry()
    tool_executor = DefensiveToolExecutor(incident_repository=incidents)
    workflow_registry, workflow_runs, workflow_executor = (
        create_default_workflow_services(
            tool_registry=registry,
            tool_repository=tools_repo,
            tool_executor=tool_executor,
            incident_repository=incidents,
            persist_runs=False,
        )
    )
    return (
        tools_repo,
        incidents,
        registry,
        tool_executor,
        workflow_registry,
        workflow_runs,
        workflow_executor,
    )


def _run(
    message: str,
    tmp_path: Path,
    *,
    client: OpenAIClient | None = None,
    services: WorkflowServices | None = None,
) -> tuple[
    CommandResult,
    JsonToolControlRepository,
    WorkflowRunRepository,
    WorkflowServices,
]:
    if services is None:
        services = _services(tmp_path)
    (
        tools_repo,
        incidents,
        registry,
        tool_executor,
        workflow_registry,
        workflow_runs,
        workflow_executor,
    ) = services
    result = handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=incidents,
        tool_registry=registry,
        tool_repository=tools_repo,
        tool_executor=tool_executor,
        workflow_registry=workflow_registry,
        workflow_run_repository=workflow_runs,
        workflow_executor=workflow_executor,
        client=client,
    )
    return result, tools_repo, workflow_runs, services


def test_playbooks_and_show(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    listed, _repo, _runs, services = _run("/playbooks", tmp_path, client=client)
    assert "platform-baseline" in (listed.message or "")
    assert "mock-log-triage" in (listed.message or "")

    shown, _repo, _runs, _services_out = _run(
        "/playbook-show platform-baseline",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Playbook details" in (shown.message or "")
    assert "system-summary" in (shown.message or "")
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_defaults_to_dry_run_and_status(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    scope_result, repo, _runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Authorized scope created" in (scope_result.message or "")
    scope_id = repo.list_scopes()[0].scope_id

    run_result, _repo, runs, services = _run(
        f"/playbook-run platform-baseline | {scope_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Mode: dry-run" in (run_result.message or "")
    run_id = runs.list_runs()[0].run_id
    assert runs.list_runs()[0].dry_run is True
    assert runs.list_runs()[0].status == "completed"

    status, _repo, _runs, _services_out = _run(
        f"/playbook-status {run_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Playbook run status" in (status.message or "")
    assert "Dry-run: True" in (status.message or "")
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_execute(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    _scope_result, repo, _runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id
    run_result, _repo, runs, _services_out = _run(
        f"/playbook-run platform-baseline --execute | {scope_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Mode: execute" in (run_result.message or "")
    assert runs.list_runs()[0].dry_run is False
    assert runs.list_runs()[0].status == "completed"
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_new_operation_flag_is_execute_only(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    _scope_result, repo, _runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id
    rejected, _repo, runs, services = _run(
        f"/playbook-run platform-baseline --new-operation | {scope_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (rejected.message or "")
    assert runs.list_runs() == []
    run_result, _repo, runs, _services_out = _run(
        f"/playbook-run platform-baseline --execute --new-operation | {scope_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Mode: execute" in (run_result.message or "")
    assert runs.list_runs()[0].status == "completed"
    assert runs.list_runs()[0].dry_run is False
    assert cast(CountingClient, client).calls == 0


def test_malformed_and_unknown_playbook(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    missing_args, _repo, _runs, services = _run(
        "/playbook-run",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (missing_args.message or "")

    unknown, _repo, _runs, services = _run(
        "/playbook-show not-a-playbook",
        tmp_path,
        client=client,
        services=services,
    )
    assert "No registered playbook" in (unknown.message or "")

    bad_status, _repo, _runs, _services_out = _run(
        "/playbook-status not-a-uuid",
        tmp_path,
        client=client,
        services=services,
    )
    assert "No workflow run found" in (bad_status.message or "")
    assert cast(CountingClient, client).calls == 0


def test_partial_workflow_injection_uses_all_or_nothing_consistent_services(
    tmp_path: Path,
) -> None:
    """Incomplete workflow DI must not pair an executor with a different run repo."""
    tools_repo = tool_repository(tmp_path)
    incidents = incident_repository(tmp_path)
    registry = build_default_tool_registry()
    tool_executor = DefensiveToolExecutor(incident_repository=incidents)
    (
        injected_workflow_registry,
        injected_runs,
        injected_workflow_executor,
    ) = create_default_workflow_services(
        tool_registry=registry,
        tool_repository=tools_repo,
        tool_executor=tool_executor,
        incident_repository=incidents,
        persist_runs=False,
    )

    # Seed a scope through the shared tool repository.
    scope_result = handle_slash_command(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=incidents,
        tool_registry=registry,
        tool_repository=tools_repo,
        tool_executor=tool_executor,
        workflow_registry=injected_workflow_registry,
        workflow_run_repository=injected_runs,
        workflow_executor=injected_workflow_executor,
    )
    assert "Authorized scope created" in (scope_result.message or "")
    scope_id = tools_repo.list_scopes()[0].scope_id

    # Inject only the executor from set A; omit registry/repo so fallback triggers.
    # Before the fix this would write into injected_runs via the partial executor
    # while status read a different ephemeral repository.
    run_result = handle_slash_command(
        f"/playbook-run platform-baseline | {scope_id}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=incidents,
        tool_registry=registry,
        tool_repository=tools_repo,
        tool_executor=tool_executor,
        workflow_executor=injected_workflow_executor,
    )
    assert "Mode: dry-run" in (run_result.message or "")
    assert "Playbook run completed" in (run_result.message or "")

    # Injected set A must remain empty; incomplete DI used one fresh consistent set.
    assert injected_runs.list_runs() == []

    # Production-style complete injection preserves the same service set.
    complete_run = handle_slash_command(
        f"/playbook-run platform-baseline | {scope_id}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=incidents,
        tool_registry=registry,
        tool_repository=tools_repo,
        tool_executor=tool_executor,
        workflow_registry=injected_workflow_registry,
        workflow_run_repository=injected_runs,
        workflow_executor=injected_workflow_executor,
    )
    assert "Mode: dry-run" in (complete_run.message or "")
    assert len(injected_runs.list_runs()) == 1
    complete_run_id = injected_runs.list_runs()[0].run_id
    complete_status = handle_slash_command(
        f"/playbook-status {complete_run_id}",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=incidents,
        tool_registry=registry,
        tool_repository=tools_repo,
        tool_executor=tool_executor,
        workflow_registry=injected_workflow_registry,
        workflow_run_repository=injected_runs,
        workflow_executor=injected_workflow_executor,
    )
    assert "Playbook run status" in (complete_status.message or "")
    assert complete_run_id in (complete_status.message or "")


def test_help_about_status_include_workflow(tmp_path: Path) -> None:
    assert "/playbooks" in HELP_TEXT
    assert "/playbook-run" in HELP_TEXT
    assert "workflow" in ABOUT_TEXT.lower()
    assert "controlled defensive" in ABOUT_TEXT.lower()

    services = _services(tmp_path)
    _result, repo, runs, _services_out = _run(
        "/playbooks",
        tmp_path,
        services=services,
    )
    status = format_status(
        _settings(),
        ConversationHistory(),
        JsonMemoryStore(tmp_path / "memories2.json"),
        ActiveMemoryContext(),
        JsonDocumentVault(tmp_path / "documents2.json"),
        tool_registry=services[2],
        tool_repository=repo,
        workflow_registry=services[4],
        workflow_run_repository=runs,
    )
    assert "Defensive workflow orchestration: enabled" in status
    assert "Workflow run persistence: enabled" in status
    assert "Workflow incident linkage: enabled" in status
    assert "Workflow single-instance coordination: disabled" in status
    assert "Maximum retained workflow runs: 100" in status
    assert "Maximum retained workflow audit entries: 500" in status
    assert "External playbook loading: disabled" in status
    assert WORKFLOW_COMMAND_NAMES


def test_playbook_run_with_incident_and_status_display(tmp_path: Path) -> None:
    from src.security_incident import create_security_incident

    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    incidents = services[1]
    incident = incidents.add_incident(
        create_security_incident(
            title="Command linked case",
            summary="Summary",
            severity="low",
        )
    )
    _scope_result, repo, _runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id

    run_result, _repo, runs, services = _run(
        f"/playbook-run platform-baseline | {scope_id} | {incident.incident_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Mode: dry-run" in (run_result.message or "")
    assert incident.incident_id in (run_result.message or "")
    assert runs.list_runs()[0].incident_id == incident.incident_id
    assert runs.list_runs()[0].status == "completed"
    assert len(incidents.list_notes(incident.incident_id)) == 1

    status, _repo, _runs, services = _run(
        f"/playbook-status {runs.list_runs()[0].run_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert f"Incident ID: {incident.incident_id}" in (status.message or "")

    execute_result, _repo, runs, _services_out = _run(
        (
            f"/playbook-run platform-baseline --execute | {scope_id} | "
            f"{incident.incident_id}"
        ),
        tmp_path,
        client=client,
        services=services,
    )
    assert "Mode: execute" in (execute_result.message or "")
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_rejects_malformed_incident_fields(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    _scope_result, repo, _runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id

    blank_incident, _repo, _runs, services = _run(
        f"/playbook-run platform-baseline | {scope_id} |    ",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (blank_incident.message or "")

    extra_fields, _repo, _runs, services = _run(
        f"/playbook-run platform-baseline | {scope_id} | {scope_id} | extra",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (extra_fields.message or "")

    unknown_incident, _repo, runs, _services_out = _run(
        (
            f"/playbook-run platform-baseline | {scope_id} | "
            "11111111-1111-1111-1111-111111111111"
        ),
        tmp_path,
        client=client,
        services=services,
    )
    # Preflight failure still returns a run message, not AI content.
    assert "Playbook run preflight_failed" in (unknown_incident.message or "")
    assert runs.list_runs()[-1].status == "preflight_failed"
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_rejects_duplicate_execute_flag(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    _scope_result, repo, runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id
    before = len(runs.list_runs())

    result, _repo, runs, _services_out = _run(
        f"/playbook-run platform-baseline --execute --execute | {scope_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (result.message or "")
    assert len(runs.list_runs()) == before
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_rejects_unknown_flag(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    _scope_result, repo, runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id
    before = len(runs.list_runs())

    result, _repo, runs, _services_out = _run(
        f"/playbook-run platform-baseline --foo | {scope_id}",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (result.message or "")
    assert len(runs.list_runs()) == before
    assert cast(CountingClient, client).calls == 0


def test_playbook_run_rejects_malformed_non_uuid_incident_id(tmp_path: Path) -> None:
    client = cast(OpenAIClient, CountingClient())
    services = _services(tmp_path)
    _scope_result, repo, runs, services = _run(
        "/scope-new Baseline | system-summary,simulated-log-check | none | review",
        tmp_path,
        client=client,
        services=services,
    )
    scope_id = repo.list_scopes()[0].scope_id
    before = len(runs.list_runs())

    result, _repo, runs, _services_out = _run(
        f"/playbook-run platform-baseline | {scope_id} | not-a-uuid",
        tmp_path,
        client=client,
        services=services,
    )
    assert "Usage" in (result.message or "")
    assert len(runs.list_runs()) == before
    assert cast(CountingClient, client).calls == 0
