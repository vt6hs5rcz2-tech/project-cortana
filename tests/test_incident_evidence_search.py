"""Tests for Milestone 17 evidence text-search bridge and bounded context."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from uuid import uuid4

import pytest

from src.commands import HELP_TEXT, handle_slash_command
from src.evidence_store import EvidenceStoreError, LocalEvidenceStore
from src.incident_analysis_packet import build_bounded_incident_context_for_display
from src.incident_evidence_search import (
    IncidentEvidenceSearchError,
    create_evidence_text_search_request,
)
from src.incident_repository import IncidentRelationshipError
from src.security_custody import create_custody_entry
from src.security_evidence import create_evidence_record
from src.security_incident import create_security_incident
from src.settings import Settings
from src.tool_approval import create_tool_approval
from src.tool_common import ToolAuthorizationError
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import build_default_tool_registry
from src.tool_request import replace_tool_execution_request
from src.tool_scope import create_authorized_scope
from src.workflow_repository import InMemoryWorkflowRunRepository
from tests.security_helpers import evidence_store, incident_repository
from tests.tool_helpers import tool_repository


def _register_evidence(
    *,
    store: LocalEvidenceStore,
    repo: object,
    source: Path,
    title: str = "Sample evidence",
) -> str:
    evidence_id = str(uuid4())
    digest, size, filename = store.copy_from_path(str(source), evidence_id=evidence_id)
    evidence = create_evidence_record(
        evidence_type="file",
        title=title,
        description="test evidence",
        sha256_hash=digest,
        source_size_bytes=size,
        collector="local-user",
        storage_status="copied",
        evidence_id=evidence_id,
        original_filename=filename,
    )
    custody = create_custody_entry(
        evidence_id=evidence_id,
        action="registered",
        actor="local-user",
        reason="test registration",
        resulting_hash=digest,
    )
    repo.add_evidence(evidence, [custody])  # type: ignore[attr-defined]
    return evidence_id


def test_link_evidence_to_incident_bidirectional_and_idempotent(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    source = tmp_path / "note.txt"
    source.write_text("hello search target\n", encoding="utf-8")
    evidence_id = _register_evidence(store=store, repo=repo, source=source)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )

    linked_incident, linked_evidence = repo.link_evidence_to_incident(
        incident.incident_id,
        evidence_id,
    )
    assert evidence_id in linked_incident.evidence_ids
    assert incident.incident_id in linked_evidence.related_incident_ids

    again_incident, again_evidence = repo.link_evidence_to_incident(
        incident.incident_id,
        evidence_id,
    )
    assert again_incident.evidence_ids.count(evidence_id) == 1
    assert again_evidence.related_incident_ids.count(incident.incident_id) == 1


def test_link_evidence_missing_records(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="low",
        )
    )
    with pytest.raises(IncidentRelationshipError):
        repo.link_evidence_to_incident(incident.incident_id, str(uuid4()))
    with pytest.raises(IncidentRelationshipError):
        repo.link_evidence_to_incident(str(uuid4()), str(uuid4()))


def test_resolve_stored_evidence_path_success_and_missing(tmp_path: Path) -> None:
    store = evidence_store(tmp_path)
    source = tmp_path / "payload.bin"
    source.write_bytes(b"abc123")
    evidence_id = str(uuid4())
    store.copy_from_path(str(source), evidence_id=evidence_id)

    resolved = store.resolve_stored_evidence_path(evidence_id)
    assert resolved.is_file()
    assert resolved.resolve() == resolved
    assert resolved.read_bytes() == b"abc123"

    with pytest.raises(EvidenceStoreError):
        store.resolve_stored_evidence_path(str(uuid4()))


@pytest.mark.skipif(os.name == "nt", reason="Symlink create often needs elevation on Windows")
def test_resolve_stored_evidence_path_rejects_symlink(tmp_path: Path) -> None:
    store = evidence_store(tmp_path)
    target = tmp_path / "real.bin"
    target.write_bytes(b"data")
    evidence_id = str(uuid4())
    store.copy_from_path(str(target), evidence_id=evidence_id)
    object_path = store.resolve_stored_evidence_path(evidence_id)
    object_path.unlink()
    link_path = store.directory_path / f"{evidence_id}.bin"
    link_path.symlink_to(target)
    with pytest.raises(EvidenceStoreError):
        store.resolve_stored_evidence_path(evidence_id)


def test_create_evidence_text_search_request_success(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    tools = tool_repository(tmp_path)
    source = tmp_path / "log.txt"
    source.write_text("alpha\nSENTINEL_M17_QUERY\n", encoding="utf-8")
    evidence_id = _register_evidence(store=store, repo=repo, source=source)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    repo.link_evidence_to_incident(incident.incident_id, evidence_id)
    scope = create_authorized_scope(
        scope_name="Evidence search scope",
        allowed_tool_ids=["text-search"],
        allowed_target_types=["local-file"],
        allowed_local_path_roots=[str(store.directory_path)],
        allowed_incident_ids=[incident.incident_id],
        notes="m17-evidence-search-scope",
    )
    tools.add_scope(scope)

    before = len(tools.list_requests())
    created = create_evidence_text_search_request(
        incident_repository=repo,
        evidence_store=store,
        tool_registry=build_default_tool_registry(),
        tool_repository=tools,
        incident_id=incident.incident_id,
        evidence_id=evidence_id,
        scope_id=scope.scope_id,
        query="SENTINEL_M17_QUERY",
    )
    assert len(tools.list_requests()) == before + 1
    request = tools.get_request(created.request_id)
    assert request is not None
    assert request.tool_id == "text-search"
    assert request.incident_id == incident.incident_id
    assert "SENTINEL_M17_QUERY" not in created.request_id
    assert str(store.directory_path) not in created.fingerprint_prefix


def test_bridge_rejects_unlinked_missing_and_auth(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    tools = tool_repository(tmp_path)
    source = tmp_path / "log.txt"
    source.write_text("needle\n", encoding="utf-8")
    evidence_id = _register_evidence(store=store, repo=repo, source=source)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    other = repo.add_incident(
        create_security_incident(
            title="Other",
            summary="Other summary text",
            severity="low",
        )
    )
    scope = create_authorized_scope(
        scope_name="Evidence search scope",
        allowed_tool_ids=["text-search"],
        allowed_target_types=["local-file"],
        allowed_local_path_roots=[str(store.directory_path)],
        allowed_incident_ids=[incident.incident_id],
        notes="m17-evidence-search-scope",
    )
    tools.add_scope(scope)
    registry = build_default_tool_registry()
    before = len(tools.list_requests())

    with pytest.raises(IncidentEvidenceSearchError, match="not linked"):
        create_evidence_text_search_request(
            incident_repository=repo,
            evidence_store=store,
            tool_registry=registry,
            tool_repository=tools,
            incident_id=incident.incident_id,
            evidence_id=evidence_id,
            scope_id=scope.scope_id,
            query="needle",
        )

    with pytest.raises(IncidentEvidenceSearchError, match="Evidence was not found"):
        create_evidence_text_search_request(
            incident_repository=repo,
            evidence_store=store,
            tool_registry=registry,
            tool_repository=tools,
            incident_id=incident.incident_id,
            evidence_id=str(uuid4()),
            scope_id=scope.scope_id,
            query="needle",
        )

    with pytest.raises(IncidentEvidenceSearchError, match="Incident was not found"):
        create_evidence_text_search_request(
            incident_repository=repo,
            evidence_store=store,
            tool_registry=registry,
            tool_repository=tools,
            incident_id=str(uuid4()),
            evidence_id=evidence_id,
            scope_id=scope.scope_id,
            query="needle",
        )

    repo.link_evidence_to_incident(incident.incident_id, evidence_id)
    with pytest.raises(IncidentEvidenceSearchError, match="not linked"):
        create_evidence_text_search_request(
            incident_repository=repo,
            evidence_store=store,
            tool_registry=registry,
            tool_repository=tools,
            incident_id=other.incident_id,
            evidence_id=evidence_id,
            scope_id=scope.scope_id,
            query="needle",
        )

    # Link other for membership, but scope still lacks other incident.
    repo.link_evidence_to_incident(other.incident_id, evidence_id)
    with pytest.raises(ToolAuthorizationError):
        create_evidence_text_search_request(
            incident_repository=repo,
            evidence_store=store,
            tool_registry=registry,
            tool_repository=tools,
            incident_id=other.incident_id,
            evidence_id=evidence_id,
            scope_id=scope.scope_id,
            query="needle",
        )

    other_root = tmp_path / "other-root"
    other_root.mkdir()
    wrong_root_scope = create_authorized_scope(
        scope_name="Wrong root",
        allowed_tool_ids=["text-search"],
        allowed_target_types=["local-file"],
        allowed_local_path_roots=[str(other_root)],
        allowed_incident_ids=[incident.incident_id],
        notes="m17-wrong-root-scope",
    )
    tools.add_scope(wrong_root_scope)
    with pytest.raises(ToolAuthorizationError):
        create_evidence_text_search_request(
            incident_repository=repo,
            evidence_store=store,
            tool_registry=registry,
            tool_repository=tools,
            incident_id=incident.incident_id,
            evidence_id=evidence_id,
            scope_id=wrong_root_scope.scope_id,
            query="needle",
        )

    assert len(tools.list_requests()) == before


def test_request_executes_through_defensive_tool_executor(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    tools = tool_repository(tmp_path)
    source = tmp_path / "log.txt"
    source.write_text("alpha\nSENTINEL_M17_LIVE\n", encoding="utf-8")
    evidence_id = _register_evidence(store=store, repo=repo, source=source)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    repo.link_evidence_to_incident(incident.incident_id, evidence_id)
    scope = create_authorized_scope(
        scope_name="Evidence search scope",
        allowed_tool_ids=["text-search"],
        allowed_target_types=["local-file"],
        allowed_local_path_roots=[str(store.directory_path)],
        allowed_incident_ids=[incident.incident_id],
        notes="m17-evidence-search-scope",
    )
    tools.add_scope(scope)
    created = create_evidence_text_search_request(
        incident_repository=repo,
        evidence_store=store,
        tool_registry=build_default_tool_registry(),
        tool_repository=tools,
        incident_id=incident.incident_id,
        evidence_id=evidence_id,
        scope_id=scope.scope_id,
        query="SENTINEL_M17_LIVE",
    )
    request = tools.get_request(created.request_id)
    assert request is not None
    definition = build_default_tool_registry().require("text-search")
    executor = DefensiveToolExecutor(incident_repository=repo)
    planned = executor.plan_dry_run(
        definition=definition,
        request=request,
        scope=scope,
    )
    assert planned.outcome == "planned"
    ready = replace_tool_execution_request(
        request,
        request_status="approved",
        dry_run_completed=True,
    )
    tools.update_request(ready)
    approval = create_tool_approval(
        request=ready,
        decision="approved",
        reason="m17 evidence search approval",
        approver="tester",
    )
    result = executor.execute(
        definition=definition,
        request=ready,
        scope=scope,
        approval=approval,
    )
    assert result.outcome == "succeeded"
    assert result.incident_id == incident.incident_id
    assert result.structured_data["match_count"] == 1


def test_bridge_has_no_process_or_direct_search_imports() -> None:
    source = Path("src/incident_evidence_search.py").read_text(encoding="utf-8")
    assert "safe_open_for_read" not in source
    assert "stream_text_search_from_safe_handle" not in source
    assert "hash_sha256_from_safe_handle" not in source
    assert "tool_process_runner" not in source
    assert "tool_process_safe_open" not in source
    assert "tool_process_text_search" not in source
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("tool_process" in name for name in imports)


def test_command_output_omits_path_and_query(tmp_path: Path) -> None:
    from src.active_memory import ActiveMemoryContext
    from src.conversation import ConversationHistory
    from src.document_extractor import DefaultTextExtractor
    from src.document_vault import JsonDocumentVault
    from src.memory_store import JsonMemoryStore

    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    tools = tool_repository(tmp_path)
    source = tmp_path / "log.txt"
    source.write_text("find SENTINEL_M17_HIDDEN\n", encoding="utf-8")
    evidence_id = _register_evidence(store=store, repo=repo, source=source)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    repo.link_evidence_to_incident(incident.incident_id, evidence_id)
    scope = create_authorized_scope(
        scope_name="Evidence search scope",
        allowed_tool_ids=["text-search"],
        allowed_target_types=["local-file"],
        allowed_local_path_roots=[str(store.directory_path)],
        allowed_incident_ids=[incident.incident_id],
        notes="m17-evidence-search-scope",
    )
    tools.add_scope(scope)

    result = handle_slash_command(
        (
            f"/evidence-search {incident.incident_id} | {evidence_id} | "
            f"{scope.scope_id} | SENTINEL_M17_HIDDEN"
        ),
        settings=Settings(openai_api_key="test-api-key", openai_model="test-model"),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memory.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "vault.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=repo,
        evidence_store=store,
        tool_registry=build_default_tool_registry(),
        tool_repository=tools,
        tool_executor=DefensiveToolExecutor(incident_repository=repo),
        workflow_run_repository=InMemoryWorkflowRunRepository(),
    )
    message = result.message
    assert "Evidence text-search request created" in message
    assert "SENTINEL_M17_HIDDEN" not in message
    assert str(store.directory_path) not in message
    assert ".bin" not in message
    assert "/incident-link-evidence" in HELP_TEXT
    assert "/evidence-search" in HELP_TEXT
    assert "/incident-context" in HELP_TEXT


def test_bounded_context_helper_shape_and_no_ai_imports(tmp_path: Path) -> None:
    repo = incident_repository(tmp_path)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    packet = build_bounded_incident_context_for_display(
        incident_repository=repo,
        workflow_run_repository=InMemoryWorkflowRunRepository(),
        incident_id=incident.incident_id,
    )
    assert packet.payload["omitted_evidence"] is True
    assert packet.payload["omitted_custody"] is True
    assert packet.payload["omitted_structured_tool_data"] is True
    assert "evidence" not in packet.payload
    assert "path" not in packet.serialized_text
    assert "structured_data" not in packet.serialized_text

    packet_source = Path("src/incident_analysis_packet.py").read_text(encoding="utf-8")
    assert "ai_service" not in packet_source
    assert "openai_client" not in packet_source
    assert "OpenAI" not in packet_source
    assert "build_bounded_incident_context_for_display" in packet_source

    bridge_commands = Path("src/incident_evidence_search_commands.py").read_text(
        encoding="utf-8"
    )
    assert "IncidentAnalysisService" not in bridge_commands
    assert "generate_incident_analysis_response" not in bridge_commands
    assert "ai_service" not in bridge_commands
    assert "openai_client" not in bridge_commands


def test_incident_link_evidence_command(tmp_path: Path) -> None:
    from src.active_memory import ActiveMemoryContext
    from src.conversation import ConversationHistory
    from src.document_extractor import DefaultTextExtractor
    from src.document_vault import JsonDocumentVault
    from src.memory_store import JsonMemoryStore

    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    source = tmp_path / "log.txt"
    source.write_text("x\n", encoding="utf-8")
    evidence_id = _register_evidence(store=store, repo=repo, source=source)
    incident = repo.add_incident(
        create_security_incident(
            title="Case",
            summary="Summary text for incident",
            severity="medium",
        )
    )
    result = handle_slash_command(
        f"/incident-link-evidence {incident.incident_id} {evidence_id}",
        settings=Settings(openai_api_key="test-api-key", openai_model="test-model"),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memory.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "vault.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=repo,
        evidence_store=store,
    )
    assert "Linked evidence" in result.message
    reloaded = repo.get_incident(incident.incident_id)
    assert reloaded is not None
    assert evidence_id in reloaded.evidence_ids
