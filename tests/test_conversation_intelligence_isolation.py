"""Isolation/security tests for Milestone 27 conversational intelligence."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.config import PROJECT_ROOT
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore

M27_MODULES = (
    "src/conversation_state.py",
    "src/conversation_intelligence.py",
)

FORBIDDEN_IMPORTS = frozenset(
    {
        "tool_executor",
        "tool_registry",
        "tool_approval",
        "workflow_executor",
        "workflow_registry",
        "calendar_service",
        "calendar_commands",
        "reminder_service",
        "reminder_commands",
        "memory_store",
        "security_commands",
        "incident_repository",
        "incident_analysis_service",
        "document_vault",
        "study_service",
        "commands",
        "assistant_orchestrator",
    }
)


def _module_level_src_imports(relative_path: str) -> set[str]:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "src" and len(parts) > 1:
                imported.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "src" and len(parts) > 1:
                    imported.add(parts[1])
    return imported


def test_m27_modules_avoid_operational_domains() -> None:
    for relative in M27_MODULES:
        imported = _module_level_src_imports(relative)
        assert imported.isdisjoint(FORBIDDEN_IMPORTS), relative


def test_interpret_does_not_call_tool_or_workflow_surfaces() -> None:
    tool_executor = MagicMock()
    workflow_executor = MagicMock()
    calendar_service = MagicMock()
    reminder_service = MagicMock()
    memory_store = MagicMock()

    state = ConversationState()
    state.set_unresolved_question("Run the playbook?")
    state.set_offered_options(["run tool", "schedule meeting"])
    intel = ConversationIntelligence()
    for text in (
        "yes",
        "the first one",
        "I meant Tuesday",
        "Forget that",
        "run a tool",
        "schedule a meeting",
        "remember this forever",
    ):
        guidance = intel.interpret(text, state)
        assert guidance.authorizes_privileged_action is False

    tool_executor.assert_not_called()
    workflow_executor.assert_not_called()
    calendar_service.assert_not_called()
    reminder_service.assert_not_called()
    memory_store.assert_not_called()


def test_correction_text_does_not_bypass_orchestrator_privileges(
    tmp_path: Path,
) -> None:
    """NL privileged phrases still require the existing orchestrator/commands."""
    vault = JsonDocumentVault(tmp_path / "docs.json")
    store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=vault,
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    # Conversational correction alone does not create memories.
    state = ConversationState()
    intel = ConversationIntelligence()
    guidance = intel.interpret("Forget that", state)
    assert guidance.authorizes_privileged_action is False
    assert store.list_memories() == []

    # Explicit remember still goes through orchestrator, not M27.
    result = orchestrator.try_handle("remember keep this note")
    assert result is not None
    assert len(store.list_memories()) == 1


def test_guidance_fields_contain_no_authorization_hooks() -> None:
    guidance = ConversationIntelligence().interpret(
        "yes",
        ConversationState(unresolved_question="Continue?"),
    )
    payload = guidance.__dict__
    forbidden_keys = {
        "tool_id",
        "workflow_id",
        "authorize",
        "approval",
        "confirm",
        "memory_write",
        "calendar_write",
    }
    assert forbidden_keys.isdisjoint(payload.keys())
    assert guidance.authorizes_privileged_action is False
