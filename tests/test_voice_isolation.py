"""Isolation tests for Milestone 24 voice transport."""

from __future__ import annotations

import ast
from pathlib import Path

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.config import PROJECT_ROOT
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore

VOICE_SERVICE_FORBIDDEN = frozenset(
    {
        "security_commands",
        "evidence_store",
        "incident_repository",
        "tool_executor",
        "tool_registry",
        "workflow_executor",
        "calendar_service",
        "reminder_service",
        "memory_store",
        "active_memory",
        "conversation",
        "conversation_loop",
        "assistant_orchestrator",
        "commands",
        "document_vault",
        "study_service",
        "vision_service",
    }
)

VOICE_INPUT_FORBIDDEN = frozenset(
    {
        "security_commands",
        "evidence_store",
        "incident_repository",
        "tool_executor",
        "calendar_service",
        "reminder_service",
        "memory_store",
        "active_memory",
        "conversation_loop",
        "assistant_orchestrator",
        "commands",
        "ai_service",
        "voice_service",
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


def test_voice_service_ast_bans_operational_imports() -> None:
    imported = _module_level_src_imports("src/voice_service.py")
    for forbidden in VOICE_SERVICE_FORBIDDEN:
        assert forbidden not in imported, forbidden


def test_voice_input_ast_bans_operational_imports() -> None:
    imported = _module_level_src_imports("src/voice_input.py")
    for forbidden in VOICE_INPUT_FORBIDDEN:
        assert forbidden not in imported, forbidden
    assert "config" in imported


def test_voice_commands_do_not_import_orchestrator_or_slash_at_module_level() -> None:
    imported = _module_level_src_imports("src/voice_commands.py")
    assert "assistant_orchestrator" not in imported
    assert "commands" not in imported
    # conversation_loop is imported lazily inside /voice-turn only.
    assert "conversation_loop" not in imported


def test_voice_modules_do_not_reference_realtime_client() -> None:
    for relative in (
        "src/voice_input.py",
        "src/voice_service.py",
        "src/voice_commands.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "client.realtime" not in source
        assert ".realtime" not in source


def test_openai_client_protocol_remains_responses_only() -> None:
    source = (PROJECT_ROOT / "src" / "ai_service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIClient":
            body_source = ast.get_source_segment(source, node) or ""
            assert "responses" in body_source
            assert "audio" not in body_source
            return
    raise AssertionError("OpenAIClient protocol not found")


def test_no_m18_voice_guidance() -> None:
    source = (
        PROJECT_ROOT / "src" / "assistant_orchestrator.py"
    ).read_text(encoding="utf-8")
    assert "voice-" not in source
    assert "voice_service" not in source
    assert "voice_turn" not in source


def test_m18_does_not_gain_voice_routing(tmp_path: Path) -> None:
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    assert orchestrator.try_handle("start voice mode") is None
    assert orchestrator.try_handle("listen to me") is None
