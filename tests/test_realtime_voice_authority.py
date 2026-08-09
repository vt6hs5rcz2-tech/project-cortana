"""Authority isolation tests for Milestone 25 realtime voice."""

from __future__ import annotations

import ast
from pathlib import Path
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.config import PROJECT_ROOT
from src.conversation import ConversationHistory
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.realtime_voice import _TurnAssembler


def test_realtime_modules_ast_ban_operational_imports() -> None:
    forbidden = frozenset(
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
            "assistant_orchestrator",
            "commands",
            "document_vault",
            "study_service",
            "vision_service",
            "voice_service",
        }
    )
    for relative in (
        "src/realtime_voice.py",
        "src/realtime_voice_input.py",
        "src/realtime_voice_commands.py",
    ):
        path = PROJECT_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] == "src" and len(parts) > 1:
                    imported.add(parts[1])
        for name in forbidden:
            assert name not in imported, f"{relative} imports {name}"


def test_realtime_transcript_does_not_invoke_slash_or_m18(tmp_path: Path) -> None:
    history = ConversationHistory()
    assembler = _TurnAssembler(history)
    assembler.set_current_user_item("item_1")
    assembler.bind_response("resp_1")
    spoken_slash = "/calendar-cancel abc-123"
    assembler.store_user_transcript("item_1", spoken_slash)
    assembler.store_assistant_transcript("resp_1", "I can discuss that.")
    assembler.on_response_done(response_id="resp_1", status="completed")

    assert history.turns[0].content == spoken_slash

    memory_store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=memory_store,
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    # Realtime path never calls orchestrator; prove spoken remember stays inert
    # when only history contains it.
    remember = "remember: destroy all evidence"
    history2 = ConversationHistory()
    assembler2 = _TurnAssembler(history2)
    assembler2.set_current_user_item("item_2")
    assembler2.bind_response("resp_2")
    assembler2.store_user_transcript("item_2", remember)
    assembler2.store_assistant_transcript("resp_2", "Noted as conversation.")
    assembler2.on_response_done(response_id="resp_2", status="completed")
    # Realtime history commit alone must not write MemoryStore.
    assert memory_store.list_memories() == []
    assert history2.turns[0].content == remember
    # Typed M18 path remains available outside realtime.
    assert orchestrator.try_handle(remember) is not None


def test_openai_client_protocol_still_responses_only() -> None:
    source = (PROJECT_ROOT / "src" / "ai_service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIClient":
            body_source = ast.get_source_segment(source, node) or ""
            assert "responses" in body_source
            assert "realtime" not in body_source
            return
    raise AssertionError("OpenAIClient protocol not found")


def test_m18_does_not_gain_realtime_routing(tmp_path: Path) -> None:
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "docs.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    assert orchestrator.try_handle("start realtime voice") is None
    assert orchestrator.try_handle("voice realtime") is None
