"""Isolation and answer-key security tests for Milestone 22 Study Partner."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.config import PROJECT_ROOT
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.study_models import StudyQuestionPublicView


FORBIDDEN_MODULES = {
    "tool_executor",
    "workflow_executor",
    "calendar_service",
    "calendar_google",
    "reminder_service",
    "memory_store",
    "active_memory",
    "incident_repository",
    "evidence_store",
    "security_commands",
    "incident_analysis_service",
}


def _assert_no_forbidden_imports(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "src" and len(parts) > 1:
                assert parts[1] not in FORBIDDEN_MODULES
            elif parts[0] in FORBIDDEN_MODULES:
                raise AssertionError(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in FORBIDDEN_MODULES


def test_study_service_ast_bans_operational_imports() -> None:
    _assert_no_forbidden_imports("src/study_service.py")


def test_study_commands_ast_bans_operational_imports() -> None:
    # study_commands may import OpenAIClient/settings/vault/knowledge only.
    # Argument parsing uses dependency-free command_argument_utils.
    path = PROJECT_ROOT / "src" / "study_commands.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_command_utils = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "src" and len(parts) > 1 and parts[1] in FORBIDDEN_MODULES:
                raise AssertionError(node.module)
            if parts[0] == "src" and len(parts) > 1 and parts[1] == "command_argument_utils":
                imported_command_utils = True
            if parts[0] == "src" and len(parts) > 1 and parts[1] == "security_commands":
                raise AssertionError("study_commands must not import security_commands")
    assert imported_command_utils


def test_public_view_has_no_answer_key_fields() -> None:
    names = {item.name for item in fields(StudyQuestionPublicView)}
    assert "correct_answer" not in names
    assert "explanation" not in names


def test_study_question_and_status_formatters_cannot_access_answer_key() -> None:
    service_path = PROJECT_ROOT / "src" / "study_service.py"
    tree = ast.parse(service_path.read_text(encoding="utf-8"))

    def function_uses_attr(func_name: str, attr: str) -> bool:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute) and child.attr == attr:
                        return True
        return False

    assert not function_uses_attr("format_study_question_message", "correct_answer")
    assert not function_uses_attr("format_study_question_message", "explanation")
    assert not function_uses_attr("format_study_status_message", "correct_answer")
    assert not function_uses_attr("format_study_status_message", "explanation")


def test_study_question_handler_types_public_view_only() -> None:
    commands_path = PROJECT_ROOT / "src" / "study_commands.py"
    tree = ast.parse(commands_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_study_question":
            source = ast.get_source_segment(
                commands_path.read_text(encoding="utf-8"), node
            )
            assert source is not None
            assert "StudyQuestionPublicView" in source or "public_view" in source
            assert "get_question(" not in source


def test_study_modules_not_in_tool_process_safe_paths() -> None:
    process_common = (
        PROJECT_ROOT / "src" / "tool_process_common.py"
    ).read_text(encoding="utf-8")
    assert "study_service" not in process_common
    assert "study_commands" not in process_common
    assert "study_repository" not in process_common


def test_m18_study_guidance_has_no_service_dependency(tmp_path: Path) -> None:
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    for phrase in ("help me study", "quiz me", "HELP ME STUDY", "Quiz Me"):
        result = orchestrator.try_handle(phrase)
        assert result is not None
        assert result.domain == "guidance"
        assert result.action == "study_partner"
        assert "/study-start" in result.safe_user_message
    assert orchestrator.try_handle("help me study please") is None
