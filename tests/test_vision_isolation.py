"""Isolation tests for Milestone 23 visual understanding."""

from __future__ import annotations

import ast
from pathlib import Path

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.config import PROJECT_ROOT
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore


# Operational/domain modules that the vision domain itself must never reach.
VISION_FORBIDDEN_MODULES = frozenset(
    {
        "security_commands",
        "evidence_store",
        "incident_repository",
        "tool_executor",
        "tool_registry",
        "tool_process_runner",
        "workflow_executor",
        "calendar_service",
        "reminder_service",
        "memory_store",
        "study_repository",
        "document_vault",
        "document_ingestion",
        "incident_analysis_service",
        "active_memory",
        "conversation",
    }
)

# Co-located AI helpers live in ai_service.py with already-approved isolated
# functions. Module-level imports from ai_service may transitively name these
# modules even though generate_visual_analysis_response does not use them.
AI_SERVICE_COLOCATION_EXCEPTION = frozenset(
    {
        "document_vault",
        "document_retrieval",
        "document_context",
        "conversation",
        "memory_context",
        "memory",
        "active_memory",
        "incident_analysis_context",
        "incident_analysis_models",
    }
)


def _module_level_src_imports(relative_path: str) -> set[str]:
    """Return direct ``src.<module>`` roots imported at module scope only."""
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


def _reachable_src_modules(
    relative_path: str,
    *,
    stop_at: frozenset[str] | None = None,
) -> set[str]:
    """Follow module-level ``src.*`` imports from one module.

    When ``stop_at`` contains a module name, that module is recorded as
    reachable but its own imports are not expanded. Used to separate the
    vision domain graph from the approved shared ``ai_service`` co-location
    boundary.
    """
    blocked = stop_at or frozenset()
    root = relative_path.removeprefix("src/").removesuffix(".py").replace("\\", "/")
    root = root.replace("/", ".")
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in blocked and current != root:
            continue
        module_path = PROJECT_ROOT / "src" / f"{current}.py"
        if not module_path.is_file():
            continue
        for imported in _module_level_src_imports(f"src/{current}.py"):
            if imported not in seen:
                stack.append(imported)
    return seen


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def _assert_no_forbidden_direct_imports(
    relative_path: str,
    *,
    forbidden: frozenset[str] = VISION_FORBIDDEN_MODULES,
) -> None:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "src" and len(parts) > 1 and parts[1] in forbidden:
                raise AssertionError(f"{relative_path} imports {node.module}")
            if parts[0] in forbidden:
                raise AssertionError(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden


def test_vision_service_ast_bans_operational_imports() -> None:
    _assert_no_forbidden_direct_imports("src/vision_service.py")


def test_vision_commands_ast_bans_operational_imports() -> None:
    _assert_no_forbidden_direct_imports("src/vision_commands.py")
    source = (PROJECT_ROOT / "src" / "vision_commands.py").read_text(encoding="utf-8")
    assert "security_commands" not in source
    assert "DocumentVault" not in source
    assert "StudyPartnerService" not in source
    assert "StudyRepository" not in source
    assert "document_ingestion" not in source
    assert "document_vault" not in source


def test_vision_input_may_reuse_safe_open_only() -> None:
    imports = _module_level_src_imports("src/vision_input.py")
    assert "tool_process_safe_open" in imports
    assert "path_argument_utils" in imports
    assert "document_ingestion" not in imports
    assert "document_vault" not in imports
    assert "security_commands" not in imports
    for forbidden in VISION_FORBIDDEN_MODULES:
        assert forbidden not in imports


def test_vision_input_full_graph_excludes_forbidden_domains() -> None:
    reachable = _reachable_src_modules("src/vision_input.py")
    assert "tool_process_safe_open" in reachable
    assert "path_argument_utils" in reachable
    for forbidden in VISION_FORBIDDEN_MODULES:
        assert forbidden not in reachable, forbidden


def test_vision_commands_full_graph_excludes_vision_domain_forbidden_modules() -> None:
    """Full reachable graph must not include vision-introduced operational deps.

    ``ai_service`` is an approved shared module containing multiple isolated AI
    paths. Imports that exist only because ``ai_service`` co-locates those paths
    are excluded from this assertion via ``stop_at={'ai_service'}``. The
    exception is module co-location only; see
    ``test_generate_visual_analysis_response_has_no_operational_runtime_deps``.
    """
    vision_domain_reachable = _reachable_src_modules(
        "src/vision_commands.py",
        stop_at=frozenset({"ai_service"}),
    )
    assert "ai_service" in vision_domain_reachable
    assert "command_argument_utils" in vision_domain_reachable
    assert "security_commands" not in vision_domain_reachable
    for forbidden in VISION_FORBIDDEN_MODULES:
        assert forbidden not in vision_domain_reachable, forbidden

    # Full expansion may touch ai_service co-located imports, but must still
    # never reach security/evidence/tool/workflow/calendar/reminder/study paths.
    full_reachable = _reachable_src_modules("src/vision_commands.py")
    hard_banned = VISION_FORBIDDEN_MODULES - AI_SERVICE_COLOCATION_EXCEPTION
    for forbidden in hard_banned:
        assert forbidden not in full_reachable, forbidden

    # Co-located modules may appear only through ai_service.
    for colocated in AI_SERVICE_COLOCATION_EXCEPTION & full_reachable:
        assert colocated not in vision_domain_reachable, colocated


def test_generate_visual_analysis_response_has_no_operational_runtime_deps() -> None:
    """ai_service co-location exception does not grant visual-path runtime deps."""
    path = PROJECT_ROOT / "src" / "ai_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = _function_def(tree, "generate_visual_analysis_response")
    source = ast.get_source_segment(path.read_text(encoding="utf-8"), func)
    assert source is not None
    banned_names = (
        "ConversationHistory",
        "ActiveMemoryContext",
        "DocumentVault",
        "StudyRepository",
        "CalendarService",
        "ReminderService",
        "IncidentRepository",
        "EvidenceStore",
        "conversation_history",
        "active_memories",
        "document_results",
        "tool_executor",
        "workflow_executor",
    )
    for name in banned_names:
        assert name not in source

    # Parameters are only client/settings/task_text/image.
    arg_names = [arg.arg for arg in func.args.args]
    kwonly = [arg.arg for arg in func.args.kwonlyargs]
    assert set(arg_names + kwonly) == {"client", "settings", "task_text", "image"}


def test_command_argument_utils_has_no_domain_imports() -> None:
    imports = _module_level_src_imports("src/command_argument_utils.py")
    assert imports == set()
    source = (PROJECT_ROOT / "src" / "command_argument_utils.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom))
    for forbidden in VISION_FORBIDDEN_MODULES:
        assert forbidden not in source


def test_path_argument_utils_has_no_domain_imports() -> None:
    imports = _module_level_src_imports("src/path_argument_utils.py")
    assert imports == set()
    source = (PROJECT_ROOT / "src" / "path_argument_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom))
    for forbidden in VISION_FORBIDDEN_MODULES:
        assert forbidden not in source


def test_vision_modules_not_in_tool_process_safe_paths() -> None:
    process_common = (
        PROJECT_ROOT / "src" / "tool_process_common.py"
    ).read_text(encoding="utf-8")
    assert "vision_service" not in process_common
    assert "vision_commands" not in process_common
    assert "vision_input" not in process_common


def test_no_m18_vision_guidance() -> None:
    source = (
        PROJECT_ROOT / "src" / "assistant_orchestrator.py"
    ).read_text(encoding="utf-8")
    assert "analyze this image" not in source.casefold()
    assert "describe this image" not in source.casefold()
    assert "vision-" not in source
    assert "VisualAnalysisService" not in source
    assert "vision_service" not in source


def test_m18_does_not_gain_vision_routing(tmp_path: Path) -> None:
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=None,
    )
    assert orchestrator.try_handle("analyze this image") is None
    assert orchestrator.try_handle("describe this image") is None


def test_ai_service_keeps_ordinary_content_as_str() -> None:
    conversation_source = (
        PROJECT_ROOT / "src" / "conversation.py"
    ).read_text(encoding="utf-8")
    assert "content: str" in conversation_source
    assert "input_image" not in conversation_source

    ai_source = (PROJECT_ROOT / "src" / "ai_service.py").read_text(encoding="utf-8")
    assert "class VisualApiMessage" in ai_source
    assert "generate_visual_analysis_response" in ai_source
    assert "VISUAL_ANALYSIS_INSTRUCTIONS" in ai_source
