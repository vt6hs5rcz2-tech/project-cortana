"""Isolation/security tests for Milestone 28 realtime conversational planning."""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

from src.config import PROJECT_ROOT
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.realtime_conversation_plan import (
    plan_realtime_turn,
    safe_plan_realtime_turn,
)

M28_MODULES = (
    "src/realtime_conversation_plan.py",
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


def test_m28_modules_avoid_operational_domains() -> None:
    for relative in M28_MODULES:
        imported = _module_level_src_imports(relative)
        assert imported.isdisjoint(FORBIDDEN_IMPORTS), relative


def test_planning_does_not_call_tools_calendar_memory_or_workflows() -> None:
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
        "do that",
        "what is that?",
    ):
        plan = plan_realtime_turn(
            intel,
            text,
            state,
            interaction_mode="realtime",
        )
        assert plan.authorizes_privileged_action is False

    tool_executor.assert_not_called()
    workflow_executor.assert_not_called()
    calendar_service.assert_not_called()
    reminder_service.assert_not_called()
    memory_store.assert_not_called()


def test_failed_planning_grants_no_authority() -> None:
    class Broken(ConversationIntelligence):
        def interpret(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    result = safe_plan_realtime_turn(
        Broken(),
        "yes",
        ConversationState(),
        interaction_mode="realtime",
    )
    assert result is None
