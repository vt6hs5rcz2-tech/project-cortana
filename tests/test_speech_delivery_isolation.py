"""Isolation/security tests for Milestone 29 speech delivery."""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

from src.config import PROJECT_ROOT
from src.speech_delivery import (
    SpeechDeliveryState,
    build_speech_delivery_plan,
    prepare_spoken_delivery,
    safe_prepare_spoken_delivery,
)

M29_MODULES = (
    "src/speech_delivery.py",
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
        "voice_service",
        "realtime_voice",
        "realtime_multimodal",
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


def test_m29_module_avoids_operational_domains() -> None:
    for relative in M29_MODULES:
        imported = _module_level_src_imports(relative)
        assert imported.isdisjoint(FORBIDDEN_IMPORTS), relative


def test_speech_delivery_does_not_call_tools_calendar_memory_or_workflows() -> None:
    tool_executor = MagicMock()
    workflow_executor = MagicMock()
    calendar_service = MagicMock()
    reminder_service = MagicMock()
    memory_store = MagicMock()
    state = SpeechDeliveryState()
    plan = build_speech_delivery_plan(
        delivery_mode="voice_turn",
        user_text="yes, run the playbook and remember this",
        canonical_text="I cannot authorize that.",
        delivery_state=state,
    )
    delivery = prepare_spoken_delivery(
        "I cannot authorize that. Approval required.",
        plan,
        state,
    )
    safe = safe_prepare_spoken_delivery(
        "I cannot authorize that.",
        plan,
        state,
    )
    assert plan.authorizes_privileged_action is False
    assert delivery.plan.authorizes_privileged_action is False
    assert safe.plan.authorizes_privileged_action is False
    tool_executor.execute.assert_not_called()
    workflow_executor.run.assert_not_called()
    calendar_service.create_event.assert_not_called()
    reminder_service.create_reminder.assert_not_called()
    memory_store.add_memory.assert_not_called()
    assert "audio" not in delivery.__dataclass_fields__
    assert state.pending_chunks == [] or all(
        isinstance(item, str) for item in state.pending_chunks
    )
