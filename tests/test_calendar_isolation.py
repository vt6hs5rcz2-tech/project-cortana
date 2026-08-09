"""Isolation tests for Milestone 20 calendar domain."""

from __future__ import annotations

import ast

import pytest

from src.config import PROJECT_ROOT
from src.tool_process_common import build_child_environment
from src.tool_registry import build_default_tool_registry

CALENDAR_SRC_FILES = (
    "secret_store.py",
    "calendar_models.py",
    "calendar_provider.py",
    "calendar_google.py",
    "calendar_repository.py",
    "calendar_service.py",
    "calendar_commands.py",
)

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "ai_service",
        "openai_client",
        "memory_store",
        "incident_repository",
        "evidence_store",
        "tool_executor",
        "workflow_executor",
        "incident_analysis_service",
    }
)


def test_calendar_domain_ast_ban() -> None:
    for filename in CALENDAR_SRC_FILES:
        path = PROJECT_ROOT / "src" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] == "src" and len(parts) > 1:
                    assert parts[1] not in FORBIDDEN_IMPORT_ROOTS, filename
                elif parts[0] in FORBIDDEN_IMPORT_ROOTS:
                    raise AssertionError(f"{filename} imports {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_IMPORT_ROOTS, filename


def test_calendar_not_in_tool_registry() -> None:
    registry = build_default_tool_registry()
    for tool in registry.list_all():
        assert "calendar" not in tool.tool_id.lower()
        assert "google" not in tool.tool_id.lower()


def test_child_env_excludes_oauth_and_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CORTANA_GOOGLE_OAUTH_CLIENT_FILE", r"C:\secret\client.json")
    monkeypatch.setenv("CORTANA_SENTINEL_REFRESH", "refresh-token-value")
    env = build_child_environment()
    joined = " ".join(f"{key}={value}" for key, value in env.items())
    assert "CORTANA_GOOGLE_OAUTH_CLIENT_FILE" not in env
    assert "refresh-token-value" not in joined
    assert "sk-test" not in joined


def test_no_idempotency_key_abstraction() -> None:
    service_source = (
        PROJECT_ROOT / "src" / "calendar_service.py"
    ).read_text(encoding="utf-8")
    models_source = (
        PROJECT_ROOT / "src" / "calendar_models.py"
    ).read_text(encoding="utf-8")
    assert "idempotency_key" not in service_source
    assert "idempotency_key" not in models_source
    assert "client_event_id" in models_source


def test_reminder_and_memory_files_untouched_by_calendar_modules() -> None:
    for filename in CALENDAR_SRC_FILES:
        source = (PROJECT_ROOT / "src" / filename).read_text(encoding="utf-8")
        assert "reminders.json" not in source
        assert "memories.json" not in source
        assert "ReminderRepository" not in source
        assert "MemoryStore" not in source
