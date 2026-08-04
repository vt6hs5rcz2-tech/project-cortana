"""Tests for authorized scopes and tool-control persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.tool_common import ToolAuthorizationError, ToolValidationError
from src.tool_repository import JsonToolControlRepository, ToolStorageError
from src.tool_request import create_tool_execution_request
from src.tool_scope import (
    assert_path_authorized,
    assert_tool_authorized,
    canonicalize_path_root,
    create_authorized_scope,
    disable_authorized_scope,
)
from tests.tool_helpers import make_scope, tool_repository


def test_scope_creation_and_path_root_canonicalization(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    scope = make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    assert scope.active is True
    assert len(scope.allowed_local_path_roots) == 1
    assert Path(scope.allowed_local_path_roots[0]) == root.resolve()
    assert canonicalize_path_root(str(root)) == str(root.resolve())


def test_scope_rejects_traversal_and_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("x", encoding="utf-8")
    scope = make_scope(tmp_path, tool_ids=["file-sha256"], root=root)

    with pytest.raises(ToolAuthorizationError):
        assert_path_authorized(scope, str(target))

    nested = root / "nested"
    nested.mkdir()
    escape = nested / ".." / ".." / "outside" / "secret.txt"
    with pytest.raises(ToolAuthorizationError):
        assert_path_authorized(scope, str(escape))


def test_scope_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_file = outside / "real.txt"
    real_file.write_text("data", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(real_file)
    except OSError:
        pytest.skip("Windows denied symlink creation")

    scope = make_scope(tmp_path, tool_ids=["file-sha256"], root=root)
    with pytest.raises(ToolAuthorizationError):
        assert_path_authorized(scope, str(link))


def test_scope_expiry_and_disable(tmp_path: Path) -> None:
    expired_at = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat().replace("+00:00", "Z")
    scope = create_authorized_scope(
        scope_name="Expired",
        allowed_tool_ids=["system-summary"],
        allowed_target_types=["system-summary", "none"],
        expires_at=expired_at,
    )
    with pytest.raises(ToolAuthorizationError):
        assert_tool_authorized(scope, "system-summary")

    active = make_scope(tmp_path, tool_ids=["system-summary"])
    disabled = disable_authorized_scope(active)
    with pytest.raises(ToolAuthorizationError):
        assert_tool_authorized(disabled, "system-summary")


def test_scope_rejects_prohibited_tools() -> None:
    with pytest.raises(ToolAuthorizationError):
        create_authorized_scope(
            scope_name="Bad",
            allowed_tool_ids=["evil-tool"],
            allowed_target_types=["none"],
            prohibited_tool_ids=frozenset({"evil-tool"}),
        )


def test_scope_persistence_and_reload(tmp_path: Path) -> None:
    repo = tool_repository(tmp_path)
    scope = make_scope(tmp_path, tool_ids=["system-summary"])
    saved = repo.add_scope(scope)

    reloaded = JsonToolControlRepository(repo.file_path)
    loaded = reloaded.get_scope(saved.scope_id)
    assert loaded is not None
    assert loaded.scope_id == saved.scope_id
    assert loaded.allowed_tool_ids == ("system-summary",)


def test_repository_corruption_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "tool_control.json"
    path.write_text("{not-json", encoding="utf-8")
    repo = JsonToolControlRepository(path)
    with pytest.raises(ToolStorageError):
        repo.list_scopes()
    assert path.read_text(encoding="utf-8") == "{not-json"


def test_request_requires_existing_scope(tmp_path: Path) -> None:
    repo = tool_repository(tmp_path)
    request = create_tool_execution_request(
        tool_id="system-summary",
        normalized_parameters={},
        scope_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        justification="summary",
    )
    with pytest.raises(ToolStorageError):
        repo.add_request(request)


def test_audit_is_append_only_and_ordered(tmp_path: Path) -> None:
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="summary",
        )
    )
    entries = repo.list_audit_entries()
    assert [entry.action for entry in entries] == [
        "scope-created",
        "request-created",
    ]
    assert entries[0].timestamp <= entries[1].timestamp
    assert request.request_id == entries[1].request_id


def test_scope_notes_are_not_required_in_list_output(tmp_path: Path) -> None:
    scope = make_scope(tmp_path, tool_ids=["system-summary"])
    assert scope.notes is not None
    # Notes exist on the record but must never be logged by commands/tests here.
    assert "test-scope-notes-marker" in scope.notes
