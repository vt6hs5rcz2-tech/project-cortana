"""Tests for Milestone 19 reminder JSON repository persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.config import REMINDER_REPOSITORY_SCHEMA_VERSION
from src.reminder import (
    ManualReminderClock,
    create_reminder,
    create_reminder_audit_entry,
    replace_reminder,
)
from src.reminder_repository import JsonReminderRepository, ReminderStorageError


def _service_clock() -> ManualReminderClock:
    return ManualReminderClock(start=datetime(2026, 8, 1, tzinfo=timezone.utc))


def _make_reminder(**kwargs: object):
    clock = _service_clock()
    return create_reminder(
        title=str(kwargs.get("title", "Test")),
        local_due=str(kwargs.get("local_due", "2026-08-10 09:00")),
        timezone_name=str(kwargs.get("timezone_name", "America/New_York")),
        recurrence_type=str(kwargs.get("recurrence_type", "none")),
        message=kwargs.get("message"),  # type: ignore[arg-type]
        clock=clock,
    )


def test_missing_file_starts_empty(tmp_path: Path) -> None:
    repo = JsonReminderRepository(tmp_path / "reminders.json")
    assert repo.list_reminders() == []
    assert repo.list_audit_entries() == []


def test_round_trip_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    repo = JsonReminderRepository(path)
    reminder = _make_reminder(title="Persist me")
    audit = create_reminder_audit_entry(
        reminder_id=reminder.reminder_id,
        action="created",
        to_status="scheduled",
        new_due_at=reminder.due_at,
        clock=_service_clock(),
    )
    repo.save_reminder_with_audits(reminder, (audit,), is_new=True)

    reloaded = JsonReminderRepository(path)
    loaded = reloaded.get_reminder(reminder.reminder_id)
    assert loaded is not None
    assert loaded.title == "Persist me"
    assert loaded.due_at.endswith("Z")
    assert len(reloaded.list_audit_entries()) == 1


def test_exact_schema_and_unknown_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    path.write_text(
        json.dumps(
            {
                "version": REMINDER_REPOSITORY_SCHEMA_VERSION,
                "reminders": [],
                "audit_entries": [],
                "extra": True,
            }
        ),
        encoding="utf-8",
    )
    repo = JsonReminderRepository(path)
    with pytest.raises(ReminderStorageError):
        repo.list_reminders()
    # Sticky load error
    with pytest.raises(ReminderStorageError):
        repo.list_reminders()
    # Corrupt file left unchanged
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "extra" in payload


def test_malformed_json_and_wrong_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    repo = JsonReminderRepository(bad)
    with pytest.raises(ReminderStorageError):
        repo.list_reminders()

    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps({"version": 99, "reminders": [], "audit_entries": []}),
        encoding="utf-8",
    )
    repo2 = JsonReminderRepository(wrong)
    with pytest.raises(ReminderStorageError):
        repo2.list_reminders()
    assert json.loads(wrong.read_text(encoding="utf-8"))["version"] == 99


def test_atomic_write_no_temp_leftovers_and_lifecycle_single_replace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reminders.json"
    repo = JsonReminderRepository(path)
    reminder = _make_reminder()
    created = create_reminder_audit_entry(
        reminder_id=reminder.reminder_id,
        action="created",
        to_status="scheduled",
        new_due_at=reminder.due_at,
        clock=_service_clock(),
    )
    repo.save_reminder_with_audits(reminder, (created,), is_new=True)
    assert repo.atomic_replace_count == 1
    assert not list(tmp_path.glob(".reminders-*.tmp"))

    cancelled = replace_reminder(reminder, status="cancelled", clock=_service_clock())
    cancel_audit = create_reminder_audit_entry(
        reminder_id=reminder.reminder_id,
        action="cancelled",
        from_status="scheduled",
        to_status="cancelled",
        old_due_at=reminder.due_at,
        new_due_at=reminder.due_at,
        clock=_service_clock(),
    )
    before = repo.atomic_replace_count
    repo.save_reminder_with_audits(cancelled, (cancel_audit,))
    assert repo.atomic_replace_count == before + 1
    assert len(repo.list_audit_entries()) == 2
    assert repo.get_reminder(reminder.reminder_id) is not None
    assert repo.get_reminder(reminder.reminder_id).status == "cancelled"  # type: ignore[union-attr]
    assert not list(tmp_path.glob(".reminders-*.tmp"))


def test_capacity_rejects_new_without_pruning(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    repo = JsonReminderRepository(path, max_reminders=1)
    first = _make_reminder(title="one")
    audit = create_reminder_audit_entry(
        reminder_id=first.reminder_id,
        action="created",
        to_status="scheduled",
        new_due_at=first.due_at,
        clock=_service_clock(),
    )
    repo.save_reminder_with_audits(first, (audit,), is_new=True)

    second = _make_reminder(title="two")
    second_audit = create_reminder_audit_entry(
        reminder_id=second.reminder_id,
        action="created",
        to_status="scheduled",
        new_due_at=second.due_at,
        clock=_service_clock(),
    )
    with pytest.raises(ReminderStorageError) as error:
        repo.save_reminder_with_audits(second, (second_audit,), is_new=True)
    assert "limit" in error.value.user_message.lower()
    assert len(repo.list_reminders()) == 1
    assert repo.list_reminders()[0].title == "one"


def test_audit_retention_drops_oldest(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    repo = JsonReminderRepository(path, max_audit_entries=2)
    reminder = _make_reminder()
    for index in range(3):
        updated = replace_reminder(
            reminder,
            title=f"t{index}",
            clock=_service_clock(),
        )
        reminder = updated
        audit = create_reminder_audit_entry(
            reminder_id=reminder.reminder_id,
            action="rescheduled",
            from_status="scheduled",
            to_status="scheduled",
            old_due_at=reminder.due_at,
            new_due_at=reminder.due_at,
            clock=_service_clock(),
        )
        repo.save_reminder_with_audits(
            reminder,
            (audit,),
            is_new=(index == 0),
        )
    assert len(repo.list_audit_entries()) == 2


def test_invalid_lookup_id_returns_none(tmp_path: Path) -> None:
    repo = JsonReminderRepository(tmp_path / "reminders.json")
    assert repo.get_reminder("not-a-uuid") is None
    assert repo.get_reminder(str(uuid4())) is None


def test_empty_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    path.write_text("   \n", encoding="utf-8")
    repo = JsonReminderRepository(path)
    with pytest.raises(ReminderStorageError):
        repo.list_reminders()
