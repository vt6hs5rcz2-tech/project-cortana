"""Tests for Milestone 20 calendar control repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.calendar_models import (
    create_calendar_account,
    create_calendar_audit_entry,
    create_change_proposal,
    generate_google_client_event_id,
    replace_change_proposal,
)
from src.calendar_repository import (
    CalendarStorageError,
    JsonCalendarRepository,
)


def _proposal() -> object:
    return create_change_proposal(
        account_id="11111111-1111-1111-1111-111111111111",
        calendar_id="primary",
        operation="create",
        event_id=None,
        client_event_id=generate_google_client_event_id(),
        normalized_payload={
            "title": "Meet",
            "start_utc": "2026-06-01T14:00:00.000000Z",
            "end_utc": "2026-06-01T15:00:00.000000Z",
            "timezone": "UTC",
        },
        remote_etag=None,
    )


def test_account_proposal_audit_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "calendar_control.json"
    repo = JsonCalendarRepository(path)
    account = create_calendar_account(display_label="Google Calendar")
    audit = create_calendar_audit_entry(
        action="connected",
        account_id=account.account_id,
    )
    repo.save_account_with_audits(account, (audit,))
    proposal = create_change_proposal(
        account_id=account.account_id,
        calendar_id="primary",
        operation="create",
        event_id=None,
        client_event_id=generate_google_client_event_id(),
        normalized_payload={
            "title": "Meet",
            "start_utc": "2026-06-01T14:00:00.000000Z",
            "end_utc": "2026-06-01T15:00:00.000000Z",
            "timezone": "UTC",
        },
        remote_etag=None,
    )
    prepared = create_calendar_audit_entry(
        action="create_prepared",
        account_id=account.account_id,
        proposal_id=proposal.proposal_id,
        operation="create",
    )
    repo.save_proposal_with_audits(proposal, (prepared,), is_new=True)

    reloaded = JsonCalendarRepository(path)
    assert reloaded.get_account() == account
    assert reloaded.get_proposal(proposal.proposal_id) == proposal
    assert len(reloaded.list_audit_entries()) == 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "refresh_token" not in serialized
    assert "access_token" not in serialized
    assert "token" not in serialized.lower().replace("client_event_id", "")


def test_atomic_proposal_and_audit_update(tmp_path: Path) -> None:
    path = tmp_path / "calendar_control.json"
    repo = JsonCalendarRepository(path)
    proposal = create_change_proposal(
        account_id="11111111-1111-1111-1111-111111111111",
        calendar_id="primary",
        operation="cancel",
        event_id="evt1",
        client_event_id=None,
        normalized_payload={},
        remote_etag="etag",
    )
    repo.save_proposal_with_audits(
        proposal,
        (
            create_calendar_audit_entry(
                action="cancel_prepared",
                proposal_id=proposal.proposal_id,
                operation="cancel",
            ),
        ),
        is_new=True,
    )
    before = repo.atomic_replace_count
    executed = replace_change_proposal(proposal, status="executed")
    repo.save_proposal_with_audits(
        executed,
        (
            create_calendar_audit_entry(
                action="cancelled",
                proposal_id=proposal.proposal_id,
                operation="cancel",
                from_status="pending",
                to_status="executed",
            ),
        ),
    )
    assert repo.atomic_replace_count == before + 1
    assert repo.get_proposal(proposal.proposal_id) is not None
    assert repo.get_proposal(proposal.proposal_id).status == "executed"  # type: ignore[union-attr]


def test_forbidden_token_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "calendar_control.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "account": None,
                "proposals": [],
                "audit_entries": [],
                "refresh_token": "nope",
            }
        ),
        encoding="utf-8",
    )
    repo = JsonCalendarRepository(path)
    with pytest.raises(CalendarStorageError):
        repo.get_account()


def test_sticky_load_error(tmp_path: Path) -> None:
    path = tmp_path / "calendar_control.json"
    path.write_text("{bad", encoding="utf-8")
    repo = JsonCalendarRepository(path)
    with pytest.raises(CalendarStorageError):
        repo.get_account()
    with pytest.raises(CalendarStorageError):
        repo.list_proposals()
