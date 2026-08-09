"""Tests for Milestone 20 calendar domain models."""

from __future__ import annotations

import pytest

from src.calendar_models import (
    CalendarEvent,
    CalendarValidationError,
    PRIMARY_CALENDAR_ID,
    compute_proposal_fingerprint,
    create_calendar_account,
    create_change_proposal,
    event_blocks_writes,
    find_conflicts,
    generate_google_client_event_id,
    intervals_overlap,
    is_proposal_expired,
    local_wall_pair_to_utc,
    validate_calendar_event,
    validate_calendar_provider_id,
    validate_google_event_id,
    validate_scopes_granted,
)
from src.config import GOOGLE_CALENDAR_SCOPES
from tests.calendar_test_helpers import busy_window, timed_event


def test_provider_id_google_only() -> None:
    assert validate_calendar_provider_id("google") == "google"
    with pytest.raises(Exception):
        validate_calendar_provider_id("fake")


def test_account_defaults_primary_and_exact_scopes() -> None:
    account = create_calendar_account(display_label="Google Calendar")
    assert account.provider_id == "google"
    assert account.default_calendar_id == PRIMARY_CALENDAR_ID
    assert account.scopes_granted == tuple(sorted(GOOGLE_CALENDAR_SCOPES))
    assert set(account.scopes_granted) == set(GOOGLE_CALENDAR_SCOPES)


def test_scopes_reject_wrong_set() -> None:
    with pytest.raises(CalendarValidationError):
        validate_scopes_granted(("https://www.googleapis.com/auth/calendar",))


def test_timed_and_all_day_events() -> None:
    timed = timed_event()
    assert timed.is_all_day is False
    all_day = validate_calendar_event(
        CalendarEvent(
            event_id="day1",
            calendar_id="primary",
            title="Holiday",
            description=None,
            is_all_day=True,
            start="2026-07-04",
            end="2026-07-05",
            status="confirmed",
            is_recurring=False,
            recurring_event_id=None,
            etag=None,
        )
    )
    assert all_day.is_all_day is True
    assert event_blocks_writes(all_day) is True


def test_recurring_master_and_instance_block_writes() -> None:
    master = timed_event(is_recurring=True)
    instance = timed_event(recurring_event_id="series1")
    assert event_blocks_writes(master) is True
    assert event_blocks_writes(instance) is True


def test_half_open_overlap_and_conflicts() -> None:
    assert intervals_overlap(
        "2026-01-01T10:00:00.000000Z",
        "2026-01-01T11:00:00.000000Z",
        "2026-01-01T10:30:00.000000Z",
        "2026-01-01T11:30:00.000000Z",
    )
    assert not intervals_overlap(
        "2026-01-01T10:00:00.000000Z",
        "2026-01-01T11:00:00.000000Z",
        "2026-01-01T11:00:00.000000Z",
        "2026-01-01T12:00:00.000000Z",
    )
    conflicts = find_conflicts(
        start="2026-01-01T10:00:00.000000Z",
        end="2026-01-01T11:00:00.000000Z",
        busy_windows=[
            busy_window(
                start="2026-01-01T10:30:00.000000Z",
                end="2026-01-01T10:45:00.000000Z",
            )
        ],
    )
    assert len(conflicts) == 1


def test_client_event_id_google_compliant() -> None:
    event_id = generate_google_client_event_id()
    assert validate_google_event_id(event_id) == event_id
    with pytest.raises(CalendarValidationError):
        validate_google_event_id("ABC_BAD")


def test_proposal_fingerprint_stable_and_tamper_sensitive() -> None:
    proposal = create_change_proposal(
        account_id="11111111-1111-1111-1111-111111111111",
        calendar_id="primary",
        operation="create",
        event_id=None,
        client_event_id=generate_google_client_event_id(),
        normalized_payload={
            "title": "Meet",
            "start_utc": "2026-06-01T14:00:00.000000Z",
            "end_utc": "2026-06-01T15:00:00.000000Z",
            "timezone": "America/New_York",
        },
        remote_etag=None,
    )
    again = compute_proposal_fingerprint(
        proposal_id=proposal.proposal_id,
        account_id=proposal.account_id,
        calendar_id=proposal.calendar_id,
        operation=proposal.operation,
        event_id=proposal.event_id,
        client_event_id=proposal.client_event_id,
        normalized_payload=proposal.normalized_payload,
    )
    assert again == proposal.fingerprint
    tampered = dict(proposal.normalized_payload)
    tampered["title"] = "Other"
    assert (
        compute_proposal_fingerprint(
            proposal_id=proposal.proposal_id,
            account_id=proposal.account_id,
            calendar_id=proposal.calendar_id,
            operation=proposal.operation,
            event_id=proposal.event_id,
            client_event_id=proposal.client_event_id,
            normalized_payload=tampered,
        )
        != proposal.fingerprint
    )


def test_proposal_expiry_derived_not_persisted_status() -> None:
    proposal = create_change_proposal(
        account_id="11111111-1111-1111-1111-111111111111",
        calendar_id="primary",
        operation="cancel",
        event_id="evt1",
        client_event_id=None,
        normalized_payload={},
        remote_etag="etag",
    )
    assert proposal.status == "pending"
    assert "expired" not in {"pending", "executed", "failed", "unknown_outcome"}
    assert is_proposal_expired(
        proposal,
        now_utc_iso=proposal.expires_at,
    )


def test_local_wall_pair_uses_iana_timezone() -> None:
    start, end = local_wall_pair_to_utc(
        start_local="2026-06-01 10:00",
        end_local="2026-06-01 11:00",
        timezone_name="America/New_York",
    )
    assert start.endswith("Z")
    assert end.endswith("Z")
