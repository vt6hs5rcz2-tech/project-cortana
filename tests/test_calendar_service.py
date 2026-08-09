"""Tests for Milestone 20 calendar service prepare/confirm flows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from google.oauth2.credentials import Credentials

from src.calendar_models import CalendarChangeProposal, CalendarError, ManualCalendarClock
from src.calendar_repository import JsonCalendarRepository
from src.calendar_service import CalendarService
from src.config import GOOGLE_CALENDAR_SCOPES
from src.secret_store import (
    InMemorySecretStore,
    SecretStoreError,
    google_refresh_token_secret_key,
)
from tests.calendar_test_helpers import (
    FakeCalendarProvider,
    busy_window,
    primary_calendar,
    timed_event,
)


def _client_file(tmp_path: Path) -> Path:
    path = tmp_path / "client.json"
    path.write_text(
        (
            '{"installed":{"client_id":"cid.apps.googleusercontent.com",'
            '"client_secret":"csecret",'
            '"token_uri":"https://oauth2.googleapis.com/token"}}'
        ),
        encoding="utf-8",
    )
    return path


def _service(
    tmp_path: Path,
    provider: FakeCalendarProvider,
    monkeypatch: pytest.MonkeyPatch,
    *,
    clock: ManualCalendarClock | None = None,
) -> tuple[CalendarService, InMemorySecretStore, JsonCalendarRepository]:
    repo = JsonCalendarRepository(tmp_path / "calendar_control.json")
    secrets = InMemorySecretStore()
    client_file = _client_file(tmp_path)
    active_clock = clock or ManualCalendarClock(
        start=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    )

    def oauth_runner(_path: Path) -> Credentials:
        return Credentials(
            token="access",
            refresh_token="refresh-token-value",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csecret",
            scopes=list(GOOGLE_CALENDAR_SCOPES),
        )

    def fake_credentials_from_refresh_token(
        *,
        client_secrets_file: Path,
        refresh_token: str,
    ) -> Credentials:
        assert client_secrets_file.exists()
        assert refresh_token
        return Credentials(
            token="access",
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csecret",
            scopes=list(GOOGLE_CALENDAR_SCOPES),
        )

    monkeypatch.setattr(
        "src.calendar_service.credentials_from_refresh_token",
        fake_credentials_from_refresh_token,
    )

    service = CalendarService(
        repo,
        secrets,
        oauth_client_file=client_file,
        clock=active_clock,
        provider_factory=lambda _creds: provider,
        oauth_flow_runner=oauth_runner,
    )
    return service, secrets, repo


def _connect(service: CalendarService, secrets: InMemorySecretStore) -> None:
    account = service.connect()
    assert secrets.get_secret(google_refresh_token_secret_key(account.account_id))


def _repo_pending_count(tmp_path: Path) -> int:
    repo = JsonCalendarRepository(tmp_path / "calendar_control.json")
    return sum(1 for item in repo.list_proposals() if item.status == "pending")


def _repo_get(tmp_path: Path, proposal_id: str) -> CalendarChangeProposal:
    proposal = JsonCalendarRepository(
        tmp_path / "calendar_control.json"
    ).get_proposal(proposal_id)
    assert proposal is not None
    return proposal


def test_connect_stores_only_refresh_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar()])
    service, secrets, repo = _service(tmp_path, provider, monkeypatch)
    account = service.connect()
    key = google_refresh_token_secret_key(account.account_id)
    assert secrets.get_secret(key) == "refresh-token-value"
    payload = (tmp_path / "calendar_control.json").read_text(encoding="utf-8")
    assert "refresh-token-value" not in payload
    assert "access" not in payload
    stored = repo.get_account()
    assert stored is not None
    assert stored.scopes_granted == tuple(sorted(GOOGLE_CALENDAR_SCOPES))


def test_single_account_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar()])
    service, _secrets, _repo = _service(tmp_path, provider, monkeypatch)
    service.connect()
    with pytest.raises(CalendarError):
        service.connect()


def test_disconnect_removes_secret_even_if_revoke_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar()])
    service, secrets, repo = _service(tmp_path, provider, monkeypatch)
    account = service.connect()
    key = google_refresh_token_secret_key(account.account_id)

    def boom(_token: str) -> None:
        raise RuntimeError("revoke failed")

    monkeypatch.setattr(
        "src.calendar_service.revoke_google_refresh_token",
        boom,
    )
    service.disconnect()
    assert secrets.get_secret(key) is None
    assert repo.get_account() is None


def test_disconnect_fails_closed_when_secret_delete_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar()])
    service, secrets, repo = _service(tmp_path, provider, monkeypatch)
    account = service.connect()
    key = google_refresh_token_secret_key(account.account_id)
    assert secrets.get_secret(key) == "refresh-token-value"
    audits_before = len(repo.list_audit_entries())

    def boom(_key: str) -> None:
        raise SecretStoreError("SecretStore delete failed: PasswordDeleteError")

    monkeypatch.setattr(secrets, "delete_secret", boom)

    with pytest.raises(CalendarError) as error:
        service.disconnect()

    assert error.value.category == "secret_store_error"
    stored = repo.get_account()
    assert stored is not None
    assert stored.account_id == account.account_id
    assert secrets.get_secret(key) == "refresh-token-value"
    audits = repo.list_audit_entries()
    assert len(audits) == audits_before
    assert all(entry.action != "disconnected" for entry in audits)


def test_conflict_blocks_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(
        calendars=[primary_calendar(timezone_name="UTC")],
        busy=[
            busy_window(
                start="2026-06-01T14:00:00.000000Z",
                end="2026-06-01T15:00:00.000000Z",
            )
        ],
    )
    service, secrets, _repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    with pytest.raises(CalendarError) as error:
        service.prepare_create(
            title="Meet",
            start_local="2026-06-01 14:00",
            end_local="2026-06-01 15:00",
        )
    assert error.value.category == "conflict"
    assert _repo_pending_count(tmp_path) == 0


def test_prepare_confirm_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar(timezone_name="UTC")])
    service, secrets, repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    proposal = service.prepare_create(
        title="Meet",
        start_local="2026-06-01 16:00",
        end_local="2026-06-01 17:00",
    )
    assert proposal.client_event_id is not None
    client_id = proposal.client_event_id
    assert len(provider.create_calls) == 0
    executed = service.confirm(proposal.proposal_id)
    assert executed.status == "executed"
    assert len(provider.create_calls) == 1
    assert provider.create_calls[0]["event_id"] == client_id
    with pytest.raises(CalendarError):
        service.confirm(proposal.proposal_id)
    stored = repo.get_proposal(proposal.proposal_id)
    assert stored is not None
    assert stored.status == "executed"


def test_unknown_outcome_create_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar(timezone_name="UTC")])
    provider.network_fail_create = True
    service, secrets, _repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    proposal = service.prepare_create(
        title="Meet",
        start_local="2026-06-01 16:00",
        end_local="2026-06-01 17:00",
    )
    with pytest.raises(CalendarError) as first:
        service.confirm(proposal.proposal_id)
    assert "uncertain" in first.value.user_message.lower()
    stored = _repo_get(tmp_path, proposal.proposal_id)
    assert stored.status == "unknown_outcome"
    assert proposal.client_event_id is not None
    provider.events[("primary", proposal.client_event_id)] = timed_event(
        event_id=proposal.client_event_id,
        title="Meet",
        start=proposal.normalized_payload["start_utc"],
        end=proposal.normalized_payload["end_utc"],
    )
    executed = service.confirm(proposal.proposal_id)
    assert executed.status == "executed"


def test_reschedule_stale_etag_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = timed_event(
        start="2026-06-01T16:00:00.000000Z",
        end="2026-06-01T17:00:00.000000Z",
        etag="etag-old",
    )
    provider = FakeCalendarProvider(
        calendars=[primary_calendar(timezone_name="UTC")],
        events={("primary", event.event_id): event},
    )
    service, secrets, _repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    proposal = service.prepare_reschedule(
        event_id=event.event_id,
        start_local="2026-06-01 18:00",
        end_local="2026-06-01 19:00",
    )
    provider.events[("primary", event.event_id)] = timed_event(
        event_id=event.event_id,
        start=event.start,
        end=event.end,
        etag="etag-new",
        title=event.title,
    )
    with pytest.raises(CalendarError) as error:
        service.confirm(proposal.proposal_id)
    assert error.value.category == "conflict"
    assert _repo_get(tmp_path, proposal.proposal_id).status == "failed"


def test_recurring_write_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = timed_event(is_recurring=True)
    provider = FakeCalendarProvider(
        calendars=[primary_calendar(timezone_name="UTC")],
        events={("primary", event.event_id): event},
    )
    service, secrets, _repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    with pytest.raises(CalendarError):
        service.prepare_reschedule(
            event_id=event.event_id,
            start_local="2026-06-01 18:00",
            end_local="2026-06-01 19:00",
        )
    with pytest.raises(CalendarError):
        service.prepare_cancel(event_id=event.event_id)


def test_cancel_already_deleted_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = timed_event()
    provider = FakeCalendarProvider(
        calendars=[primary_calendar(timezone_name="UTC")],
        events={("primary", event.event_id): event},
    )
    service, secrets, _repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    proposal = service.prepare_cancel(event_id=event.event_id)
    del provider.events[("primary", event.event_id)]
    executed = service.confirm(proposal.proposal_id)
    assert executed.status == "executed"


def test_expired_pending_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualCalendarClock(
        start=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    )
    provider = FakeCalendarProvider(calendars=[primary_calendar(timezone_name="UTC")])
    service, secrets, _repo = _service(
        tmp_path, provider, monkeypatch, clock=clock
    )
    _connect(service, secrets)
    proposal = service.prepare_create(
        title="Meet",
        start_local="2026-06-01 16:00",
        end_local="2026-06-01 17:00",
    )
    clock.advance(16 * 60)
    with pytest.raises(CalendarError, match="expired"):
        service.confirm(proposal.proposal_id)


def test_conflict_at_confirm_fails_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar(timezone_name="UTC")])
    service, secrets, _repo = _service(tmp_path, provider, monkeypatch)
    _connect(service, secrets)
    proposal = service.prepare_create(
        title="Meet",
        start_local="2026-06-01 16:00",
        end_local="2026-06-01 17:00",
    )
    provider.busy.append(
        busy_window(
            start=proposal.normalized_payload["start_utc"],
            end=proposal.normalized_payload["end_utc"],
        )
    )
    with pytest.raises(CalendarError) as error:
        service.confirm(proposal.proposal_id)
    assert error.value.category == "conflict"
    assert _repo_get(tmp_path, proposal.proposal_id).status == "failed"


def test_needs_reauth_when_secret_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar()])
    service, secrets, repo = _service(tmp_path, provider, monkeypatch)
    account = service.connect()
    secrets.delete_secret(google_refresh_token_secret_key(account.account_id))
    with pytest.raises(CalendarError):
        service.list_calendars()
    stored = repo.get_account()
    assert stored is not None
    assert stored.status == "needs_reauth"
