"""Tests for Milestone 20 calendar slash commands."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from google.oauth2.credentials import Credentials

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.calendar_commands import CALENDAR_COMMAND_NAMES, create_default_calendar_service
from src.calendar_models import ManualCalendarClock
from src.commands import handle_slash_command
from src.config import GOOGLE_CALENDAR_SCOPES
from src.conversation import ConversationHistory
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.secret_store import InMemorySecretStore
from src.settings import Settings
from tests.calendar_test_helpers import FakeCalendarProvider, primary_calendar, timed_event

FAKE_CLIENT = cast(OpenAIClient, object())


def _settings(client_file: Path | None = None) -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="test-model",
        google_oauth_client_file=client_file,
    )


def _run(
    message: str,
    *,
    tmp_path: Path,
    calendar_service,
) -> str:
    history = ConversationHistory()
    result = handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=history,
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        calendar_service=calendar_service,
        client=FAKE_CLIENT,
    )
    assert history.turns == []
    return result.message or ""


def _wired_service(
    tmp_path: Path,
    provider: FakeCalendarProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    client_file = tmp_path / "client.json"
    client_file.write_text(
        (
            '{"installed":{"client_id":"cid.apps.googleusercontent.com",'
            '"client_secret":"csecret",'
            '"token_uri":"https://oauth2.googleapis.com/token"}}'
        ),
        encoding="utf-8",
    )
    secrets = InMemorySecretStore()
    service = create_default_calendar_service(
        repository_file_path=tmp_path / "calendar_control.json",
        secret_store=secrets,
        oauth_client_file=client_file,
    )
    service._clock = ManualCalendarClock(  # type: ignore[attr-defined]
        start=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    )
    service._provider_factory = lambda _creds: provider  # type: ignore[method-assign]
    service._oauth_flow_runner = lambda _path: Credentials(  # type: ignore[method-assign]
        token="access",
        refresh_token="refresh-token-value",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csecret",
        scopes=list(GOOGLE_CALENDAR_SCOPES),
    )
    monkeypatch.setattr(
        "src.calendar_service.credentials_from_refresh_token",
        lambda **_kwargs: Credentials(
            token="access",
            refresh_token="refresh-token-value",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csecret",
            scopes=list(GOOGLE_CALENDAR_SCOPES),
        ),
    )
    return service


def test_exact_eleven_commands() -> None:
    assert len(CALENDAR_COMMAND_NAMES) == 11
    assert "calendar-reject" not in CALENDAR_COMMAND_NAMES
    assert "calendar-status" not in CALENDAR_COMMAND_NAMES


def test_create_requires_confirm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar(timezone_name="UTC")])
    service = _wired_service(tmp_path, provider, monkeypatch)
    connect = _run("/calendar-connect", tmp_path=tmp_path, calendar_service=service)
    assert "connected" in connect.lower()
    prepared = _run(
        "/calendar-create - | Meet | 2026-06-01 16:00 | 2026-06-01 17:00",
        tmp_path=tmp_path,
        calendar_service=service,
    )
    assert "prepared" in prepared.lower()
    assert len(provider.create_calls) == 0
    proposal_id = prepared.split("(")[1].split(")")[0]
    confirmed = _run(
        f"/calendar-confirm {proposal_id}",
        tmp_path=tmp_path,
        calendar_service=service,
    )
    assert "executed" in confirmed.lower()
    assert len(provider.create_calls) == 1


def test_fixed_pipe_grammar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(
        calendars=[primary_calendar(timezone_name="UTC")],
        events={
            ("primary", "evt1"): timed_event(
                event_id="evt1",
                start="2026-06-01T16:00:00.000000Z",
                end="2026-06-01T17:00:00.000000Z",
            )
        },
    )
    service = _wired_service(tmp_path, provider, monkeypatch)
    _run("/calendar-connect", tmp_path=tmp_path, calendar_service=service)
    bad = _run(
        "/calendar-event evt1",
        tmp_path=tmp_path,
        calendar_service=service,
    )
    assert "Usage:" in bad
    good = _run(
        "/calendar-event - | evt1",
        tmp_path=tmp_path,
        calendar_service=service,
    )
    assert "Event evt1" in good


def test_status_shows_bounded_calendar_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeCalendarProvider(calendars=[primary_calendar()])
    service = _wired_service(tmp_path, provider, monkeypatch)
    _run("/calendar-connect", tmp_path=tmp_path, calendar_service=service)
    status = _run("/status", tmp_path=tmp_path, calendar_service=service)
    assert "Calendar connection: connected" in status
    assert "Calendar provider: google" in status
    assert "refresh" not in status.lower()
    assert "token" not in status.lower()
