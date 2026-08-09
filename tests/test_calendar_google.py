"""Tests for Google Calendar adapter normalization and OAuth helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.calendar_google import (
    GoogleCalendarProvider,
    normalize_google_event,
    validate_google_oauth_client_file,
)
from src.calendar_models import CalendarError
from src.config import GOOGLE_CALENDAR_SCOPES


class _FakeRequest:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {}

    def execute(self) -> dict[str, Any]:
        return self._payload


class _FakeEvents:
    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def list(self, **kwargs: Any) -> _FakeRequest:
        return _FakeRequest({"items": list(self._store.get("events", []))})

    def get(self, **kwargs: Any) -> _FakeRequest:
        event_id = kwargs["eventId"]
        for item in self._store.get("events", []):
            if item.get("id") == event_id:
                return _FakeRequest(item)
        raise CalendarError("missing", category="not_found")

    def insert(self, **kwargs: Any) -> _FakeRequest:
        body = dict(kwargs["body"])
        self._store.setdefault("events", []).append(body)
        return _FakeRequest(body)

    def patch(self, **kwargs: Any) -> _FakeRequest:
        return _FakeRequest(kwargs["body"])

    def delete(self, **kwargs: Any) -> _FakeRequest:
        return _FakeRequest({})


class _FakeFreeBusy:
    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def query(self, **kwargs: Any) -> _FakeRequest:
        return _FakeRequest(self._store.get("freebusy", {"calendars": {}}))


class _FakeCalendarList:
    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def list(self, **kwargs: Any) -> _FakeRequest:
        return _FakeRequest({"items": list(self._store.get("calendars", []))})


class _FakeService:
    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def calendarList(self) -> _FakeCalendarList:
        return _FakeCalendarList(self._store)

    def events(self) -> _FakeEvents:
        return _FakeEvents(self._store)

    def freebusy(self) -> _FakeFreeBusy:
        return _FakeFreeBusy(self._store)


def test_exact_three_scopes() -> None:
    assert GOOGLE_CALENDAR_SCOPES == (
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/calendar.freebusy",
    )
    assert "https://www.googleapis.com/auth/calendar" not in GOOGLE_CALENDAR_SCOPES


def test_oauth_client_file_validation(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(CalendarError):
        validate_google_oauth_client_file(missing)
    bad = tmp_path / "bad.json"
    bad.write_text('{"web":{}}', encoding="utf-8")
    with pytest.raises(CalendarError):
        validate_google_oauth_client_file(bad)
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "cid.apps.googleusercontent.com",
                    "client_secret": "secret",
                }
            }
        ),
        encoding="utf-8",
    )
    assert validate_google_oauth_client_file(good) == good


def test_normalize_timed_all_day_recurring() -> None:
    timed = normalize_google_event(
        {
            "id": "t1",
            "summary": "Meet",
            "status": "confirmed",
            "etag": "e1",
            "start": {"dateTime": "2026-06-01T14:00:00Z"},
            "end": {"dateTime": "2026-06-01T15:00:00Z"},
        },
        calendar_id="primary",
    )
    assert timed.is_all_day is False
    assert timed.start.endswith("Z")

    all_day = normalize_google_event(
        {
            "id": "d1",
            "summary": "Holiday",
            "status": "confirmed",
            "start": {"date": "2026-07-04"},
            "end": {"date": "2026-07-05"},
        },
        calendar_id="primary",
    )
    assert all_day.is_all_day is True
    assert all_day.start == "2026-07-04"

    master = normalize_google_event(
        {
            "id": "r1",
            "summary": "Series",
            "status": "confirmed",
            "recurrence": ["RRULE:FREQ=WEEKLY"],
            "start": {"dateTime": "2026-06-01T14:00:00Z"},
            "end": {"dateTime": "2026-06-01T15:00:00Z"},
        },
        calendar_id="primary",
    )
    assert master.is_recurring is True

    instance = normalize_google_event(
        {
            "id": "r1_20260601",
            "summary": "Series",
            "status": "confirmed",
            "recurringEventId": "r1",
            "start": {"dateTime": "2026-06-01T14:00:00Z"},
            "end": {"dateTime": "2026-06-01T15:00:00Z"},
        },
        calendar_id="primary",
    )
    assert instance.recurring_event_id == "r1"


def test_freebusy_fails_closed_on_calendar_errors() -> None:
    store = {
        "freebusy": {
            "calendars": {
                "primary": {"errors": [{"domain": "global", "reason": "notFound"}]}
            }
        }
    }
    provider = GoogleCalendarProvider(
        credentials=object(),  # type: ignore[arg-type]
        service_factory=lambda _creds: _FakeService(store),
    )
    with pytest.raises(CalendarError):
        provider.get_freebusy(
            ["primary"],
            time_min_utc="2026-06-01T00:00:00.000000Z",
            time_max_utc="2026-06-02T00:00:00.000000Z",
        )


def test_provider_list_calendars_and_create() -> None:
    store: dict[str, Any] = {
        "calendars": [
            {
                "id": "primary",
                "summary": "Primary",
                "timeZone": "UTC",
                "primary": True,
                "accessRole": "owner",
            }
        ],
        "events": [],
    }
    provider = GoogleCalendarProvider(
        credentials=object(),  # type: ignore[arg-type]
        service_factory=lambda _creds: _FakeService(store),
    )
    calendars = provider.list_calendars()
    assert calendars[0].calendar_id == "primary"
    created = provider.create_event(
        "primary",
        event_id="c200123456789abcdefghijklmnopqrstu",
        title="Meet",
        start_utc="2026-06-01T14:00:00.000000Z",
        end_utc="2026-06-01T15:00:00.000000Z",
        timezone_name="UTC",
    )
    assert created.event_id == "c200123456789abcdefghijklmnopqrstu"
