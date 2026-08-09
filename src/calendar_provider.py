"""Calendar provider protocol for Milestone 20.

OAuth/account lifecycle is owned by CalendarService, not the provider.
"""

from __future__ import annotations

from typing import Protocol

from src.calendar_models import CalendarEvent, CalendarInfo, FreeBusyWindow


class CalendarProvider(Protocol):
    """Provider boundary for authorized calendar reads and mutations."""

    def list_calendars(self) -> list[CalendarInfo]:
        """Return calendars available to the authorized account."""

    def list_events(
        self,
        calendar_id: str,
        *,
        time_min_utc: str,
        time_max_utc: str,
        limit: int,
    ) -> list[CalendarEvent]:
        """Return upcoming events in the half-open UTC window."""

    def get_event(self, calendar_id: str, event_id: str) -> CalendarEvent:
        """Return one event by ID."""

    def get_freebusy(
        self,
        calendar_ids: list[str],
        *,
        time_min_utc: str,
        time_max_utc: str,
    ) -> list[FreeBusyWindow]:
        """Return busy windows for the given calendars and window."""

    def create_event(
        self,
        calendar_id: str,
        *,
        event_id: str,
        title: str,
        start_utc: str,
        end_utc: str,
        timezone_name: str,
    ) -> CalendarEvent:
        """Create one timed non-recurring event with a client-supplied ID."""

    def update_event(
        self,
        calendar_id: str,
        event_id: str,
        *,
        start_utc: str,
        end_utc: str,
        timezone_name: str,
        etag: str | None,
    ) -> CalendarEvent:
        """Reschedule one timed non-recurring event."""

    def delete_event(
        self,
        calendar_id: str,
        event_id: str,
        *,
        etag: str | None,
    ) -> None:
        """Delete one event."""
