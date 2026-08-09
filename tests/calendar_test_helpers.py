"""Test-only fake calendar provider and helpers for Milestone 20."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.calendar_models import (
    CalendarError,
    CalendarEvent,
    CalendarInfo,
    FreeBusyWindow,
    find_conflicts,
    validate_calendar_event,
    validate_calendar_info,
    validate_freebusy_window,
)


@dataclass
class FakeCalendarProvider:
    """In-memory CalendarProvider used only by tests."""

    calendars: list[CalendarInfo] = field(default_factory=list)
    events: dict[tuple[str, str], CalendarEvent] = field(default_factory=dict)
    busy: list[FreeBusyWindow] = field(default_factory=list)
    freebusy_errors: set[str] = field(default_factory=set)
    create_calls: list[dict[str, str]] = field(default_factory=list)
    update_calls: list[dict[str, str | None]] = field(default_factory=list)
    delete_calls: list[dict[str, str | None]] = field(default_factory=list)
    fail_next_create_with: str | None = None
    fail_next_update_with: str | None = None
    fail_next_delete_with: str | None = None
    network_fail_create: bool = False
    network_fail_update: bool = False
    network_fail_delete: bool = False

    def list_calendars(self) -> list[CalendarInfo]:
        return list(self.calendars)

    def list_events(
        self,
        calendar_id: str,
        *,
        time_min_utc: str,
        time_max_utc: str,
        limit: int,
    ) -> list[CalendarEvent]:
        items = [
            event
            for (cal_id, _), event in self.events.items()
            if cal_id == calendar_id
            and not event.is_all_day
            and event.start >= time_min_utc
            and event.start < time_max_utc
        ]
        items.sort(key=lambda item: (item.start, item.event_id))
        return items[:limit]

    def get_event(self, calendar_id: str, event_id: str) -> CalendarEvent:
        event = self.events.get((calendar_id, event_id))
        if event is None:
            raise CalendarError("missing", category="not_found")
        return event

    def get_freebusy(
        self,
        calendar_ids: list[str],
        *,
        time_min_utc: str,
        time_max_utc: str,
    ) -> list[FreeBusyWindow]:
        for calendar_id in calendar_ids:
            if calendar_id in self.freebusy_errors:
                raise CalendarError(
                    "freebusy access error",
                    category="auth_error",
                    user_message=(
                        "Cortana: Free/busy could not be verified for one or more "
                        "calendars."
                    ),
                )
        return [
            window
            for window in self.busy
            if window.calendar_id in calendar_ids
            and find_conflicts(
                start=time_min_utc,
                end=time_max_utc,
                busy_windows=[window],
            )
        ]

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
        self.create_calls.append(
            {
                "calendar_id": calendar_id,
                "event_id": event_id,
                "title": title,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "timezone_name": timezone_name,
            }
        )
        if self.network_fail_create:
            self.network_fail_create = False
            raise CalendarError("network", category="network_error")
        if self.fail_next_create_with == "conflict":
            self.fail_next_create_with = None
            raise CalendarError("conflict", category="conflict")
        if (calendar_id, event_id) in self.events:
            raise CalendarError("duplicate", category="conflict")
        event = validate_calendar_event(
            CalendarEvent(
                event_id=event_id,
                calendar_id=calendar_id,
                title=title,
                description=None,
                is_all_day=False,
                start=start_utc,
                end=end_utc,
                status="confirmed",
                is_recurring=False,
                recurring_event_id=None,
                etag="etag-created",
            )
        )
        self.events[(calendar_id, event_id)] = event
        return event

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
        self.update_calls.append(
            {
                "calendar_id": calendar_id,
                "event_id": event_id,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "timezone_name": timezone_name,
                "etag": etag,
            }
        )
        if self.network_fail_update:
            self.network_fail_update = False
            raise CalendarError("network", category="network_error")
        current = self.get_event(calendar_id, event_id)
        if etag and current.etag and etag != current.etag:
            raise CalendarError("stale", category="conflict")
        updated = validate_calendar_event(
            CalendarEvent(
                event_id=current.event_id,
                calendar_id=current.calendar_id,
                title=current.title,
                description=current.description,
                is_all_day=False,
                start=start_utc,
                end=end_utc,
                status=current.status,
                is_recurring=current.is_recurring,
                recurring_event_id=current.recurring_event_id,
                etag="etag-updated",
            )
        )
        self.events[(calendar_id, event_id)] = updated
        return updated

    def delete_event(
        self,
        calendar_id: str,
        event_id: str,
        *,
        etag: str | None,
    ) -> None:
        self.delete_calls.append(
            {
                "calendar_id": calendar_id,
                "event_id": event_id,
                "etag": etag,
            }
        )
        if self.network_fail_delete:
            self.network_fail_delete = False
            raise CalendarError("network", category="network_error")
        current = self.events.get((calendar_id, event_id))
        if current is None:
            raise CalendarError("missing", category="not_found")
        if etag and current.etag and etag != current.etag:
            raise CalendarError("stale", category="conflict")
        del self.events[(calendar_id, event_id)]


def primary_calendar(*, timezone_name: str = "America/New_York") -> CalendarInfo:
    return validate_calendar_info(
        CalendarInfo(
            calendar_id="primary",
            summary="Primary",
            timezone=timezone_name,
            primary=True,
            access_role="owner",
        )
    )


def timed_event(
    *,
    event_id: str = "evt1",
    calendar_id: str = "primary",
    title: str = "Standup",
    start: str = "2026-06-01T14:00:00.000000Z",
    end: str = "2026-06-01T15:00:00.000000Z",
    etag: str | None = "etag-1",
    is_recurring: bool = False,
    recurring_event_id: str | None = None,
) -> CalendarEvent:
    return validate_calendar_event(
        CalendarEvent(
            event_id=event_id,
            calendar_id=calendar_id,
            title=title,
            description=None,
            is_all_day=False,
            start=start,
            end=end,
            status="confirmed",
            is_recurring=is_recurring,
            recurring_event_id=recurring_event_id,
            etag=etag,
        )
    )


def busy_window(
    *,
    start: str,
    end: str,
    calendar_id: str = "primary",
) -> FreeBusyWindow:
    return validate_freebusy_window(
        FreeBusyWindow(calendar_id=calendar_id, start=start, end=end)
    )
