"""Google Calendar provider and OAuth helpers for Milestone 20.

Cortana delegates Google endpoint selection and TLS validation to the official
Google SDKs. This module does not expose arbitrary user-supplied URLs and is
not registered with the defensive tool executor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

from src.calendar_models import (
    CalendarError,
    CalendarEvent,
    CalendarInfo,
    CalendarValidationError,
    FreeBusyWindow,
    validate_calendar_event,
    validate_calendar_info,
    validate_freebusy_window,
    validate_google_event_id,
)
from src.config import GOOGLE_CALENDAR_SCOPES
from src.reminder import validate_iana_timezone
from src.tool_common import validate_utc_timestamp

logger = logging.getLogger("ProjectCortana")

GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URI = "https://oauth2.googleapis.com/revoke"


class GoogleCalendarProvider:
    """CalendarProvider implementation using the official Google Calendar API."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        service_factory: Callable[[Credentials], Any] | None = None,
    ) -> None:
        self._credentials = credentials
        if service_factory is None:
            self._service = build(
                "calendar",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
        else:
            self._service = service_factory(credentials)

    def list_calendars(self) -> list[CalendarInfo]:
        try:
            response = (
                self._service.calendarList()
                .list(maxResults=250, showHidden=False)
                .execute()
            )
        except Exception as error:
            raise _map_provider_error(error, operation="list_calendars") from error
        items = response.get("items") or []
        results: list[CalendarInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            calendar_id = str(item.get("id") or "").strip()
            if not calendar_id:
                continue
            timezone_name = str(item.get("timeZone") or "").strip()
            summary = str(item.get("summary") or calendar_id).strip() or calendar_id
            access_role = str(item.get("accessRole") or "reader").strip() or "reader"
            try:
                results.append(
                    validate_calendar_info(
                        CalendarInfo(
                            calendar_id=calendar_id,
                            summary=summary,
                            timezone=timezone_name,
                            primary=bool(item.get("primary")),
                            access_role=access_role,
                        )
                    )
                )
            except (CalendarValidationError, ValueError):
                logger.error(
                    "Skipping calendar with invalid metadata calendar_present=yes"
                )
                continue
        return results

    def list_events(
        self,
        calendar_id: str,
        *,
        time_min_utc: str,
        time_max_utc: str,
        limit: int,
    ) -> list[CalendarEvent]:
        validate_utc_timestamp(time_min_utc, field_name="time_min")
        validate_utc_timestamp(time_max_utc, field_name="time_max")
        try:
            response = (
                self._service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min_utc,
                    timeMax=time_max_utc,
                    maxResults=limit,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except Exception as error:
            raise _map_provider_error(error, operation="list_events") from error
        items = response.get("items") or []
        events: list[CalendarEvent] = []
        for item in items:
            if isinstance(item, dict):
                events.append(normalize_google_event(item, calendar_id=calendar_id))
        return events

    def get_event(self, calendar_id: str, event_id: str) -> CalendarEvent:
        try:
            item = (
                self._service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
        except Exception as error:
            raise _map_provider_error(error, operation="get_event") from error
        if not isinstance(item, dict):
            raise CalendarError(
                "Provider returned an invalid event payload.",
                category="server_error",
            )
        return normalize_google_event(item, calendar_id=calendar_id)

    def get_freebusy(
        self,
        calendar_ids: list[str],
        *,
        time_min_utc: str,
        time_max_utc: str,
    ) -> list[FreeBusyWindow]:
        validate_utc_timestamp(time_min_utc, field_name="time_min")
        validate_utc_timestamp(time_max_utc, field_name="time_max")
        body = {
            "timeMin": time_min_utc,
            "timeMax": time_max_utc,
            "items": [{"id": calendar_id} for calendar_id in calendar_ids],
        }
        try:
            response = self._service.freebusy().query(body=body).execute()
        except Exception as error:
            raise _map_provider_error(error, operation="get_freebusy") from error
        calendars = response.get("calendars")
        if not isinstance(calendars, dict):
            raise CalendarError(
                "Provider returned invalid freebusy payload.",
                category="server_error",
            )
        windows: list[FreeBusyWindow] = []
        for calendar_id in calendar_ids:
            entry = calendars.get(calendar_id)
            if not isinstance(entry, dict):
                raise CalendarError(
                    f"Freebusy missing calendar entry for {calendar_id}.",
                    category="server_error",
                    user_message=(
                        "Cortana: Free/busy could not be verified for one or more "
                        "calendars."
                    ),
                )
            errors = entry.get("errors")
            if errors:
                raise CalendarError(
                    f"Freebusy access error for calendar {calendar_id}.",
                    category="auth_error",
                    user_message=(
                        "Cortana: Free/busy could not be verified for one or more "
                        "calendars."
                    ),
                )
            busy_list = entry.get("busy") or []
            if not isinstance(busy_list, list):
                raise CalendarError(
                    "Provider returned invalid busy list.",
                    category="server_error",
                )
            for busy in busy_list:
                if not isinstance(busy, dict):
                    continue
                start = str(busy.get("start") or "").strip()
                end = str(busy.get("end") or "").strip()
                windows.append(
                    validate_freebusy_window(
                        FreeBusyWindow(
                            calendar_id=calendar_id,
                            start=_normalize_provider_timestamp(start),
                            end=_normalize_provider_timestamp(end),
                        )
                    )
                )
        return windows

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
        validate_google_event_id(event_id)
        validate_iana_timezone(timezone_name)
        body = {
            "id": event_id,
            "summary": title,
            "start": {
                "dateTime": _to_rfc3339(start_utc),
                "timeZone": timezone_name,
            },
            "end": {
                "dateTime": _to_rfc3339(end_utc),
                "timeZone": timezone_name,
            },
        }
        try:
            item = (
                self._service.events()
                .insert(calendarId=calendar_id, body=body)
                .execute()
            )
        except Exception as error:
            raise _map_provider_error(error, operation="create_event") from error
        if not isinstance(item, dict):
            raise CalendarError(
                "Provider returned an invalid created event.",
                category="server_error",
            )
        return normalize_google_event(item, calendar_id=calendar_id)

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
        validate_iana_timezone(timezone_name)
        body = {
            "start": {
                "dateTime": _to_rfc3339(start_utc),
                "timeZone": timezone_name,
            },
            "end": {
                "dateTime": _to_rfc3339(end_utc),
                "timeZone": timezone_name,
            },
        }
        headers = {"If-Match": etag} if etag else None
        try:
            request = self._service.events().patch(
                calendarId=calendar_id,
                eventId=event_id,
                body=body,
            )
            if headers is not None:
                request.headers = {**getattr(request, "headers", {}), **headers}
            item = request.execute()
        except Exception as error:
            raise _map_provider_error(error, operation="update_event") from error
        if not isinstance(item, dict):
            raise CalendarError(
                "Provider returned an invalid updated event.",
                category="server_error",
            )
        return normalize_google_event(item, calendar_id=calendar_id)

    def delete_event(
        self,
        calendar_id: str,
        event_id: str,
        *,
        etag: str | None,
    ) -> None:
        headers = {"If-Match": etag} if etag else None
        try:
            request = self._service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
            )
            if headers is not None:
                request.headers = {**getattr(request, "headers", {}), **headers}
            request.execute()
        except Exception as error:
            mapped = _map_provider_error(error, operation="delete_event")
            # Google may return 410/404 for already-deleted events.
            if mapped.category == "not_found":
                return
            raise mapped from error


def validate_google_oauth_client_file(path: Path) -> Path:
    """Validate the Desktop OAuth client JSON path without logging contents."""
    if not path.exists():
        raise CalendarError(
            "OAuth client file does not exist.",
            category="validation_error",
            user_message=(
                "Cortana: Google OAuth client file is missing. "
                "Set CORTANA_GOOGLE_OAUTH_CLIENT_FILE to a Desktop client JSON file."
            ),
        )
    if not path.is_file():
        raise CalendarError(
            "OAuth client path is not a regular file.",
            category="validation_error",
            user_message=(
                "Cortana: CORTANA_GOOGLE_OAUTH_CLIENT_FILE must point to a regular file."
            ),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalendarError(
            f"OAuth client file could not be parsed: {type(error).__name__}",
            category="validation_error",
            user_message=(
                "Cortana: Google OAuth client file could not be read safely."
            ),
        ) from error
    if not isinstance(payload, dict):
        raise CalendarError(
            "OAuth client file root must be an object.",
            category="validation_error",
            user_message=(
                "Cortana: Google OAuth client file has an invalid structure."
            ),
        )
    installed = payload.get("installed")
    if not isinstance(installed, dict):
        raise CalendarError(
            "OAuth client file must contain an installed Desktop client.",
            category="validation_error",
            user_message=(
                "Cortana: Google OAuth client file must be a Desktop (installed) client."
            ),
        )
    client_id = str(installed.get("client_id") or "").strip()
    if not client_id:
        raise CalendarError(
            "OAuth client file is missing client_id.",
            category="validation_error",
            user_message=(
                "Cortana: Google OAuth client file is missing required fields."
            ),
        )
    return path


def run_google_oauth_flow(client_secrets_file: Path) -> Credentials:
    """Run the installed-app loopback OAuth flow and return credentials."""
    validate_google_oauth_client_file(client_secrets_file)
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_file),
            scopes=list(GOOGLE_CALENDAR_SCOPES),
        )
        credentials = flow.run_local_server(port=0, open_browser=True)
    except CalendarError:
        raise
    except Exception as error:
        logger.error(
            "Google OAuth flow failed error_type=%s",
            type(error).__name__,
        )
        raise CalendarError(
            f"OAuth flow failed: {type(error).__name__}",
            category="auth_error",
            user_message=(
                "Cortana: Google Calendar authorization did not complete."
            ),
        ) from error
    if not isinstance(credentials, Credentials):
        raise CalendarError(
            "OAuth flow returned invalid credentials.",
            category="auth_error",
        )
    if not credentials.refresh_token:
        raise CalendarError(
            "OAuth flow did not return a refresh token.",
            category="auth_error",
            user_message=(
                "Cortana: Google did not return a refresh token. "
                "Revoke prior consent and reconnect."
            ),
        )
    granted = tuple(sorted(credentials.scopes or ()))
    expected = tuple(sorted(GOOGLE_CALENDAR_SCOPES))
    if granted and set(granted) != set(expected):
        # Allow supersets only if all required scopes are present.
        if not set(expected).issubset(set(granted)):
            raise CalendarError(
                "OAuth granted scopes do not include required calendar scopes.",
                category="auth_error",
                user_message=(
                    "Cortana: Google authorization did not grant the required "
                    "calendar scopes."
                ),
            )
    return credentials


def credentials_from_refresh_token(
    *,
    client_secrets_file: Path,
    refresh_token: str,
) -> Credentials:
    """Build credentials from a stored refresh token and Desktop client file."""
    validate_google_oauth_client_file(client_secrets_file)
    payload = json.loads(client_secrets_file.read_text(encoding="utf-8"))
    installed = payload["installed"]
    client_id = str(installed["client_id"])
    client_secret = str(installed.get("client_secret") or "")
    token_uri = str(installed.get("token_uri") or GOOGLE_TOKEN_URI)
    credentials = Credentials(  # type: ignore[no-untyped-call]
        token=None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(GOOGLE_CALENDAR_SCOPES),
    )
    try:
        credentials.refresh(Request())  # type: ignore[no-untyped-call]
    except RefreshError as error:
        logger.error("Google token refresh failed error_type=RefreshError")
        raise CalendarError(
            "Refresh token rejected by Google.",
            category="auth_error",
            user_message=(
                "Cortana: Calendar authorization is no longer valid. "
                "Reconnect with /calendar-connect."
            ),
        ) from error
    except Exception as error:
        logger.error(
            "Google token refresh failed error_type=%s",
            type(error).__name__,
        )
        raise CalendarError(
            f"Token refresh failed: {type(error).__name__}",
            category="network_error",
        ) from error
    return credentials


def revoke_google_refresh_token(refresh_token: str) -> None:
    """Best-effort revoke of a Google refresh token."""
    try:
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode({"token": refresh_token}).encode("utf-8")
        request = urllib.request.Request(
            GOOGLE_REVOKE_URI,
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            _ = response.status
    except Exception as error:
        logger.error(
            "Google token revoke failed error_type=%s",
            type(error).__name__,
        )


def normalize_google_event(
    item: dict[str, Any],
    *,
    calendar_id: str,
) -> CalendarEvent:
    """Normalize one Google event resource into a transient CalendarEvent."""
    event_id = str(item.get("id") or "").strip()
    title = str(item.get("summary") or "").strip() or "(untitled)"
    description = item.get("description")
    description_text = (
        str(description) if description is not None else None
    )
    status = str(item.get("status") or "confirmed").strip() or "confirmed"
    etag = str(item.get("etag") or "").strip() or None
    recurring_event_id = item.get("recurringEventId")
    recurring_id = (
        str(recurring_event_id).strip()
        if recurring_event_id is not None and str(recurring_event_id).strip()
        else None
    )
    recurrence = item.get("recurrence")
    is_recurring = bool(recurring_id) or bool(recurrence)
    start_raw = item.get("start") or {}
    end_raw = item.get("end") or {}
    if not isinstance(start_raw, dict) or not isinstance(end_raw, dict):
        raise CalendarError(
            "Event is missing start/end.",
            category="server_error",
        )
    if "date" in start_raw or "date" in end_raw:
        start = str(start_raw.get("date") or "").strip()
        end = str(end_raw.get("date") or "").strip()
        return validate_calendar_event(
            CalendarEvent(
                event_id=event_id,
                calendar_id=calendar_id,
                title=title,
                description=description_text,
                is_all_day=True,
                start=start,
                end=end,
                status=status,
                is_recurring=is_recurring,
                recurring_event_id=recurring_id,
                etag=etag,
            )
        )
    start = _normalize_provider_timestamp(str(start_raw.get("dateTime") or ""))
    end = _normalize_provider_timestamp(str(end_raw.get("dateTime") or ""))
    return validate_calendar_event(
        CalendarEvent(
            event_id=event_id,
            calendar_id=calendar_id,
            title=title,
            description=description_text,
            is_all_day=False,
            start=start,
            end=end,
            status=status,
            is_recurring=is_recurring,
            recurring_event_id=recurring_id,
            etag=etag,
        )
    )


def _normalize_provider_timestamp(value: str) -> str:
    from datetime import datetime, timezone

    cleaned = value.strip()
    if not cleaned:
        raise CalendarValidationError("Provider timestamp is blank.")
    normalized = cleaned.replace("Z", "+00:00")
    try:
        aware = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise CalendarValidationError("Provider timestamp is invalid.") from error
    if aware.tzinfo is None:
        raise CalendarValidationError("Provider timestamp must be aware.")
    return aware.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _to_rfc3339(utc_iso: str) -> str:
    cleaned = validate_utc_timestamp(utc_iso, field_name="Timestamp")
    return cleaned.replace("Z", "+00:00")


def _map_provider_error(error: Exception, *, operation: str) -> CalendarError:
    if isinstance(error, CalendarError):
        return error
    if isinstance(error, RefreshError):
        return CalendarError(
            f"{operation} auth refresh failed.",
            category="auth_error",
        )
    if isinstance(error, HttpError):
        status = int(getattr(error.resp, "status", 0) or 0)
        if status in {401, 403}:
            return CalendarError(
                f"{operation} unauthorized.",
                category="auth_error",
            )
        if status in {404, 410}:
            return CalendarError(
                f"{operation} not found.",
                category="not_found",
            )
        if status == 409 or status == 412:
            return CalendarError(
                f"{operation} conflict.",
                category="conflict",
                user_message=(
                    "Cortana: The calendar event changed remotely. "
                    "Prepare the change again."
                ),
            )
        if status == 429:
            return CalendarError(
                f"{operation} rate limited.",
                category="rate_limited",
            )
        if status >= 500:
            return CalendarError(
                f"{operation} server error.",
                category="server_error",
            )
        return CalendarError(
            f"{operation} http error status={status}.",
            category="server_error",
        )
    name = type(error).__name__
    if "Timeout" in name or "Connection" in name or "Transport" in name:
        return CalendarError(
            f"{operation} network error.",
            category="network_error",
        )
    logger.error(
        "Calendar provider error operation=%s error_type=%s",
        operation,
        name,
    )
    return CalendarError(
        f"{operation} failed: {name}",
        category="server_error",
    )
