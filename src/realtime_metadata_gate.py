"""Manual M30 live Realtime metadata-echo gate helpers.

This module never runs a live API call on import. Pytest must not invoke
``run_live_metadata_validation``. The live check is a release gate only.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from src.realtime_voice import build_session_update_payload
from src.settings import Settings

CORTANA_USER_ITEM_META = "cortana_user_item_id"
CORTANA_GENERATION_META = "cortana_generation"
METADATA_GATE_INSTRUCTIONS = (
    "Metadata echo validation only. Do not produce user content."
)
_GATE_TIMEOUT_SECONDS = 20.0

# Validated M30 release-gate record from the manual live metadata check.
# Diagnostics reports this status. Startup and /status never contact the
# Realtime API to re-measure it.
RealtimeMetadataGateReleaseOutcome = Literal["PASS", "FAIL"]
REALTIME_METADATA_GATE_RELEASE_OUTCOME: RealtimeMetadataGateReleaseOutcome = (
    "PASS"
)


@dataclass(frozen=True)
class MetadataEchoResult:
    """Exact-match evaluation of response.create metadata echo."""

    outcome: Literal["PASS", "FAIL"]
    reason: str
    sent: dict[str, str]
    received: dict[str, str] | None
    commit_monotonic: float | None = None
    create_send_monotonic: float | None = None
    created_monotonic: float | None = None


def build_gate_metadata(
    *,
    user_item_id: str = "gate-item-1",
    generation: str = "1",
) -> dict[str, str]:
    """Return the metadata keys the live gate must send and receive."""
    return {
        CORTANA_USER_ITEM_META: user_item_id,
        CORTANA_GENERATION_META: generation,
    }


def extract_response_metadata(event: object) -> dict[str, str] | None:
    """Read response.created metadata from an SDK event or mapping."""
    response = _attr_or_key(event, "response")
    if response is None:
        return None
    metadata = _attr_or_key(response, "metadata")
    if not isinstance(metadata, dict):
        return None
    extracted: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(key, str) and isinstance(value, str):
            extracted[key] = value
    return extracted


def evaluate_metadata_echo(
    *,
    sent: Mapping[str, str],
    received: Mapping[str, str] | None,
    commit_monotonic: float | None = None,
    create_send_monotonic: float | None = None,
    created_monotonic: float | None = None,
    reason: str | None = None,
) -> MetadataEchoResult:
    """PASS only when both required keys are present and values match exactly."""
    sent_payload = {
        CORTANA_USER_ITEM_META: str(sent.get(CORTANA_USER_ITEM_META, "")),
        CORTANA_GENERATION_META: str(sent.get(CORTANA_GENERATION_META, "")),
    }
    if received is None:
        return MetadataEchoResult(
            outcome="FAIL",
            reason=reason or "missing metadata",
            sent=sent_payload,
            received=None,
            commit_monotonic=commit_monotonic,
            create_send_monotonic=create_send_monotonic,
            created_monotonic=created_monotonic,
        )
    received_payload = {
        key: value
        for key, value in received.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    for key in (CORTANA_USER_ITEM_META, CORTANA_GENERATION_META):
        if key not in received_payload:
            return MetadataEchoResult(
                outcome="FAIL",
                reason=reason or "missing metadata",
                sent=sent_payload,
                received=received_payload,
                commit_monotonic=commit_monotonic,
                create_send_monotonic=create_send_monotonic,
                created_monotonic=created_monotonic,
            )
        if received_payload[key] != sent_payload[key]:
            return MetadataEchoResult(
                outcome="FAIL",
                reason=reason or "metadata mismatch",
                sent=sent_payload,
                received=received_payload,
                commit_monotonic=commit_monotonic,
                create_send_monotonic=create_send_monotonic,
                created_monotonic=created_monotonic,
            )
    return MetadataEchoResult(
        outcome="PASS",
        reason=reason or "metadata echoed exactly",
        sent=sent_payload,
        received=received_payload,
        commit_monotonic=commit_monotonic,
        create_send_monotonic=create_send_monotonic,
        created_monotonic=created_monotonic,
    )


def realtime_metadata_gate_diagnostics_line() -> str:
    """Return the concise diagnostics line for the M30 metadata gate.

    This is the recorded release-gate result, not a per-startup live
    measurement.
    """
    return f"Realtime metadata gate: {REALTIME_METADATA_GATE_RELEASE_OUTCOME}"


def fail_metadata_gate(reason: str, sent: Mapping[str, str] | None = None) -> MetadataEchoResult:
    """Return a FAIL result for connection or API errors."""
    payload = dict(sent) if sent is not None else build_gate_metadata()
    return evaluate_metadata_echo(sent=payload, received=None, reason=reason)


def format_latency_line(label: str, start: float | None, end: float | None) -> str:
    """Return a measured duration line or NOT MEASURED."""
    if start is None or end is None:
        return f"{label}: NOT MEASURED"
    milliseconds = (end - start) * 1000.0
    return f"{label}: {milliseconds:.1f} ms"


def format_gate_report(result: MetadataEchoResult) -> str:
    """Return a secret-free PASS/FAIL report with observed latency only."""
    lines = [
        f"Realtime metadata gate: {result.outcome}",
        result.reason,
        format_latency_line(
            "commit -> create-send",
            result.commit_monotonic,
            result.create_send_monotonic,
        ),
        format_latency_line(
            "create-send -> response-created",
            result.create_send_monotonic,
            result.created_monotonic,
        ),
        format_latency_line(
            "commit -> response-created",
            result.commit_monotonic,
            result.created_monotonic,
        ),
    ]
    return "\n".join(lines)


def session_update_payload(settings: Settings) -> dict[str, object]:
    """Reuse the M25 session.update payload for the live gate."""
    return build_session_update_payload(
        settings=settings,
        instructions=METADATA_GATE_INSTRUCTIONS,
    )


def run_live_metadata_validation(settings: Settings, client: object) -> MetadataEchoResult:
    """Connect to the live Realtime API and evaluate metadata echo.

    Manual release-gate use only. Do not call from pytest or startup.
    """
    sent = build_gate_metadata()
    connect = getattr(getattr(client, "realtime", None), "connect", None)
    if not callable(connect):
        return fail_metadata_gate("connection error", sent)
    try:
        manager = connect(model=settings.realtime_model, max_retries=0)
    except Exception as error:
        return fail_metadata_gate(f"connection error ({type(error).__name__})", sent)
    try:
        with manager as connection:
            connection.session.update(session=session_update_payload(settings))
            create_send = time.monotonic()
            connection.response.create(response={"metadata": sent})
            event = _wait_for_response_created(connection)
            created = time.monotonic()
            if event is None:
                return evaluate_metadata_echo(
                    sent=sent,
                    received=None,
                    create_send_monotonic=create_send,
                    created_monotonic=None,
                    reason="missing metadata",
                )
            received = extract_response_metadata(event)
            return evaluate_metadata_echo(
                sent=sent,
                received=received,
                create_send_monotonic=create_send,
                created_monotonic=created,
            )
    except Exception as error:
        return fail_metadata_gate(f"API error ({type(error).__name__})", sent)


def main() -> int:
    """CLI entry used only by scripts/validate_realtime_metadata.py."""
    from src.openai_client import create_openai_client
    from src.settings import load_settings

    try:
        settings = load_settings()
    except Exception as error:
        print(format_gate_report(fail_metadata_gate(f"configuration error ({type(error).__name__})")))
        return 1
    client = create_openai_client(settings)
    result = run_live_metadata_validation(settings, client)
    print(format_gate_report(result))
    return 0 if result.outcome == "PASS" else 1


def _attr_or_key(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _is_response_created(event: object) -> bool:
    if str(getattr(event, "type", "")) == "response.created":
        return True
    return isinstance(event, Mapping) and event.get("type") == "response.created"


def _wait_for_response_created(connection: object) -> object | None:
    deadline = time.monotonic() + _GATE_TIMEOUT_SECONDS
    recv = getattr(connection, "recv", None)
    if callable(recv):
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                event = cast(object, recv(timeout=remaining))
            except TypeError:
                event = cast(object, recv())
            except Exception:
                return None
            if _is_response_created(event):
                return event
        return None
    iterate = getattr(connection, "__iter__", None)
    if not callable(iterate):
        return None
    try:
        stream = iterate()
    except Exception:
        return None
    while time.monotonic() < deadline:
        try:
            event = cast(object, next(stream))
        except StopIteration:
            return None
        except Exception:
            return None
        if _is_response_created(event):
            return event
    return None
