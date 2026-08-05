"""Exact-key JSON request/response envelopes for process-isolated tool IPC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from src.config import (
    MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS,
    MAX_PROCESS_IPC_REQUEST_BYTES,
    MAX_PROCESS_IPC_RESPONSE_BYTES,
    MAX_PROCESS_PARAMETER_COUNT,
    MAX_PROCESS_PARAMETER_NAME_CHARS,
    MAX_PROCESS_PARAMETER_NESTING_DEPTH,
    MAX_PROCESS_PARAMETER_STRING_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TOOL_TIMEOUT_SECONDS,
    PROCESS_IPC_SCHEMA_VERSION,
)
from src.tool_process_common import (
    CHILD_ALLOWED_OUTCOMES,
    PROCESS_SAFE_FILE_IMPLEMENTATION_IDS,
    ToolProcessError,
    ToolProcessResultRejectedError,
    assert_implementation_process_safe,
)
from src.tool_process_file_auth import validate_file_tool_process_parameters

REQUEST_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "correlation_id",
        "implementation_identifier",
        "normalized_parameters",
        "execution_timeout_seconds",
        "max_output_characters",
    }
)
RESPONSE_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "correlation_id",
        "outcome",
        "structured_data",
        "safe_summary",
        "output_truncated",
        "error_class",
    }
)


@dataclass(frozen=True)
class ProcessExecutionRequest:
    """Immutable parent-to-child process execution request envelope."""

    schema_version: int
    correlation_id: str
    implementation_identifier: str
    normalized_parameters: dict[str, Any]
    execution_timeout_seconds: int
    max_output_characters: int


@dataclass(frozen=True)
class ProcessExecutionResponse:
    """Immutable child-to-parent process execution response envelope."""

    schema_version: int
    correlation_id: str
    outcome: str
    structured_data: dict[str, Any]
    safe_summary: str
    output_truncated: bool
    error_class: str | None


def create_process_execution_request(
    *,
    correlation_id: str,
    implementation_identifier: str,
    normalized_parameters: dict[str, Any],
    execution_timeout_seconds: int,
    max_output_characters: int,
    schema_version: int = PROCESS_IPC_SCHEMA_VERSION,
) -> ProcessExecutionRequest:
    """Create and validate one process execution request envelope."""
    return validate_process_execution_request(
        {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "implementation_identifier": implementation_identifier,
            "normalized_parameters": dict(normalized_parameters),
            "execution_timeout_seconds": execution_timeout_seconds,
            "max_output_characters": max_output_characters,
        }
    )


def validate_process_execution_request(
    payload: dict[str, Any],
) -> ProcessExecutionRequest:
    """Validate an exact-key process request dictionary."""
    if not isinstance(payload, dict):
        raise ToolProcessError("Process request must be a JSON object.")
    _assert_exact_keys(payload, REQUEST_KEYS, label="process request")

    schema_version = payload["schema_version"]
    if schema_version != PROCESS_IPC_SCHEMA_VERSION:
        raise ToolProcessError("Unsupported process request schema version.")

    correlation_id = _require_uuid(payload["correlation_id"], "correlation_id")
    implementation = assert_implementation_process_safe(
        str(payload["implementation_identifier"])
    )
    parameters = _validate_parameters(payload["normalized_parameters"])
    if implementation in PROCESS_SAFE_FILE_IMPLEMENTATION_IDS:
        parameters = validate_file_tool_process_parameters(
            implementation,
            parameters,
        )
    elif "file_authorization" in parameters:
        raise ToolProcessError(
            "file_authorization is not permitted for this implementation."
        )
    timeout = _require_bounded_int(
        payload["execution_timeout_seconds"],
        field_name="execution_timeout_seconds",
        minimum=1,
        maximum=MAX_TOOL_TIMEOUT_SECONDS,
    )
    max_output = _require_bounded_int(
        payload["max_output_characters"],
        field_name="max_output_characters",
        minimum=1,
        maximum=MAX_TOOL_OUTPUT_CHARS,
    )

    request = ProcessExecutionRequest(
        schema_version=int(schema_version),
        correlation_id=correlation_id,
        implementation_identifier=implementation,
        normalized_parameters=parameters,
        execution_timeout_seconds=timeout,
        max_output_characters=max_output,
    )
    encoded = encode_process_request(request)
    if len(encoded) > MAX_PROCESS_IPC_REQUEST_BYTES:
        raise ToolProcessError("Process request exceeds the maximum byte size.")
    return request


def encode_process_request(request: ProcessExecutionRequest) -> bytes:
    """Serialize a validated request to UTF-8 JSON bytes."""
    payload = {
        "schema_version": request.schema_version,
        "correlation_id": request.correlation_id,
        "implementation_identifier": request.implementation_identifier,
        "normalized_parameters": request.normalized_parameters,
        "execution_timeout_seconds": request.execution_timeout_seconds,
        "max_output_characters": request.max_output_characters,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_process_request(raw: bytes) -> ProcessExecutionRequest:
    """Decode and validate a process request from bounded JSON bytes."""
    if len(raw) > MAX_PROCESS_IPC_REQUEST_BYTES:
        raise ToolProcessError("Process request exceeds the maximum byte size.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToolProcessError("Malformed process request JSON.") from error
    if not isinstance(payload, dict):
        raise ToolProcessError("Process request must be a JSON object.")
    return validate_process_execution_request(payload)


def create_process_execution_response(
    *,
    correlation_id: str,
    outcome: str,
    structured_data: dict[str, Any] | None = None,
    safe_summary: str,
    output_truncated: bool = False,
    error_class: str | None = None,
    schema_version: int = PROCESS_IPC_SCHEMA_VERSION,
) -> ProcessExecutionResponse:
    """Create and validate one child process response envelope."""
    return validate_process_execution_response(
        {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "outcome": outcome,
            "structured_data": dict(structured_data or {}),
            "safe_summary": safe_summary,
            "output_truncated": output_truncated,
            "error_class": error_class,
        },
        expected_correlation_id=correlation_id,
    )


def validate_process_execution_response(
    payload: dict[str, Any],
    *,
    expected_correlation_id: str,
) -> ProcessExecutionResponse:
    """Validate an exact-key process response dictionary."""
    if not isinstance(payload, dict):
        raise ToolProcessResultRejectedError("Process response must be a JSON object.")
    try:
        _assert_exact_keys(payload, RESPONSE_KEYS, label="process response")
    except ToolProcessError as error:
        raise ToolProcessResultRejectedError(str(error)) from error

    schema_version = payload["schema_version"]
    if schema_version != PROCESS_IPC_SCHEMA_VERSION:
        raise ToolProcessResultRejectedError(
            "Unsupported process response schema version."
        )

    correlation_id = _require_uuid(payload["correlation_id"], "correlation_id")
    if correlation_id != expected_correlation_id:
        raise ToolProcessResultRejectedError("Process response correlation mismatch.")

    outcome = payload["outcome"]
    if not isinstance(outcome, str) or outcome not in CHILD_ALLOWED_OUTCOMES:
        raise ToolProcessResultRejectedError("Invalid child process outcome.")

    structured = payload["structured_data"]
    if not isinstance(structured, dict):
        raise ToolProcessResultRejectedError("structured_data must be an object.")
    _assert_json_primitives(structured, depth=0)

    summary = payload["safe_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ToolProcessResultRejectedError("safe_summary must be a non-empty string.")
    if len(summary) > 2_000:
        raise ToolProcessResultRejectedError("safe_summary exceeds the maximum length.")

    truncated = payload["output_truncated"]
    if not isinstance(truncated, bool):
        raise ToolProcessResultRejectedError("output_truncated must be a boolean.")

    error_class = payload["error_class"]
    if error_class is not None:
        if not isinstance(error_class, str) or not error_class.strip():
            raise ToolProcessResultRejectedError("error_class must be a string or null.")
        if len(error_class) > 200:
            raise ToolProcessResultRejectedError("error_class exceeds the maximum length.")
        error_class = error_class.strip()

    response = ProcessExecutionResponse(
        schema_version=int(schema_version),
        correlation_id=correlation_id,
        outcome=outcome,
        structured_data=dict(structured),
        safe_summary=summary.strip(),
        output_truncated=truncated,
        error_class=error_class,
    )
    encoded = encode_process_response(response)
    if len(encoded) > MAX_PROCESS_IPC_RESPONSE_BYTES:
        raise ToolProcessResultRejectedError(
            "Process response exceeds the maximum byte size."
        )
    return response


def encode_process_response(response: ProcessExecutionResponse) -> bytes:
    """Serialize a validated response to UTF-8 JSON bytes."""
    payload = {
        "schema_version": response.schema_version,
        "correlation_id": response.correlation_id,
        "outcome": response.outcome,
        "structured_data": response.structured_data,
        "safe_summary": response.safe_summary,
        "output_truncated": response.output_truncated,
        "error_class": response.error_class,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_process_response(
    raw: bytes,
    *,
    expected_correlation_id: str,
) -> ProcessExecutionResponse:
    """Decode exactly one schema-validated response document from bytes."""
    if not raw:
        raise ToolProcessResultRejectedError("Process response document is missing.")
    if len(raw) > MAX_PROCESS_IPC_RESPONSE_BYTES:
        raise ToolProcessResultRejectedError(
            "Process response exceeds the maximum byte size."
        )
    text = raw.decode("utf-8")
    decoder = json.JSONDecoder()
    try:
        payload, index = decoder.raw_decode(text.lstrip())
    except json.JSONDecodeError as error:
        raise ToolProcessResultRejectedError("Malformed process response JSON.") from error
    trailing = text.lstrip()[index:].strip()
    if trailing:
        raise ToolProcessResultRejectedError(
            "Process response contains unexpected extra data."
        )
    if not isinstance(payload, dict):
        raise ToolProcessResultRejectedError("Process response must be a JSON object.")
    return validate_process_execution_response(
        payload,
        expected_correlation_id=expected_correlation_id,
    )


def _assert_exact_keys(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    label: str,
) -> None:
    keys = frozenset(payload)
    if keys != allowed:
        raise ToolProcessError(f"{label} keys are not an exact match.")


def _require_uuid(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolProcessError(f"{field_name} must be a UUID string.")
    try:
        return str(UUID(value.strip()))
    except ValueError as error:
        raise ToolProcessError(f"{field_name} must be a UUID string.") from error


def _require_bounded_int(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolProcessError(f"{field_name} must be an integer.")
    if value < minimum or value > maximum:
        raise ToolProcessError(f"{field_name} is outside the allowed range.")
    return value


def _validate_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolProcessError("normalized_parameters must be an object.")
    if len(value) > MAX_PROCESS_PARAMETER_COUNT:
        raise ToolProcessError("normalized_parameters exceed the parameter count limit.")
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ToolProcessError("Parameter names must be non-empty strings.")
        name = key.strip()
        if len(name) > MAX_PROCESS_PARAMETER_NAME_CHARS:
            raise ToolProcessError("Parameter name exceeds the maximum length.")
        if name == "file_authorization":
            cleaned[name] = _assert_json_primitives(
                item,
                depth=0,
                string_limit=MAX_PROCESS_PARAMETER_STRING_CHARS,
                allow_long_path_strings=True,
            )
        else:
            cleaned[name] = _assert_json_primitives(item, depth=0)
    return cleaned


def _assert_json_primitives(
    value: Any,
    *,
    depth: int,
    string_limit: int = MAX_PROCESS_PARAMETER_STRING_CHARS,
    allow_long_path_strings: bool = False,
) -> Any:
    if depth > MAX_PROCESS_PARAMETER_NESTING_DEPTH:
        raise ToolProcessError("Parameter nesting exceeds the maximum depth.")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        limit = (
            MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS
            if allow_long_path_strings
            else string_limit
        )
        if len(value) > limit:
            raise ToolProcessError("Parameter string exceeds the maximum length.")
        return value
    if isinstance(value, list):
        return [
            _assert_json_primitives(
                item,
                depth=depth + 1,
                string_limit=string_limit,
                allow_long_path_strings=False,
            )
            for item in value
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolProcessError("Object keys must be strings.")
            # Only canonical_path / authorized_root may use the path bound.
            nested_long = allow_long_path_strings and key in {
                "canonical_path",
                "authorized_root",
            }
            result[key] = _assert_json_primitives(
                item,
                depth=depth + 1,
                string_limit=string_limit,
                allow_long_path_strings=nested_long,
            )
        return result
    raise ToolProcessError("Parameters must use JSON-primitive values only.")
