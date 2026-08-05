"""Child process entry point for Milestone 13 process-isolated tool execution.

Invoked only as:
    python -m src.tool_process_runner <result_path>

Reads one validated JSON request from stdin and writes exactly one validated JSON
response to the parent-created result file. stdout/stderr are diagnostic-only and
must never carry the authoritative result.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.config import MAX_PROCESS_IPC_REQUEST_BYTES, MAX_TOOL_OUTPUT_CHARS
from src.tool_process_callables import PROCESS_SAFE_CALLABLES
from src.tool_process_common import (
    PROCESS_SAFE_IMPLEMENTATION_IDS,
    ToolProcessError,
)
from src.tool_process_envelope import (
    ProcessExecutionRequest,
    create_process_execution_response,
    decode_process_request,
    encode_process_response,
)

# Minimal hardcoded dispatch table. Must match registry eligible/required IDs.
CHILD_DISPATCH: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {
    identifier: PROCESS_SAFE_CALLABLES[identifier]
    for identifier in sorted(PROCESS_SAFE_IMPLEMENTATION_IDS)
}

CHILD_DISPATCH_IMPLEMENTATION_IDS: frozenset[str] = frozenset(CHILD_DISPATCH)


def _bound_structured_output(
    data: dict[str, Any],
    *,
    maximum_characters: int,
) -> tuple[dict[str, Any], bool]:
    """Bound structured output size inside the child before responding."""
    rendered = str(data)
    if len(rendered) <= maximum_characters:
        return dict(data), False
    compact: dict[str, Any] = {"output_truncated": True}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            candidate = {**compact, key: value}
            if len(str(candidate)) <= maximum_characters:
                compact = candidate
        elif isinstance(value, list):
            bounded_list: list[Any] = []
            for item in value:
                trial = {**compact, key: [*bounded_list, item]}
                if len(str(trial)) > maximum_characters:
                    break
                bounded_list.append(item)
            compact[key] = bounded_list
    return compact, True


def _write_response(result_path: Path, response_bytes: bytes) -> None:
    """Write exactly one response document to the dedicated result channel."""
    result_path.write_bytes(response_bytes)


def _failed_response(
    request: ProcessExecutionRequest | None,
    *,
    correlation_id: str,
    error_class: str,
    summary: str,
) -> bytes:
    response = create_process_execution_response(
        correlation_id=correlation_id,
        outcome="failed",
        structured_data={"failed": True},
        safe_summary=summary,
        output_truncated=False,
        error_class=error_class,
    )
    return encode_process_response(response)


def run_child(result_path: Path) -> int:
    """Execute one process-isolated tool request and write the result channel."""
    correlation_id = "00000000-0000-0000-0000-000000000000"
    request: ProcessExecutionRequest | None = None
    try:
        raw = sys.stdin.buffer.read(MAX_PROCESS_IPC_REQUEST_BYTES + 1)
        if len(raw) > MAX_PROCESS_IPC_REQUEST_BYTES:
            raise ToolProcessError("Process request exceeds the maximum byte size.")
        request = decode_process_request(raw)
        correlation_id = request.correlation_id

        callable_impl = CHILD_DISPATCH.get(request.implementation_identifier)
        if callable_impl is None:
            raise ToolProcessError("Implementation identifier is not in child dispatch.")

        try:
            raw_data = callable_impl(request.normalized_parameters, {})
        except Exception as error:  # noqa: BLE001 - contained child failure
            response_bytes = _failed_response(
                request,
                correlation_id=correlation_id,
                error_class=type(error).__name__,
                summary="Tool execution failed.",
            )
            _write_response(result_path, response_bytes)
            return 1

        if not isinstance(raw_data, dict):
            raise ToolProcessError("Tool implementation returned invalid data.")

        max_chars = min(request.max_output_characters, MAX_TOOL_OUTPUT_CHARS)
        structured, truncated = _bound_structured_output(
            raw_data,
            maximum_characters=max_chars,
        )
        response = create_process_execution_response(
            correlation_id=correlation_id,
            outcome="succeeded",
            structured_data=structured,
            safe_summary="Process-isolated tool completed successfully.",
            output_truncated=truncated,
            error_class=None,
        )
        _write_response(result_path, encode_process_response(response))
        return 0
    except Exception as error:  # noqa: BLE001 - always emit a bounded failure
        try:
            response_bytes = _failed_response(
                request,
                correlation_id=correlation_id,
                error_class=type(error).__name__,
                summary="Tool execution failed.",
            )
            _write_response(result_path, response_bytes)
        except Exception:
            return 2
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry for ``python -m src.tool_process_runner``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        # Do not write authoritative JSON to stdout.
        print("usage: python -m src.tool_process_runner <result_path>", file=sys.stderr)
        return 2
    result_path = Path(args[0])
    return run_child(result_path)


if __name__ == "__main__":
    raise SystemExit(main())
