"""Coordinated local JSON persistence for the defensive tool framework.

One application instance should access a tool-control repository file at a time.
Atomic writes do not coordinate concurrent Cortana processes; last-writer-wins
lost updates remain possible. Cross-process locking is not implemented yet.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from src.config import TOOL_CONTROL_REPOSITORY_SCHEMA_VERSION
from src.tool_approval import ToolApprovalRecord, validate_tool_approval
from src.tool_audit import ToolAuditEntry, create_tool_audit_entry, validate_tool_audit_entry
from src.tool_common import ToolTargetType, ToolValidationError
from src.tool_request import ToolExecutionRequest, validate_tool_execution_request
from src.tool_result import ToolExecutionResult, validate_tool_execution_result
from src.tool_scope import AuthorizedScope, validate_authorized_scope

logger = logging.getLogger("ProjectCortana")

TOOL_LOAD_ERROR_MESSAGE = (
    "Cortana: Tool-control repository data could not be loaded safely. "
    "The existing repository file was left unchanged for inspection."
)
TOOL_SAVE_ERROR_MESSAGE = (
    "Cortana: Tool-control repository data could not be updated safely."
)


class ToolStorageError(RuntimeError):
    """Raised when tool-control repository data cannot be loaded or saved safely."""

    def __init__(self, message: str = TOOL_LOAD_ERROR_MESSAGE) -> None:
        super().__init__(message)
        self.user_message = message


@dataclass
class _ToolRepositoryState:
    """In-memory snapshot of the coordinated tool-control repository."""

    scopes: list[AuthorizedScope] = field(default_factory=list)
    requests: list[ToolExecutionRequest] = field(default_factory=list)
    approvals: list[ToolApprovalRecord] = field(default_factory=list)
    results: list[ToolExecutionResult] = field(default_factory=list)
    audit_entries: list[ToolAuditEntry] = field(default_factory=list)


class ToolControlRepository(Protocol):
    """Protocol for coordinated defensive-tool persistence."""

    def list_scopes(self) -> list[AuthorizedScope]:
        """Return saved scopes in storage order."""

    def get_scope(self, scope_id: str) -> AuthorizedScope | None:
        """Return one scope by ID, or None when not found."""

    def add_scope(self, scope: AuthorizedScope) -> AuthorizedScope:
        """Persist one validated scope and append an audit entry."""

    def update_scope(self, scope: AuthorizedScope) -> AuthorizedScope:
        """Replace one existing scope with a validated record."""

    def active_scope_count(self) -> int:
        """Return count of active unexpired scopes is computed by callers; this is active flag count."""

    def list_requests(self) -> list[ToolExecutionRequest]:
        """Return saved requests in storage order."""

    def get_request(self, request_id: str) -> ToolExecutionRequest | None:
        """Return one request by ID, or None when not found."""

    def add_request(self, request: ToolExecutionRequest) -> ToolExecutionRequest:
        """Persist one validated request and append an audit entry."""

    def update_request(self, request: ToolExecutionRequest) -> ToolExecutionRequest:
        """Replace one existing request with a validated record."""

    def pending_approval_count(self) -> int:
        """Return count of requests awaiting approval."""

    def list_approvals(self) -> list[ToolApprovalRecord]:
        """Return saved approvals in storage order."""

    def get_approval_for_request(
        self,
        request_id: str,
    ) -> ToolApprovalRecord | None:
        """Return the latest approval record for a request, if any."""

    def add_approval(self, approval: ToolApprovalRecord) -> ToolApprovalRecord:
        """Persist one validated approval and append an audit entry."""

    def list_results(self) -> list[ToolExecutionResult]:
        """Return saved results in storage order."""

    def get_result(self, result_id: str) -> ToolExecutionResult | None:
        """Return one result by ID, or None when not found."""

    def add_result(self, result: ToolExecutionResult) -> ToolExecutionResult:
        """Persist one validated result."""

    def list_audit_entries(self) -> list[ToolAuditEntry]:
        """Return audit entries in append order."""

    def append_audit_entry(self, entry: ToolAuditEntry) -> ToolAuditEntry:
        """Append one validated audit entry."""


class JsonToolControlRepository:
    """UTF-8 JSON-backed coordinated repository for tool-control data."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._state: _ToolRepositoryState | None = None
        self._load_error: ToolStorageError | None = None

    @property
    def file_path(self) -> Path:
        """Return the configured repository file path."""
        return self._file_path

    def list_scopes(self) -> list[AuthorizedScope]:
        return list(self._ensure_loaded().scopes)

    def get_scope(self, scope_id: str) -> AuthorizedScope | None:
        for scope in self._ensure_loaded().scopes:
            if scope.scope_id == scope_id:
                return scope
        return None

    def add_scope(self, scope: AuthorizedScope) -> AuthorizedScope:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_authorized_scope(scope)
        if any(item.scope_id == validated.scope_id for item in state.scopes):
            raise ToolStorageError("Cortana: Duplicate scope ID rejected.")
        state.scopes.append(validated)
        state.audit_entries.append(
            create_tool_audit_entry(
                action="scope-created",
                scope_id=validated.scope_id,
                safe_details={
                    "allowed_tool_count": len(validated.allowed_tool_ids),
                    "allowed_root_count": len(validated.allowed_local_path_roots),
                    "active": validated.active,
                },
            )
        )
        self._persist(state)
        return validated

    def update_scope(self, scope: AuthorizedScope) -> AuthorizedScope:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_authorized_scope(scope)
        index = self._scope_index(state, validated.scope_id)
        if index is None:
            raise ToolStorageError("Cortana: Scope was not found.")
        state.scopes[index] = validated
        self._persist(state)
        return validated

    def active_scope_count(self) -> int:
        return sum(1 for scope in self._ensure_loaded().scopes if scope.active)

    def list_requests(self) -> list[ToolExecutionRequest]:
        return list(self._ensure_loaded().requests)

    def get_request(self, request_id: str) -> ToolExecutionRequest | None:
        for request in self._ensure_loaded().requests:
            if request.request_id == request_id:
                return request
        return None

    def add_request(self, request: ToolExecutionRequest) -> ToolExecutionRequest:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_tool_execution_request(request)
        if any(item.request_id == validated.request_id for item in state.requests):
            raise ToolStorageError("Cortana: Duplicate request ID rejected.")
        if self._find_scope(state, validated.scope_id) is None:
            raise ToolStorageError("Cortana: Referenced scope was not found.")
        state.requests.append(validated)
        state.audit_entries.append(
            create_tool_audit_entry(
                action="request-created",
                request_id=validated.request_id,
                tool_id=validated.tool_id,
                scope_id=validated.scope_id,
                safe_details={
                    "status": validated.request_status,
                    "dry_run_requested": validated.dry_run_requested,
                    "has_incident": validated.incident_id is not None,
                },
            )
        )
        self._persist(state)
        return validated

    def update_request(self, request: ToolExecutionRequest) -> ToolExecutionRequest:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_tool_execution_request(request)
        index = self._request_index(state, validated.request_id)
        if index is None:
            raise ToolStorageError("Cortana: Request was not found.")
        state.requests[index] = validated
        self._persist(state)
        return validated

    def pending_approval_count(self) -> int:
        return sum(
            1
            for request in self._ensure_loaded().requests
            if request.request_status == "awaiting-approval"
        )

    def list_approvals(self) -> list[ToolApprovalRecord]:
        return list(self._ensure_loaded().approvals)

    def get_approval_for_request(
        self,
        request_id: str,
    ) -> ToolApprovalRecord | None:
        matches = [
            approval
            for approval in self._ensure_loaded().approvals
            if approval.request_id == request_id
        ]
        if not matches:
            return None
        return matches[-1]

    def add_approval(self, approval: ToolApprovalRecord) -> ToolApprovalRecord:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_tool_approval(approval)
        if any(item.approval_id == validated.approval_id for item in state.approvals):
            raise ToolStorageError("Cortana: Duplicate approval ID rejected.")
        request = self._find_request(state, validated.request_id)
        if request is None:
            raise ToolStorageError("Cortana: Referenced request was not found.")
        if validated.request_fingerprint != request.fingerprint:
            raise ToolStorageError(
                "Cortana: Approval fingerprint does not match the request."
            )
        state.approvals.append(validated)
        state.audit_entries.append(
            create_tool_audit_entry(
                action="approval-recorded",
                request_id=validated.request_id,
                approval_id=validated.approval_id,
                tool_id=request.tool_id,
                safe_details={"decision": validated.decision},
            )
        )
        self._persist(state)
        return validated

    def list_results(self) -> list[ToolExecutionResult]:
        return list(self._ensure_loaded().results)

    def get_result(self, result_id: str) -> ToolExecutionResult | None:
        for result in self._ensure_loaded().results:
            if result.result_id == result_id:
                return result
        return None

    def add_result(self, result: ToolExecutionResult) -> ToolExecutionResult:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_tool_execution_result(result)
        if any(item.result_id == validated.result_id for item in state.results):
            raise ToolStorageError("Cortana: Duplicate result ID rejected.")
        if self._find_request(state, validated.request_id) is None:
            raise ToolStorageError("Cortana: Referenced request was not found.")
        state.results.append(validated)

        action = "dry-run-generated" if validated.dry_run else None
        if action is None:
            if validated.outcome == "succeeded":
                action = "execution-succeeded"
            elif validated.outcome in {
                "failed",
                "timed_out_terminated",
                "termination_unconfirmed",
                "resource_limit_exceeded",
            }:
                action = "execution-failed"
            elif validated.outcome == "denied":
                action = "execution-denied"
            elif validated.outcome == "cancelled":
                action = "request-cancelled"
            else:
                action = "execution-started"
        state.audit_entries.append(
            create_tool_audit_entry(
                action=action,
                request_id=validated.request_id,
                tool_id=validated.tool_id,
                result_id=validated.result_id,
                safe_details={
                    "outcome": validated.outcome,
                    "dry_run": validated.dry_run,
                    "output_truncated": validated.output_truncated,
                },
            )
        )
        self._persist(state)
        return validated

    def list_audit_entries(self) -> list[ToolAuditEntry]:
        return list(self._ensure_loaded().audit_entries)

    def append_audit_entry(self, entry: ToolAuditEntry) -> ToolAuditEntry:
        state = self._clone_state(self._ensure_loaded())
        validated = validate_tool_audit_entry(entry)
        if any(item.audit_id == validated.audit_id for item in state.audit_entries):
            raise ToolStorageError("Cortana: Duplicate audit ID rejected.")
        state.audit_entries.append(validated)
        self._persist(state)
        return validated

    def _ensure_loaded(self) -> _ToolRepositoryState:
        if self._load_error is not None:
            raise self._load_error
        if self._state is not None:
            return self._state

        if not self._file_path.exists():
            self._state = _ToolRepositoryState()
            return self._state

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                raise ToolStorageError
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise ToolStorageError
            state = self._deserialize_state(payload)
            self._validate_referential_integrity(state)
            self._state = state
            return self._state
        except ToolStorageError as error:
            self._load_error = error
            raise
        except (OSError, json.JSONDecodeError, TypeError, ValueError, ToolValidationError) as error:
            logger.error(
                "Tool-control repository load failed error_type=%s",
                type(error).__name__,
            )
            self._load_error = ToolStorageError()
            raise self._load_error from error

    def _deserialize_state(self, payload: dict[str, Any]) -> _ToolRepositoryState:
        version = payload.get("version")
        if version != TOOL_CONTROL_REPOSITORY_SCHEMA_VERSION:
            logger.error("Tool-control repository has unsupported schema version.")
            raise ToolStorageError

        scopes = [
            validate_authorized_scope(_deserialize_scope(item))
            for item in _require_list(payload, "scopes")
        ]
        requests = [
            validate_tool_execution_request(_deserialize_request(item))
            for item in _require_list(payload, "requests")
        ]
        approvals = [
            validate_tool_approval(_deserialize_approval(item))
            for item in _require_list(payload, "approvals")
        ]
        results = [
            validate_tool_execution_result(_deserialize_result(item))
            for item in _require_list(payload, "results")
        ]
        audit_entries = [
            validate_tool_audit_entry(_deserialize_audit(item))
            for item in _require_list(payload, "audit_entries")
        ]
        return _ToolRepositoryState(
            scopes=scopes,
            requests=requests,
            approvals=approvals,
            results=results,
            audit_entries=audit_entries,
        )

    def _validate_referential_integrity(self, state: _ToolRepositoryState) -> None:
        scope_ids = {scope.scope_id for scope in state.scopes}
        request_ids = {request.request_id for request in state.requests}
        approval_ids = {approval.approval_id for approval in state.approvals}
        result_ids = {result.result_id for result in state.results}
        audit_ids = {entry.audit_id for entry in state.audit_entries}

        if len(scope_ids) != len(state.scopes):
            raise ToolStorageError
        if len(request_ids) != len(state.requests):
            raise ToolStorageError
        if len(approval_ids) != len(state.approvals):
            raise ToolStorageError
        if len(result_ids) != len(state.results):
            raise ToolStorageError
        if len(audit_ids) != len(state.audit_entries):
            raise ToolStorageError

        for request in state.requests:
            if request.scope_id not in scope_ids:
                logger.error("Tool-control repository has dangling request scope.")
                raise ToolStorageError

        for approval in state.approvals:
            if approval.request_id not in request_ids:
                logger.error("Tool-control repository has dangling approval request.")
                raise ToolStorageError

        for result in state.results:
            if result.request_id not in request_ids:
                logger.error("Tool-control repository has dangling result request.")
                raise ToolStorageError

    def _persist(self, state: _ToolRepositoryState) -> None:
        self._validate_referential_integrity(state)
        payload = {
            "version": TOOL_CONTROL_REPOSITORY_SCHEMA_VERSION,
            "scopes": [_serialize_scope(scope) for scope in state.scopes],
            "requests": [_serialize_request(request) for request in state.requests],
            "approvals": [
                _serialize_approval(approval) for approval in state.approvals
            ],
            "results": [_serialize_result(result) for result in state.results],
            "audit_entries": [
                _serialize_audit(entry) for entry in state.audit_entries
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(serialized)
        self._state = self._clone_state(state)

    def _atomic_write(self, content: str) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".tool-control-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._file_path)
            temporary_path = None
        except OSError as error:
            logger.error(
                "Unable to write tool-control repository due to OS error type: %s",
                type(error).__name__,
            )
            raise ToolStorageError(TOOL_SAVE_ERROR_MESSAGE) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _clone_state(self, state: _ToolRepositoryState) -> _ToolRepositoryState:
        return _ToolRepositoryState(
            scopes=list(state.scopes),
            requests=list(state.requests),
            approvals=list(state.approvals),
            results=list(state.results),
            audit_entries=list(state.audit_entries),
        )

    def _scope_index(self, state: _ToolRepositoryState, scope_id: str) -> int | None:
        for index, scope in enumerate(state.scopes):
            if scope.scope_id == scope_id:
                return index
        return None

    def _request_index(
        self,
        state: _ToolRepositoryState,
        request_id: str,
    ) -> int | None:
        for index, request in enumerate(state.requests):
            if request.request_id == request_id:
                return index
        return None

    def _find_scope(
        self,
        state: _ToolRepositoryState,
        scope_id: str,
    ) -> AuthorizedScope | None:
        for scope in state.scopes:
            if scope.scope_id == scope_id:
                return scope
        return None

    def _find_request(
        self,
        state: _ToolRepositoryState,
        request_id: str,
    ) -> ToolExecutionRequest | None:
        for request in state.requests:
            if request.request_id == request_id:
                return request
        return None


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ToolStorageError
    return value


def _require_dict(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ToolStorageError
    return item


def _deserialize_scope(item: Any) -> AuthorizedScope:
    data = _require_dict(item)
    return AuthorizedScope(
        scope_id=str(data["scope_id"]),
        scope_name=str(data["scope_name"]),
        allowed_tool_ids=tuple(str(value) for value in data["allowed_tool_ids"]),
        allowed_target_types=cast(
            tuple[ToolTargetType, ...],
            tuple(str(value) for value in data["allowed_target_types"]),
        ),
        allowed_local_path_roots=tuple(
            str(value) for value in data["allowed_local_path_roots"]
        ),
        allowed_incident_ids=tuple(
            str(value) for value in data["allowed_incident_ids"]
        ),
        expires_at=data.get("expires_at"),
        created_at=str(data["created_at"]),
        active=bool(data["active"]),
        notes=data.get("notes"),
    )


def _deserialize_request(item: Any) -> ToolExecutionRequest:
    data = _require_dict(item)
    parameters = data["normalized_parameters"]
    if not isinstance(parameters, dict):
        raise ToolStorageError
    return ToolExecutionRequest(
        request_id=str(data["request_id"]),
        tool_id=str(data["tool_id"]),
        normalized_parameters=dict(parameters),
        scope_id=str(data["scope_id"]),
        requested_by=str(data["requested_by"]),
        requested_timestamp=str(data["requested_timestamp"]),
        dry_run_requested=bool(data["dry_run_requested"]),
        incident_id=data.get("incident_id"),
        justification=str(data["justification"]),
        request_status=str(data["request_status"]),  # type: ignore[arg-type]
        fingerprint=str(data["fingerprint"]),
        dry_run_completed=bool(data.get("dry_run_completed", False)),
    )


def _deserialize_approval(item: Any) -> ToolApprovalRecord:
    data = _require_dict(item)
    return ToolApprovalRecord(
        approval_id=str(data["approval_id"]),
        request_id=str(data["request_id"]),
        decision=str(data["decision"]),  # type: ignore[arg-type]
        approver=str(data["approver"]),
        decision_timestamp=str(data["decision_timestamp"]),
        reason=str(data["reason"]),
        request_fingerprint=str(data["request_fingerprint"]),
        expires_at=data.get("expires_at"),
    )


def _deserialize_result(item: Any) -> ToolExecutionResult:
    data = _require_dict(item)
    structured = data["structured_data"]
    if not isinstance(structured, dict):
        raise ToolStorageError
    return ToolExecutionResult(
        result_id=str(data["result_id"]),
        request_id=str(data["request_id"]),
        tool_id=str(data["tool_id"]),
        started_timestamp=str(data["started_timestamp"]),
        completed_timestamp=str(data["completed_timestamp"]),
        outcome=str(data["outcome"]),  # type: ignore[arg-type]
        safe_summary=str(data["safe_summary"]),
        structured_data=dict(structured),
        output_truncated=bool(data["output_truncated"]),
        error_class=data.get("error_class"),
        dry_run=bool(data["dry_run"]),
        incident_id=data.get("incident_id"),
    )


def _deserialize_audit(item: Any) -> ToolAuditEntry:
    data = _require_dict(item)
    details = data["safe_details"]
    if not isinstance(details, dict):
        raise ToolStorageError
    return ToolAuditEntry(
        audit_id=str(data["audit_id"]),
        timestamp=str(data["timestamp"]),
        action=str(data["action"]),  # type: ignore[arg-type]
        actor=str(data["actor"]),
        request_id=data.get("request_id"),
        tool_id=data.get("tool_id"),
        scope_id=data.get("scope_id"),
        approval_id=data.get("approval_id"),
        result_id=data.get("result_id"),
        safe_details=dict(details),
    )


def _serialize_scope(scope: AuthorizedScope) -> dict[str, Any]:
    return {
        "scope_id": scope.scope_id,
        "scope_name": scope.scope_name,
        "allowed_tool_ids": list(scope.allowed_tool_ids),
        "allowed_target_types": list(scope.allowed_target_types),
        "allowed_local_path_roots": list(scope.allowed_local_path_roots),
        "allowed_incident_ids": list(scope.allowed_incident_ids),
        "expires_at": scope.expires_at,
        "created_at": scope.created_at,
        "active": scope.active,
        "notes": scope.notes,
    }


def _serialize_request(request: ToolExecutionRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "tool_id": request.tool_id,
        "normalized_parameters": dict(request.normalized_parameters),
        "scope_id": request.scope_id,
        "requested_by": request.requested_by,
        "requested_timestamp": request.requested_timestamp,
        "dry_run_requested": request.dry_run_requested,
        "incident_id": request.incident_id,
        "justification": request.justification,
        "request_status": request.request_status,
        "fingerprint": request.fingerprint,
        "dry_run_completed": request.dry_run_completed,
    }


def _serialize_approval(approval: ToolApprovalRecord) -> dict[str, Any]:
    return {
        "approval_id": approval.approval_id,
        "request_id": approval.request_id,
        "decision": approval.decision,
        "approver": approval.approver,
        "decision_timestamp": approval.decision_timestamp,
        "reason": approval.reason,
        "request_fingerprint": approval.request_fingerprint,
        "expires_at": approval.expires_at,
    }


def _serialize_result(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "request_id": result.request_id,
        "tool_id": result.tool_id,
        "started_timestamp": result.started_timestamp,
        "completed_timestamp": result.completed_timestamp,
        "outcome": result.outcome,
        "safe_summary": result.safe_summary,
        "structured_data": dict(result.structured_data),
        "output_truncated": result.output_truncated,
        "error_class": result.error_class,
        "dry_run": result.dry_run,
        "incident_id": result.incident_id,
    }


def _serialize_audit(entry: ToolAuditEntry) -> dict[str, Any]:
    return {
        "audit_id": entry.audit_id,
        "timestamp": entry.timestamp,
        "action": entry.action,
        "actor": entry.actor,
        "request_id": entry.request_id,
        "tool_id": entry.tool_id,
        "scope_id": entry.scope_id,
        "approval_id": entry.approval_id,
        "result_id": entry.result_id,
        "safe_details": dict(entry.safe_details),
    }
