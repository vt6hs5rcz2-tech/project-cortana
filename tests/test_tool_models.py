"""Tests for defensive tool definitions, registry, requests, and approvals."""

from __future__ import annotations

import pytest

from src.tool_approval import (
    assert_approval_allows_execution,
    create_tool_approval,
)
from src.tool_common import (
    ToolAuthorizationError,
    ToolParameterError,
    ToolValidationError,
    compute_request_fingerprint,
    fingerprint_display_prefix,
)
from src.tool_definition import (
    FORBIDDEN_PARAMETER_NAMES,
    create_parameter_definition,
    create_tool_definition,
    validate_parameters_against_schema,
)
from src.tool_policy import approval_required, assert_requestable, initial_request_status
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_request import (
    create_tool_execution_request,
    replace_tool_execution_request,
)
from src.tool_scope import create_authorized_scope


def test_tool_definitions_are_immutable() -> None:
    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    with pytest.raises(Exception):
        definition.name = "mutated"  # type: ignore[misc]


def test_parameter_schema_rejects_command_fields() -> None:
    for name in ("command", "shell", "powershell", "bash"):
        assert name in FORBIDDEN_PARAMETER_NAMES
        with pytest.raises(ToolValidationError):
            create_parameter_definition(
                name=name,
                parameter_type="string",
                required=True,
                description="bad",
            )


def test_registry_rejects_duplicate_ids_and_implementations() -> None:
    registry = build_default_tool_registry()
    first = registry.require("system-summary")
    with pytest.raises(ToolValidationError):
        registry.register(first)

    other = create_tool_definition(
        tool_id="custom-summary",
        name="Custom",
        description="Custom tool",
        category="diagnostics",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("none",),
        parameter_schema=(),
        requires_approval=False,
        implementation_identifier="impl_system_summary",
    )
    with pytest.raises(ToolValidationError):
        registry.register(other)


def test_registry_ordering_and_filters_are_deterministic() -> None:
    registry = build_default_tool_registry()
    ids = [item.tool_id for item in registry.list_all()]
    assert ids == sorted(ids, key=ids.index)
    assert [item.tool_id for item in registry.list_enabled()] == ids
    file_tools = registry.filter(category="file-integrity")
    assert {item.tool_id for item in file_tools} == {"file-sha256", "compare-sha256"}


def test_prohibited_and_future_external_cannot_be_enabled() -> None:
    with pytest.raises(ToolValidationError):
        create_tool_definition(
            tool_id="bad-tool",
            name="Bad",
            description="Prohibited",
            category="diagnostics",
            version="1.0.0",
            risk_level="prohibited",
            execution_mode="internal-python",
            supported_objective_types=("inspect",),
            supported_target_types=("none",),
            parameter_schema=(),
            requires_approval=True,
            enabled=True,
            implementation_identifier="impl_bad_tool",
        )
    with pytest.raises(ToolValidationError):
        create_tool_definition(
            tool_id="future-tool",
            name="Future",
            description="Future external",
            category="diagnostics",
            version="1.0.0",
            risk_level="low",
            execution_mode="future-external",
            supported_objective_types=("inspect",),
            supported_target_types=("none",),
            parameter_schema=(),
            requires_approval=True,
            enabled=True,
            implementation_identifier="impl_future_tool",
        )


def test_parameter_validation_unknown_and_missing() -> None:
    schema = (
        create_parameter_definition(
            name="path",
            parameter_type="file-path",
            required=True,
            description="path",
        ),
    )
    with pytest.raises(ToolParameterError):
        validate_parameters_against_schema({"path": "a", "extra": "b"}, schema)
    with pytest.raises(ToolParameterError):
        validate_parameters_against_schema({}, schema)
    with pytest.raises(ToolParameterError):
        validate_parameters_against_schema({"path": 123}, schema)


def test_fingerprint_is_deterministic_and_sensitive_to_changes() -> None:
    first = compute_request_fingerprint(
        request_id="11111111-1111-1111-1111-111111111111",
        tool_id="file-sha256",
        normalized_parameters={"path": "a.txt"},
        scope_id="22222222-2222-2222-2222-222222222222",
        incident_id=None,
        dry_run_requested=False,
        justification="because",
    )
    second = compute_request_fingerprint(
        request_id="11111111-1111-1111-1111-111111111111",
        tool_id="file-sha256",
        normalized_parameters={"path": "a.txt"},
        scope_id="22222222-2222-2222-2222-222222222222",
        incident_id=None,
        dry_run_requested=False,
        justification="because",
    )
    assert first == second
    changed = compute_request_fingerprint(
        request_id="11111111-1111-1111-1111-111111111111",
        tool_id="file-sha256",
        normalized_parameters={"path": "a.txt"},
        scope_id="22222222-2222-2222-2222-222222222222",
        incident_id=None,
        dry_run_requested=False,
        justification="changed",
    )
    assert changed != first
    assert len(fingerprint_display_prefix(first)) == 12


def test_request_status_transitions_and_required_justification() -> None:
    with pytest.raises(ToolValidationError):
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id="33333333-3333-3333-3333-333333333333",
            justification="   ",
        )
    request = create_tool_execution_request(
        tool_id="system-summary",
        normalized_parameters={},
        scope_id="33333333-3333-3333-3333-333333333333",
        justification="Need summary",
        request_status="drafted",
    )
    approved = replace_tool_execution_request(request, request_status="approved")
    assert approved.request_status == "approved"
    with pytest.raises(Exception):
        replace_tool_execution_request(approved, request_status="drafted")


def test_approval_binds_fingerprint_and_rejects_replay_mismatch() -> None:
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": "a.txt"},
        scope_id="44444444-4444-4444-4444-444444444444",
        justification="hash file",
        request_status="awaiting-approval",
    )
    approval = create_tool_approval(
        request=request,
        decision="approved",
        reason="looks good",
    )
    approved_request = replace_tool_execution_request(
        request,
        request_status="approved",
    )
    assert_approval_allows_execution(approval, approved_request)

    mutated = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": "b.txt"},
        scope_id="44444444-4444-4444-4444-444444444444",
        justification="hash file",
        request_status="approved",
    )
    # Force same request id/fingerprint mismatch path via approval binding.
    with pytest.raises(ToolAuthorizationError):
        assert_approval_allows_execution(approval, mutated)


def test_rejected_request_cannot_execute() -> None:
    request = create_tool_execution_request(
        tool_id="file-sha256",
        normalized_parameters={"path": "a.txt"},
        scope_id="55555555-5555-5555-5555-555555555555",
        justification="hash file",
        request_status="awaiting-approval",
    )
    approval = create_tool_approval(
        request=request,
        decision="rejected",
        reason="not now",
    )
    rejected = replace_tool_execution_request(request, request_status="rejected")
    with pytest.raises(ToolAuthorizationError):
        assert_approval_allows_execution(approval, rejected)


def test_policy_for_builtin_tools() -> None:
    registry = build_default_tool_registry()
    summary = registry.require("system-summary")
    file_tool = registry.require("file-sha256")
    assert approval_required(summary) is False
    assert approval_required(file_tool) is True
    assert initial_request_status(summary) == "drafted"
    assert initial_request_status(file_tool) == "awaiting-approval"
    assert_requestable(summary)
    for definition in registry.list_all():
        assert definition.capability_class == "internal-readonly"


def test_empty_scope_tool_list_authorizes_nothing() -> None:
    scope = create_authorized_scope(
        scope_name="Empty",
        allowed_tool_ids=[],
        allowed_target_types=["none"],
    )
    from src.tool_scope import assert_tool_authorized

    with pytest.raises(ToolAuthorizationError):
        assert_tool_authorized(scope, "system-summary")


def test_no_blanket_approval_helper_exists() -> None:
    import src.tool_approval as approval_module

    assert not hasattr(approval_module, "approve_all")
    assert not hasattr(approval_module, "blanket_approve")
