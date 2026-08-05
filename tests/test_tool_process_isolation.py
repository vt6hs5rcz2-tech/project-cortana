"""Tests for Milestone 13 process-isolated defensive tool execution."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.config import (
    PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED,
    PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED,
)
from src.tool_definition import create_parameter_definition, create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_process_adapter import ToolProcessAdapter
from src.tool_process_common import (
    PROCESS_SAFE_IMPLEMENTATION_IDS,
    PROCESS_SAFE_TOOL_IDS,
    ToolProcessError,
    assert_process_isolation_tables_match,
    build_child_environment,
    registry_process_isolation_implementation_ids,
)
from src.tool_process_envelope import (
    create_process_execution_request,
    create_process_execution_response,
    decode_process_request,
    decode_process_response,
    encode_process_request,
    encode_process_response,
    validate_process_execution_request,
)
from src.tool_process_runner import CHILD_DISPATCH_IMPLEMENTATION_IDS, run_child
from src.tool_registry import build_default_tool_registry
from src.tool_request import (
    create_tool_execution_request,
    replace_tool_execution_request,
)
from tests.tool_helpers import make_scope, tool_repository


def _enable_process_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.config.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.config.PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED",
        True,
    )


def _ready_request(
    tmp_path: Path,
    *,
    tool_id: str = "system-summary",
    parameters: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=[tool_id]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id=tool_id,
            normalized_parameters=parameters or {},
            scope_id=scope.scope_id,
            justification="process isolation coverage",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    repo.update_request(running)
    return repo, scope, running


def test_flags_default_disabled() -> None:
    assert PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED is False
    assert PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED is False


def test_process_isolation_values_validated() -> None:
    with pytest.raises(Exception):
        create_tool_definition(
            tool_id="bad-isolation",
            name="Bad",
            description="Invalid isolation value",
            category="diagnostics",
            version="1.0.0",
            risk_level="informational",
            execution_mode="internal-python",
            supported_objective_types=("inspect",),
            supported_target_types=("none",),
            parameter_schema=(),
            requires_approval=False,
            implementation_identifier="impl_system_summary",
            process_isolation="maybe",
        )


def test_file_tools_cannot_be_process_eligible() -> None:
    with pytest.raises(Exception):
        create_tool_definition(
            tool_id="file-isolated",
            name="File Isolated",
            description="Must remain prohibited",
            category="file-integrity",
            version="1.0.0",
            risk_level="low",
            execution_mode="internal-python",
            supported_objective_types=("hash",),
            supported_target_types=("local-file",),
            parameter_schema=(
                create_parameter_definition(
                    name="path",
                    parameter_type="file-path",
                    required=True,
                    description="path",
                ),
            ),
            requires_approval=True,
            implementation_identifier="impl_system_summary",
            process_isolation="eligible",
        )


def test_registry_and_child_dispatch_match_both_directions() -> None:
    registry = build_default_tool_registry()
    eligible = {
        definition.tool_id
        for definition in registry.list_all()
        if definition.process_isolation in {"eligible", "required"}
    }
    assert eligible == PROCESS_SAFE_TOOL_IDS
    registry_ids = registry_process_isolation_implementation_ids(registry.list_all())
    assert registry_ids == CHILD_DISPATCH_IMPLEMENTATION_IDS
    assert registry_ids == PROCESS_SAFE_IMPLEMENTATION_IDS
    assert_process_isolation_tables_match(
        registry_implementation_ids=registry_ids,
        child_dispatch_ids=CHILD_DISPATCH_IMPLEMENTATION_IDS,
    )
    for definition in registry.list_all():
        if definition.tool_id in {"file-sha256", "compare-sha256", "text-search"}:
            assert definition.process_isolation == "eligible"


def test_envelope_exact_keys_and_bounds() -> None:
    correlation = str(uuid4())
    request = create_process_execution_request(
        correlation_id=correlation,
        implementation_identifier="impl_system_summary",
        normalized_parameters={},
        execution_timeout_seconds=5,
        max_output_characters=1000,
    )
    raw = encode_process_request(request)
    decoded = decode_process_request(raw)
    assert decoded.correlation_id == correlation

    with pytest.raises(ToolProcessError):
        validate_process_execution_request(
            {
                "schema_version": 1,
                "correlation_id": correlation,
                "implementation_identifier": "impl_system_summary",
                "normalized_parameters": {},
                "execution_timeout_seconds": 5,
                "max_output_characters": 1000,
                "extra": True,
            }
        )

    with pytest.raises(ToolProcessError):
        decode_process_request(b"{not-json")

    with pytest.raises(ToolProcessError):
        create_process_execution_request(
            correlation_id=correlation,
            implementation_identifier="impl_file_sha256",
            normalized_parameters={},
            execution_timeout_seconds=5,
            max_output_characters=1000,
        )

    with pytest.raises(ToolProcessError):
        create_process_execution_request(
            correlation_id=correlation,
            implementation_identifier="impl_text_search",
            normalized_parameters={},
            execution_timeout_seconds=5,
            max_output_characters=1000,
        )


def test_response_correlation_and_extra_data_rejected() -> None:
    correlation = str(uuid4())
    response = create_process_execution_response(
        correlation_id=correlation,
        outcome="succeeded",
        structured_data={"ok": True},
        safe_summary="ok",
    )
    raw = encode_process_response(response)
    assert decode_process_response(
        raw,
        expected_correlation_id=correlation,
    ).outcome == "succeeded"

    with pytest.raises(Exception):
        decode_process_response(
            raw,
            expected_correlation_id=str(uuid4()),
        )

    with pytest.raises(Exception):
        decode_process_response(
            raw + b"\n{}",
            expected_correlation_id=correlation,
        )

    with pytest.raises(Exception):
        decode_process_response(b"", expected_correlation_id=correlation)


def test_no_pickle_multiprocessing_in_process_modules() -> None:
    modules = [
        Path("src/tool_process_adapter.py"),
        Path("src/tool_process_runner.py"),
        Path("src/tool_process_envelope.py"),
        Path("src/tool_process_common.py"),
        Path("src/tool_process_callables.py"),
        Path("src/tool_executor.py"),
    ]
    forbidden = ("pickle", "dill", "cloudpickle", "multiprocessing")
    for path in modules:
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden


def test_child_environment_excludes_sentinel_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("CORTANA_SENTINEL_SECRET", "parent-only-secret")
    monkeypatch.setenv("CORTANA_INCIDENT_REPO", r"C:\secret\incidents.json")
    env = build_child_environment()
    assert "OPENAI_API_KEY" not in env
    assert "CORTANA_SENTINEL_SECRET" not in env
    assert "CORTANA_INCIDENT_REPO" not in env
    assert env.get("PYTHONUTF8") == "1"
    assert "PYTHONPATH" in env
    joined = json.dumps(env)
    assert "sk-test-secret" not in joined
    assert "parent-only-secret" not in joined


def test_child_runner_dispatch_and_result_file(tmp_path: Path) -> None:
    correlation = str(uuid4())
    request = create_process_execution_request(
        correlation_id=correlation,
        implementation_identifier="impl_system_summary",
        normalized_parameters={},
        execution_timeout_seconds=5,
        max_output_characters=4000,
    )
    result_path = tmp_path / "result.json"
    stdin_bytes = encode_process_request(request)

    class _FakeStdin:
        def __init__(self, data: bytes) -> None:
            self.buffer = __import__("io").BytesIO(data)

    old_stdin = sys.stdin
    try:
        sys.stdin = _FakeStdin(stdin_bytes)
        code = run_child(result_path)
    finally:
        sys.stdin = old_stdin
    assert code == 0
    response = decode_process_response(
        result_path.read_bytes(),
        expected_correlation_id=correlation,
    )
    assert response.outcome == "succeeded"
    assert "os_family" in response.structured_data


def test_eligible_fallback_in_process_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        False,
    )
    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    assert definition.process_isolation == "eligible"
    _, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor()
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "succeeded"
    assert result.structured_data.get("process_isolated") is not True


def test_required_fails_when_isolation_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        False,
    )
    definition = create_tool_definition(
        tool_id="system-summary",
        name="System Summary",
        description="Required isolation test definition",
        category="system-information",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("system-summary", "none"),
        parameter_schema=(),
        requires_approval=False,
        implementation_identifier="impl_system_summary",
        process_isolation="required",
    )
    _, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor()
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "denied"
    assert result.error_class == "ToolPolicyError"


def test_process_execution_parity_system_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)
    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    repo, scope, running = _ready_request(tmp_path)
    adapter = ToolProcessAdapter(
        scratch_dir=tmp_path / "scratch",
        audit_appender=repo.append_audit_entry,
    )
    executor = DefensiveToolExecutor(process_adapter=adapter)
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "succeeded"
    assert result.structured_data.get("process_isolated") is True
    assert "os_family" in result.structured_data
    assert "python_version" in result.structured_data
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_execution_started" in actions
    assert "process_execution_completed" in actions


def test_process_execution_parity_simulated_log_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)
    registry = build_default_tool_registry()
    definition = registry.require("simulated-log-check")
    _, scope, running = _ready_request(
        tmp_path,
        tool_id="simulated-log-check",
        parameters={"fixture": "auth-noise"},
    )
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(scratch_dir=tmp_path / "scratch"),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "succeeded"
    assert result.structured_data.get("simulation") is True
    assert result.structured_data.get("process_isolated") is True


def test_dry_run_does_not_launch_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)
    launched = {"count": 0}

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        launched["count"] += 1
        raise AssertionError("dry-run must not launch a child")

    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="dry-run no child",
            request_status="drafted",
            dry_run_completed=False,
        )
    )
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(popen_factory=boom),
    )
    try:
        result = executor.plan_dry_run(
            definition=definition,
            request=request,
            scope=scope,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "planned"
    assert result.dry_run is True
    assert launched["count"] == 0


def test_stdout_noise_cannot_spoof_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)
    correlation_holder: dict[str, str] = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 4242
            self.returncode: int | None = None
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self._result_path = ""

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            del timeout
            assert input is not None
            request = decode_process_request(input)
            correlation_holder["id"] = request.correlation_id
            # Write a valid result channel document.
            response = create_process_execution_response(
                correlation_id=request.correlation_id,
                outcome="succeeded",
                structured_data={"from_channel": True},
                safe_summary="channel ok",
            )
            # Discover result path from the command argv stored on factory.
            result_path = Path(self._result_path)
            result_path.write_bytes(encode_process_response(response))
            spoof = json.dumps(
                {
                    "schema_version": 1,
                    "correlation_id": request.correlation_id,
                    "outcome": "succeeded",
                    "structured_data": {"spoofed": True},
                    "safe_summary": "spoof",
                    "output_truncated": False,
                    "error_class": None,
                }
            ).encode("utf-8")
            self.returncode = 0
            return spoof, b"stderr-noise"

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    fake = FakeProcess()

    def factory(command: list[str], **kwargs: Any) -> FakeProcess:
        assert kwargs.get("close_fds") is True
        assert kwargs.get("shell") is False
        assert isinstance(kwargs.get("env"), dict)
        assert "OPENAI_API_KEY" not in kwargs["env"]
        fake._result_path = command[-1]
        return fake

    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    _, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
        ),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "succeeded"
    assert result.structured_data.get("from_channel") is True
    assert result.structured_data.get("spoofed") is not True


def test_timeout_terminates_and_rejects_late_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)

    class SlowProcess:
        def __init__(self) -> None:
            self.pid = 99
            self.returncode: int | None = None
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self.killed = False
            self._result_path = ""

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            del input
            raise subprocess.TimeoutExpired(cmd="slow", timeout=timeout or 1)

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            # Late result written after timeout decision must be ignored.
            if self._result_path:
                late = create_process_execution_response(
                    correlation_id=str(uuid4()),
                    outcome="succeeded",
                    structured_data={"late": True},
                    safe_summary="late",
                )
                Path(self._result_path).write_bytes(encode_process_response(late))

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    slow = SlowProcess()

    def factory(command: list[str], **_kwargs: Any) -> SlowProcess:
        slow._result_path = command[-1]
        return slow

    registry = build_default_tool_registry()
    definition = create_tool_definition(
        tool_id="system-summary",
        name="System Summary",
        description="Timeout coverage",
        category="system-information",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("system-summary", "none"),
        parameter_schema=(),
        timeout_seconds=1,
        requires_approval=False,
        implementation_identifier="impl_system_summary",
        process_isolation="eligible",
    )
    repo, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            audit_appender=repo.append_audit_entry,
        ),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert slow.killed is True
    assert result.outcome == "timed_out_terminated"
    assert result.structured_data.get("termination_confirmed") is True
    assert result.structured_data.get("late") is not True
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_timeout_observed" in actions
    assert "process_timeout_terminated" in actions


def test_termination_unconfirmed_via_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)

    class HangProcess:
        def __init__(self) -> None:
            self.pid = 77
            self.returncode: int | None = None
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            del input
            raise subprocess.TimeoutExpired(cmd="hang", timeout=timeout or 1)

        def poll(self) -> int | None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired(cmd="hang", timeout=timeout or 1)

    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    repo, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=lambda *_a, **_k: HangProcess(),
            audit_appender=repo.append_audit_entry,
        ),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "termination_unconfirmed"
    assert result.structured_data.get("termination_confirmed") is False
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_termination_unconfirmed" in actions


def test_pre_launch_cancellation_does_not_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)
    launched = {"count": 0}

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        launched["count"] += 1
        raise AssertionError("cancelled request must not spawn")

    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="cancel before launch",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    cancelled = replace_tool_execution_request(request, request_status="cancelled")
    repo.update_request(cancelled)
    # Bypass assert_ready_for_execution by calling adapter directly.
    adapter = ToolProcessAdapter(
        scratch_dir=tmp_path / "scratch",
        popen_factory=factory,
        audit_appender=repo.append_audit_entry,
    )
    result = adapter.execute(definition=definition, request=cancelled)
    assert result.outcome == "cancelled"
    assert launched["count"] == 0
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_cancellation_requested" in actions


def test_keyboard_interrupt_terminates_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)

    class InterruptProcess:
        def __init__(self) -> None:
            self.pid = 55
            self.returncode: int | None = None
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self.killed = False
            self._interrupt_remaining = 1

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            del input, timeout
            if self._interrupt_remaining > 0:
                self._interrupt_remaining -= 1
                raise KeyboardInterrupt
            return b"", b""

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    proc = InterruptProcess()
    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    repo, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=lambda *_a, **_k: proc,
            audit_appender=repo.append_audit_entry,
        ),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert proc.killed is True
    assert result.outcome == "cancelled"
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_cancellation_requested" in actions
    assert "process_cancellation_terminated" in actions


def test_popen_failure_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("launch blocked")

    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    repo, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            audit_appender=repo.append_audit_entry,
        ),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "failed"
    assert result.error_class == "OSError"
    assert "launch blocked" not in result.safe_summary
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_launch_failed" in actions


def test_live_child_environment_omits_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove a real child process does not inherit a parent sentinel secret."""
    _enable_process_isolation(monkeypatch)
    monkeypatch.setenv("CORTANA_SENTINEL_SECRET", "must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")

    probe = tmp_path / "env_probe.py"
    probe.write_text(
        "import json,os,sys\n"
        "json.dump(dict(os.environ), open(sys.argv[1], 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "child_env.json"
    env = build_child_environment()
    completed = subprocess.run(
        [sys.executable, str(probe), str(out_path)],
        env=env,
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        timeout=10,
        shell=False,
    )
    assert completed.returncode == 0
    child_env = json.loads(out_path.read_text(encoding="utf-8"))
    assert "CORTANA_SENTINEL_SECRET" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert "must-not-leak" not in json.dumps(child_env)


@pytest.mark.skipif(os.name != "nt", reason="Windows handle inheritance check")
def test_windows_close_fds_true_on_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_process_isolation(monkeypatch)
    seen: dict[str, Any] = {}

    class QuickProcess:
        def __init__(self) -> None:
            self.pid = 1
            self.returncode: int | None = None
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self._result_path = ""

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            del timeout
            assert input is not None
            request = decode_process_request(input)
            response = create_process_execution_response(
                correlation_id=request.correlation_id,
                outcome="succeeded",
                structured_data={"ok": True},
                safe_summary="ok",
            )
            Path(self._result_path).write_bytes(encode_process_response(response))
            self.returncode = 0
            return b"", b""

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    proc = QuickProcess()

    def factory(command: list[str], **kwargs: Any) -> QuickProcess:
        seen.update(kwargs)
        proc._result_path = command[-1]
        return proc

    registry = build_default_tool_registry()
    definition = registry.require("system-summary")
    _, scope, running = _ready_request(tmp_path)
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
        ),
    )
    try:
        result = executor.execute(
            definition=definition,
            request=running,
            scope=scope,
            approval=None,
        )
    finally:
        executor.shutdown()
    assert result.outcome == "succeeded"
    assert seen.get("close_fds") is True
    assert seen.get("shell") is False


def test_child_runner_source_has_no_forbidden_imports() -> None:
    source = Path("src/tool_process_runner.py").read_text(encoding="utf-8")
    assert "ToolRegistry" not in source
    assert "incident_repository" not in source
    assert "subprocess" not in source
    assert "openai" not in source.lower()
    assert "workflow" not in source
    assert "eval(" not in source
    assert "exec(" not in source


def test_real_process_timeout_kills_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a deliberately slow child is terminated and no longer running."""
    _enable_process_isolation(monkeypatch)

    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import sys, time\n"
        "time.sleep(30)\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write('late')\n",
        encoding="utf-8",
    )

    class TrackingPopen(subprocess.Popen):  # type: ignore[type-arg]
        pass

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        # Replace the module runner with a sleeper while preserving result path.
        result_path = command[-1]
        return subprocess.Popen(
            [sys.executable, str(sleeper), result_path],
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            env=kwargs["env"],
            cwd=kwargs["cwd"],
            close_fds=kwargs["close_fds"],
            shell=False,
        )

    definition = create_tool_definition(
        tool_id="system-summary",
        name="System Summary",
        description="Real timeout coverage",
        category="system-information",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("system-summary", "none"),
        parameter_schema=(),
        timeout_seconds=1,
        requires_approval=False,
        implementation_identifier="impl_system_summary",
        process_isolation="eligible",
    )
    _, scope, running = _ready_request(tmp_path)
    adapter = ToolProcessAdapter(
        scratch_dir=tmp_path / "scratch",
        popen_factory=factory,
    )
    started = time.monotonic()
    result = adapter.execute(definition=definition, request=running)
    elapsed = time.monotonic() - started
    assert elapsed < 10
    assert result.outcome in {"timed_out_terminated", "termination_unconfirmed"}
    assert result.structured_data.get("timeout") is True
