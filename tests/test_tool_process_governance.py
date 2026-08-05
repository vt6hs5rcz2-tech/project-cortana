"""Tests for Milestone 14 process resource governance (Job Objects)."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.config import (
    MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES,
    PROCESS_JOB_ACTIVE_PROCESS_LIMIT,
    PROCESS_RESOURCE_LIMITS_ENABLED,
)
from src.tool_definition import create_tool_definition
from src.tool_executor import DefensiveToolExecutor
from src.tool_process_adapter import ToolProcessAdapter
from src.tool_process_envelope import (
    create_process_execution_response,
    decode_process_request,
    encode_process_response,
)
from src.tool_process_job import (
    JobLimitSnapshot,
    ToolProcessJobError,
    ToolProcessJobUnavailableError,
    UnavailableJobController,
    WindowsJobController,
    create_default_job_controller,
    default_job_memory_limit_bytes,
    validate_active_process_limit,
    validate_job_memory_limit,
)
from src.tool_registry import build_default_tool_registry
from src.tool_request import (
    create_tool_execution_request,
    replace_tool_execution_request,
)
from src.tool_result import create_tool_execution_result
from src.workflow_executor import _map_execute_step_status
from tests.tool_helpers import make_scope, tool_repository


def _enable_isolation_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.config.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED", True)
    monkeypatch.setattr("src.config.PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED", True)
    monkeypatch.setattr("src.config.PROCESS_RESOURCE_LIMITS_ENABLED", True)
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_RESOURCE_LIMITS_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_job.PROCESS_RESOURCE_LIMITS_ENABLED",
        True,
    )


def _ready_request(tmp_path: Path) -> tuple[Any, Any, Any]:
    repo = tool_repository(tmp_path)
    scope = repo.add_scope(make_scope(tmp_path, tool_ids=["system-summary"]))
    request = repo.add_request(
        create_tool_execution_request(
            tool_id="system-summary",
            normalized_parameters={},
            scope_id=scope.scope_id,
            justification="resource governance coverage",
            request_status="drafted",
            dry_run_completed=True,
        )
    )
    running = replace_tool_execution_request(request, request_status="running")
    repo.update_request(running)
    return repo, scope, running


@dataclass
class FakeJobSession:
    memory_limit_bytes: int
    active_process_limit: int = 1
    assigned_pid: int | None = None
    closed: bool = False
    terminate_calls: int = 0
    force_limit_exceeded: bool = False
    active_count: int = 1
    peak_memory: int = 0
    fail_assign: bool = False
    fail_verify_after_assign: bool = False

    def assign(self, pid: int) -> None:
        if self.fail_assign:
            raise ToolProcessJobError("assignment failed")
        self.assigned_pid = pid
        self.active_count = 1

    def verified_limits(self) -> JobLimitSnapshot:
        if self.fail_verify_after_assign and self.assigned_pid is not None:
            raise ToolProcessJobError("verification failed")
        return JobLimitSnapshot(
            active_process_limit=self.active_process_limit,
            memory_limit_bytes=self.memory_limit_bytes,
            kill_on_job_close=True,
            limits_verified=True,
            assignment_confirmed=self.assigned_pid is not None,
            active_process_count=self.active_count if self.assigned_pid else 0,
            peak_job_memory_used=self.peak_memory,
            total_terminated_processes=0,
        )

    def terminate_tree(self, exit_code: int = 1) -> bool:
        del exit_code
        self.terminate_calls += 1
        self.active_count = 0
        return True

    def active_process_count(self) -> int:
        return self.active_count

    def peak_job_memory_used(self) -> int:
        return self.peak_memory

    def likely_resource_limit_exceeded(self, *, configured_memory_bytes: int) -> bool:
        if self.force_limit_exceeded:
            return True
        return self.peak_memory >= configured_memory_bytes

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeJobController:
    available: bool = True
    unavailable_reason: str | None = None
    session: FakeJobSession | None = None
    create_calls: int = 0
    fail_create: bool = False

    def create_session(
        self,
        *,
        memory_limit_bytes: int,
        active_process_limit: int = 1,
    ) -> FakeJobSession:
        self.create_calls += 1
        if not self.available or self.fail_create:
            raise ToolProcessJobUnavailableError("unavailable")
        self.session = FakeJobSession(
            memory_limit_bytes=memory_limit_bytes,
            active_process_limit=active_process_limit,
        )
        return self.session


@dataclass
class RecordingProcess:
    result_path: str
    pid: int = 1234
    returncode: int | None = None
    stdin: Any = None
    stdout: Any = None
    stderr: Any = None
    communicate_calls: int = 0
    killed: bool = False
    raise_timeout: bool = False
    empty_result: bool = False
    succeed: bool = True
    events: list[str] = field(default_factory=list)

    def communicate(
        self,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        del timeout
        self.communicate_calls += 1
        self.events.append("communicate")
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd="child", timeout=1)
        assert input is not None
        request = decode_process_request(input)
        if self.empty_result:
            Path(self.result_path).write_bytes(b"")
            self.returncode = 1
            return b"", b""
        outcome = "succeeded" if self.succeed else "failed"
        response = create_process_execution_response(
            correlation_id=request.correlation_id,
            outcome=outcome,
            structured_data={"ok": self.succeed},
            safe_summary="ok" if self.succeed else "failed",
            error_class=None if self.succeed else "ToolError",
        )
        Path(self.result_path).write_bytes(encode_process_response(response))
        self.returncode = 0 if self.succeed else 1
        return b"", b""

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.events.append("kill")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


def test_flags_and_limits_defaults() -> None:
    assert PROCESS_RESOURCE_LIMITS_ENABLED is False
    assert PROCESS_JOB_ACTIVE_PROCESS_LIMIT == 1
    assert default_job_memory_limit_bytes() == MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES


def test_memory_and_process_limit_validation() -> None:
    assert validate_job_memory_limit(MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES) == (
        MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES
    )
    with pytest.raises(Exception):
        validate_job_memory_limit(1024)
    with pytest.raises(Exception):
        validate_active_process_limit(2)
    assert validate_active_process_limit(1) == 1


def test_resource_limit_exceeded_is_distinct_outcome() -> None:
    result = create_tool_execution_result(
        request_id=str(uuid4()),
        tool_id="system-summary",
        outcome="resource_limit_exceeded",
        safe_summary="limit exceeded",
        structured_data={"resource_limit_exceeded": True, "timeout": False},
        error_class="ResourceLimitExceeded",
    )
    assert result.outcome == "resource_limit_exceeded"
    forbidden = {"failed", "timed_out_terminated", "cancelled"}
    assert result.outcome not in forbidden
    assert _map_execute_step_status(result) == "failed"


def test_file_tools_remain_prohibited() -> None:
    registry = build_default_tool_registry()
    assert registry.require("file-sha256").process_isolation == "eligible"
    assert registry.require("compare-sha256").process_isolation == "eligible"
    assert registry.require("text-search").process_isolation == "prohibited"


def test_assign_before_communicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolation_and_limits(monkeypatch)
    controller = FakeJobController()
    process_holder: dict[str, RecordingProcess] = {}
    order: list[str] = []

    def factory(command: list[str], **_kwargs: Any) -> RecordingProcess:
        proc = RecordingProcess(result_path=command[-1])
        process_holder["proc"] = proc
        return proc

    repo, scope, running = _ready_request(tmp_path)
    definition = build_default_tool_registry().require("system-summary")

    original_create = controller.create_session

    def create_session(**kwargs: Any) -> FakeJobSession:
        session = original_create(**kwargs)
        original_assign = session.assign

        def tracked_assign(pid: int) -> None:
            order.append("assign")
            original_assign(pid)

        session.assign = tracked_assign  # type: ignore[method-assign]
        return session

    controller.create_session = create_session  # type: ignore[method-assign]

    original_communicate = RecordingProcess.communicate

    def tracked_communicate(
        self: RecordingProcess,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        order.append("communicate")
        return original_communicate(self, input=input, timeout=timeout)

    monkeypatch.setattr(RecordingProcess, "communicate", tracked_communicate)

    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            job_controller_factory=lambda: controller,
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
    assert result.outcome == "succeeded"
    assert order == ["assign", "communicate"]
    assert controller.session is not None
    assert controller.session.closed is True
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_job_object_created" in actions
    assert "process_job_object_configured" in actions
    assert "process_job_object_assigned" in actions


def test_setup_failure_prevents_request_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolation_and_limits(monkeypatch)
    controller = FakeJobController(fail_create=True)
    launched = {"communicate": 0}

    class BoomProcess(RecordingProcess):
        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            if input is not None:
                launched["communicate"] += 1
            return super().communicate(input=input, timeout=timeout)

    def factory(command: list[str], **_kwargs: Any) -> BoomProcess:
        return BoomProcess(result_path=command[-1])

    repo, scope, running = _ready_request(tmp_path)
    definition = build_default_tool_registry().require("system-summary")
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            job_controller_factory=lambda: controller,
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
    assert result.structured_data.get("request_sent") is False
    assert launched["communicate"] == 0
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_job_object_setup_failed" in actions


def test_unavailable_controller_fails_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolation_and_limits(monkeypatch)
    launched = {"count": 0}

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        launched["count"] += 1
        raise AssertionError("must not launch")

    repo, scope, running = _ready_request(tmp_path)
    definition = build_default_tool_registry().require("system-summary")
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            job_controller_factory=lambda: UnavailableJobController(
                reason="pywin32_missing"
            ),
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
    assert launched["count"] == 0
    assert result.outcome == "failed"
    assert result.structured_data.get("resource_governance_unavailable") is True


def test_resource_limit_exceeded_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolation_and_limits(monkeypatch)
    controller = FakeJobController()

    def factory(command: list[str], **_kwargs: Any) -> RecordingProcess:
        return RecordingProcess(result_path=command[-1], empty_result=True)

    original_create = controller.create_session

    def create_session(**kwargs: Any) -> FakeJobSession:
        session = original_create(**kwargs)
        session.force_limit_exceeded = True
        session.peak_memory = session.memory_limit_bytes
        return session

    controller.create_session = create_session  # type: ignore[method-assign]
    repo, scope, running = _ready_request(tmp_path)
    definition = build_default_tool_registry().require("system-summary")
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            job_controller_factory=lambda: controller,
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
    assert result.outcome == "resource_limit_exceeded"
    assert result.structured_data.get("timeout") is False
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_resource_limit_exceeded" in actions


def test_timeout_uses_job_tree_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolation_and_limits(monkeypatch)
    controller = FakeJobController()

    def factory(command: list[str], **_kwargs: Any) -> RecordingProcess:
        return RecordingProcess(result_path=command[-1], raise_timeout=True)

    repo, scope, running = _ready_request(tmp_path)
    definition = create_tool_definition(
        tool_id="system-summary",
        name="System Summary",
        description="Timeout with job",
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
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            job_controller_factory=lambda: controller,
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
    assert result.outcome == "timed_out_terminated"
    assert controller.session is not None
    assert controller.session.terminate_calls == 1
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_tree_termination_requested" in actions
    assert "process_tree_terminated" in actions


def test_limits_disabled_preserves_milestone_13_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_executor.PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.tool_process_adapter.PROCESS_RESOURCE_LIMITS_ENABLED",
        False,
    )
    controller = FakeJobController()

    def factory(command: list[str], **_kwargs: Any) -> RecordingProcess:
        return RecordingProcess(result_path=command[-1])

    repo, scope, running = _ready_request(tmp_path)
    definition = build_default_tool_registry().require("system-summary")
    executor = DefensiveToolExecutor(
        process_adapter=ToolProcessAdapter(
            scratch_dir=tmp_path / "scratch",
            popen_factory=factory,
            job_controller_factory=lambda: controller,
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
    assert result.outcome == "succeeded"
    assert controller.create_calls == 0
    actions = {entry.action for entry in repo.list_audit_entries()}
    assert "process_job_object_created" not in actions


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_windows_job_controller_configure_and_query() -> None:
    controller = WindowsJobController()
    if not controller.available:
        pytest.skip("pywin32 unavailable")
    session = controller.create_session(
        memory_limit_bytes=MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES,
        active_process_limit=1,
    )
    try:
        snapshot = session.verified_limits()
        assert snapshot.active_process_limit == 1
        assert snapshot.memory_limit_bytes == MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES
        assert snapshot.kill_on_job_close is True
        assert snapshot.limits_verified is True
    finally:
        session.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_windows_active_process_limit_blocks_second_assignment(
    tmp_path: Path,
) -> None:
    controller = WindowsJobController()
    if not controller.available:
        pytest.skip("pywin32 unavailable")

    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    session = controller.create_session(
        memory_limit_bytes=MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES,
        active_process_limit=1,
    )
    first: subprocess.Popen[bytes] | None = None
    second: subprocess.Popen[bytes] | None = None
    try:
        first = subprocess.Popen(
            [sys.executable, str(sleeper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        session.assign(first.pid)
        assert session.active_process_count() == 1

        second = subprocess.Popen(
            [sys.executable, str(sleeper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with pytest.raises(ToolProcessJobError):
            session.assign(second.pid)
        assert session.active_process_count() == 1
    finally:
        session.terminate_tree()
        if first is not None:
            first.wait(timeout=5)
        if second is not None:
            second.kill()
            second.wait(timeout=5)
        session.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_windows_memory_limit_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_isolation_and_limits(monkeypatch)
    controller = WindowsJobController()
    if not controller.available:
        pytest.skip("pywin32 unavailable")

    hog = tmp_path / "hog.py"
    hog.write_text(
        "import sys\n"
        "data = []\n"
        "try:\n"
        "    while True:\n"
        "        data.append(bytearray(8 * 1024 * 1024))\n"
        "except Exception:\n"
        "    pass\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write('late')\n",
        encoding="utf-8",
    )

    low_limit = 64 * 1024 * 1024
    monkeypatch.setattr(
        "src.tool_process_adapter.default_job_memory_limit_bytes",
        lambda: low_limit,
    )

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        result_path = command[-1]
        return subprocess.Popen(
            [sys.executable, str(hog), result_path],
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
        description="Memory limit coverage",
        category="system-information",
        version="1.0.0",
        risk_level="informational",
        execution_mode="internal-python",
        supported_objective_types=("inspect",),
        supported_target_types=("system-summary", "none"),
        parameter_schema=(),
        timeout_seconds=20,
        requires_approval=False,
        implementation_identifier="impl_system_summary",
        process_isolation="eligible",
    )
    _, scope, running = _ready_request(tmp_path)
    adapter = ToolProcessAdapter(
        scratch_dir=tmp_path / "scratch",
        popen_factory=factory,
        job_controller_factory=lambda: controller,
    )
    result = adapter.execute(definition=definition, request=running)
    assert result.outcome != "succeeded"
    assert result.outcome != "cancelled"
    # Prefer the distinct resource-limit outcome; accept hard kill variants if
    # the OS reports the exit before peak-memory accounting is queryable.
    assert result.outcome in {
        "resource_limit_exceeded",
        "failed",
        "timed_out_terminated",
        "termination_unconfirmed",
    }


def test_default_controller_unavailable_when_disabled() -> None:
    controller = create_default_job_controller()
    assert controller.available is False
    assert controller.unavailable_reason == "resource_limits_disabled"


def test_no_ctypes_job_implementation() -> None:
    source = Path("src/tool_process_job.py").read_text(encoding="utf-8")
    assert "import ctypes" not in source
    assert "from ctypes" not in source
    assert "ToolRegistry" not in source
    assert "import openai" not in source.lower()
    assert "from src.workflow" not in source


def test_requirements_marker_is_windows_only() -> None:
    text = Path("requirements.txt").read_text(encoding="utf-8")
    assert "pywin32" in text
    assert 'sys_platform == "win32"' in text
