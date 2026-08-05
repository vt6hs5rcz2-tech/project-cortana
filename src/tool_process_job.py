"""Windows Job Object controller for process-isolated tool resource governance.

This module knows only about process handles/PIDs and numeric limits. It must not
import tool registries, parameters, approvals, scopes, AI, workflows, or incidents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, cast

from src.config import (
    MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES,
    MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES_CAP,
    MIN_PROCESS_ISOLATED_JOB_MEMORY_BYTES,
    PROCESS_JOB_ACTIVE_PROCESS_LIMIT,
    PROCESS_RESOURCE_LIMITS_ENABLED,
)
from src.tool_common import ToolValidationError


class ToolProcessJobError(ToolValidationError):
    """Raised when Job Object governance cannot be established or verified."""


class ToolProcessJobUnavailableError(ToolProcessJobError):
    """Raised when resource governance is required but unavailable."""


@dataclass(frozen=True)
class JobLimitSnapshot:
    """Verified Job Object limit configuration."""

    active_process_limit: int
    memory_limit_bytes: int
    kill_on_job_close: bool
    limits_verified: bool
    assignment_confirmed: bool
    active_process_count: int
    peak_job_memory_used: int
    total_terminated_processes: int


class ProcessJobSession(Protocol):
    """One Job Object bound to a single isolated child execution."""

    def assign(self, pid: int) -> None:
        """Assign one child PID to the Job Object."""

    def verified_limits(self) -> JobLimitSnapshot:
        """Return query-confirmed limit and accounting state."""

    def terminate_tree(self, exit_code: int = 1) -> bool:
        """Terminate all processes in the Job Object and report confirmation."""

    def active_process_count(self) -> int:
        """Return the Job Object active process count."""

    def peak_job_memory_used(self) -> int:
        """Return peak job memory usage in bytes."""

    def likely_resource_limit_exceeded(self, *, configured_memory_bytes: int) -> bool:
        """Return whether accounting suggests a resource-limit termination."""

    def close(self) -> None:
        """Close Job Object and related handles."""


class ProcessJobController(Protocol):
    """Factory for per-execution Job Object sessions."""

    @property
    def available(self) -> bool:
        """Return whether this controller can create real Job Objects."""

    @property
    def unavailable_reason(self) -> str | None:
        """Stable reason when unavailable, otherwise None."""

    def create_session(
        self,
        *,
        memory_limit_bytes: int,
        active_process_limit: int = PROCESS_JOB_ACTIVE_PROCESS_LIMIT,
    ) -> ProcessJobSession:
        """Create and configure one Job Object session."""


def validate_job_memory_limit(memory_limit_bytes: int) -> int:
    """Validate the centralized Job Object memory limit."""
    if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int):
        raise ToolProcessJobError("Job memory limit must be an integer.")
    if memory_limit_bytes < MIN_PROCESS_ISOLATED_JOB_MEMORY_BYTES:
        raise ToolProcessJobError("Job memory limit is below the minimum bound.")
    if memory_limit_bytes > MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES_CAP:
        raise ToolProcessJobError("Job memory limit exceeds the maximum bound.")
    return memory_limit_bytes


def validate_active_process_limit(active_process_limit: int) -> int:
    """Validate that the active process limit remains fixed at one."""
    if isinstance(active_process_limit, bool) or not isinstance(active_process_limit, int):
        raise ToolProcessJobError("Active process limit must be an integer.")
    if active_process_limit != 1:
        raise ToolProcessJobError("Active process limit must be exactly 1.")
    return active_process_limit


class UnavailableJobController:
    """Controller used when Job Object governance cannot run."""

    def __init__(self, *, reason: str) -> None:
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str | None:
        return self._reason

    def create_session(
        self,
        *,
        memory_limit_bytes: int,
        active_process_limit: int = PROCESS_JOB_ACTIVE_PROCESS_LIMIT,
    ) -> ProcessJobSession:
        del memory_limit_bytes, active_process_limit
        raise ToolProcessJobUnavailableError(
            "Process resource governance is unavailable."
        )


class _WindowsJobSession:
    """Concrete Windows Job Object session backed by pywin32."""

    def __init__(
        self,
        *,
        job_handle: Any,
        memory_limit_bytes: int,
        active_process_limit: int,
        win32job: Any,
        win32api: Any,
        win32con: Any,
    ) -> None:
        self._job = job_handle
        self._memory_limit_bytes = memory_limit_bytes
        self._active_process_limit = active_process_limit
        self._win32job = win32job
        self._win32api = win32api
        self._win32con = win32con
        self._process_handle: Any | None = None
        self._assignment_confirmed = False
        self._closed = False

    def assign(self, pid: int) -> None:
        if self._closed:
            raise ToolProcessJobError("Job Object session is closed.")
        if not isinstance(pid, int) or pid <= 0:
            raise ToolProcessJobError("Child PID is invalid.")
        access = (
            self._win32con.PROCESS_SET_QUOTA
            | self._win32con.PROCESS_TERMINATE
            | self._win32con.PROCESS_QUERY_INFORMATION
        )
        try:
            process_handle = self._win32api.OpenProcess(access, False, pid)
            self._win32job.AssignProcessToJobObject(self._job, process_handle)
        except Exception as error:
            raise ToolProcessJobError("Job Object assignment failed.") from error
        self._process_handle = process_handle
        self._assignment_confirmed = True

    def verified_limits(self) -> JobLimitSnapshot:
        extended = self._query_extended()
        basic = extended["BasicLimitInformation"]
        flags = int(basic["LimitFlags"])
        kill_on_close = bool(
            flags & self._win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        active_flag = bool(flags & self._win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS)
        memory_flag = bool(flags & self._win32job.JOB_OBJECT_LIMIT_JOB_MEMORY)
        active_limit = int(basic["ActiveProcessLimit"])
        memory_limit = int(extended["JobMemoryLimit"])
        accounting = self._query_accounting()
        limits_verified = (
            kill_on_close
            and active_flag
            and memory_flag
            and active_limit == self._active_process_limit
            and memory_limit == self._memory_limit_bytes
        )
        if not limits_verified:
            raise ToolProcessJobError("Job Object limits failed verification.")
        return JobLimitSnapshot(
            active_process_limit=active_limit,
            memory_limit_bytes=memory_limit,
            kill_on_job_close=kill_on_close,
            limits_verified=True,
            assignment_confirmed=self._assignment_confirmed,
            active_process_count=int(accounting["BasicInfo"]["ActiveProcesses"]),
            peak_job_memory_used=int(extended["PeakJobMemoryUsed"]),
            total_terminated_processes=int(
                accounting["BasicInfo"]["TotalTerminatedProcesses"]
            ),
        )

    def terminate_tree(self, exit_code: int = 1) -> bool:
        if self._closed:
            return False
        try:
            self._win32job.TerminateJobObject(self._job, int(exit_code))
        except Exception as error:
            raise ToolProcessJobError("Job Object tree termination failed.") from error
        # Confirmation of individual child exit is performed by the adapter.
        return self.active_process_count() == 0

    def active_process_count(self) -> int:
        accounting = self._query_accounting()
        return int(accounting["BasicInfo"]["ActiveProcesses"])

    def peak_job_memory_used(self) -> int:
        extended = self._query_extended()
        return int(extended["PeakJobMemoryUsed"])

    def likely_resource_limit_exceeded(self, *, configured_memory_bytes: int) -> bool:
        try:
            peak = self.peak_job_memory_used()
            accounting = self._query_accounting()
            terminated = int(accounting["BasicInfo"]["TotalTerminatedProcesses"])
        except Exception:
            return False
        if peak >= configured_memory_bytes:
            return True
        # Near-limit peak plus terminated members is a strong signal.
        threshold = max(1, int(configured_memory_bytes * 0.90))
        return peak >= threshold and terminated > 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process_handle is not None:
            try:
                self._win32api.CloseHandle(self._process_handle)
            except Exception:
                pass
            self._process_handle = None
        try:
            self._win32api.CloseHandle(self._job)
        except Exception:
            pass
        self._job = None

    def _query_extended(self) -> dict[str, Any]:
        try:
            return cast(
                dict[str, Any],
                self._win32job.QueryInformationJobObject(
                    self._job,
                    self._win32job.JobObjectExtendedLimitInformation,
                ),
            )
        except Exception as error:
            raise ToolProcessJobError("Job Object query failed.") from error

    def _query_accounting(self) -> dict[str, Any]:
        try:
            return cast(
                dict[str, Any],
                self._win32job.QueryInformationJobObject(
                    self._job,
                    self._win32job.JobObjectBasicAndIoAccountingInformation,
                ),
            )
        except Exception as error:
            raise ToolProcessJobError("Job Object accounting query failed.") from error


class WindowsJobController:
    """Create and configure Windows Job Objects via pywin32."""

    def __init__(self) -> None:
        self._win32job: Any | None = None
        self._win32api: Any | None = None
        self._win32con: Any | None = None
        self._import_error: str | None = None
        self._load_modules()

    def _load_modules(self) -> None:
        try:
            import win32api  # type: ignore[import-untyped]
            import win32con  # type: ignore[import-untyped]
            import win32job  # type: ignore[import-untyped]
        except ImportError:
            self._import_error = "pywin32_missing"
            return
        self._win32api = win32api
        self._win32con = win32con
        self._win32job = win32job

    @property
    def available(self) -> bool:
        return self._import_error is None and self._win32job is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._import_error

    def create_session(
        self,
        *,
        memory_limit_bytes: int,
        active_process_limit: int = PROCESS_JOB_ACTIVE_PROCESS_LIMIT,
    ) -> ProcessJobSession:
        if not self.available:
            raise ToolProcessJobUnavailableError(
                "Process resource governance is unavailable."
            )
        assert self._win32job is not None
        assert self._win32api is not None
        assert self._win32con is not None

        memory = validate_job_memory_limit(memory_limit_bytes)
        processes = validate_active_process_limit(active_process_limit)
        try:
            job = self._win32job.CreateJobObject(None, "")
            info = self._win32job.QueryInformationJobObject(
                job,
                self._win32job.JobObjectExtendedLimitInformation,
            )
            info["BasicLimitInformation"]["LimitFlags"] = (
                self._win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | self._win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | self._win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
            )
            info["BasicLimitInformation"]["ActiveProcessLimit"] = processes
            info["JobMemoryLimit"] = memory
            self._win32job.SetInformationJobObject(
                job,
                self._win32job.JobObjectExtendedLimitInformation,
                info,
            )
        except Exception as error:
            raise ToolProcessJobError("Job Object configuration failed.") from error

        session = _WindowsJobSession(
            job_handle=job,
            memory_limit_bytes=memory,
            active_process_limit=processes,
            win32job=self._win32job,
            win32api=self._win32api,
            win32con=self._win32con,
        )
        # Fail closed if QueryInformationJobObject does not confirm limits.
        session.verified_limits()
        return session


def create_default_job_controller() -> ProcessJobController:
    """Return the appropriate Job Object controller for the current environment."""
    if not PROCESS_RESOURCE_LIMITS_ENABLED:
        return UnavailableJobController(reason="resource_limits_disabled")
    if os.name != "nt":
        return UnavailableJobController(reason="non_windows")
    controller = WindowsJobController()
    if not controller.available:
        return UnavailableJobController(
            reason=controller.unavailable_reason or "pywin32_missing"
        )
    return controller


def default_job_memory_limit_bytes() -> int:
    """Return the validated centralized Job Object memory limit."""
    return validate_job_memory_limit(MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES)
