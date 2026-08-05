"""Parent-authorized file identity for process-isolated file integrity tools.

Canonical paths and authorized roots are internal authorization data only.
They must never appear in ToolExecutionResult, audits, command output,
workflow records, or incident notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS,
    MAX_TOOL_FILE_BYTES,
)
from src.tool_common import ToolValidationError, validate_sha256_digest
from src.tool_definition import DefensiveToolDefinition
from src.tool_process_common import (
    PROCESS_SAFE_FILE_IMPLEMENTATION_IDS,
    ToolProcessError,
)
from src.tool_process_safe_open import (
    FileTooLarge,
    SafeOpenError,
    WindowsFileIdentity,
    assert_file_tool_isolation_readiness,
    capture_windows_file_identity,
    create_windows_file_identity,
    validate_windows_path_format,
)
from src.tool_request import ToolExecutionRequest
from src.tool_safe_files import filename_only
from src.tool_scope import AuthorizedScope, assert_path_authorized

FILE_AUTHORIZATION_KEYS: frozenset[str] = frozenset(
    {
        "canonical_path",
        "authorized_root",
        "volume_serial_number",
        "file_index_high",
        "file_index_low",
        "expected_size_bytes",
        "baseline_last_write_time_filetime",
    }
)

_FILE_SHA256_PARAM_KEYS: frozenset[str] = frozenset({"file_authorization"})
_COMPARE_SHA256_PARAM_KEYS: frozenset[str] = frozenset(
    {"file_authorization", "expected_sha256"}
)
_MAX_IDENTITY_UINT = (2**64) - 1


@dataclass(frozen=True)
class FileAuthorization:
    """Immutable parent-captured Windows file authorization for the child."""

    canonical_path: str
    authorized_root: str
    volume_serial_number: int
    file_index_high: int
    file_index_low: int
    expected_size_bytes: int
    baseline_last_write_time_filetime: int

    def to_exact_dict(self) -> dict[str, Any]:
        """Return the exact-key JSON object sent to the child."""
        return {
            "canonical_path": self.canonical_path,
            "authorized_root": self.authorized_root,
            "volume_serial_number": self.volume_serial_number,
            "file_index_high": self.file_index_high,
            "file_index_low": self.file_index_low,
            "expected_size_bytes": self.expected_size_bytes,
            "baseline_last_write_time_filetime": (
                self.baseline_last_write_time_filetime
            ),
        }

    def to_windows_identity(self) -> WindowsFileIdentity:
        """Build the Windows identity expected by safe_open_for_read."""
        return create_windows_file_identity(
            canonical_path=self.canonical_path,
            volume_serial_number=self.volume_serial_number,
            file_index_high=self.file_index_high,
            file_index_low=self.file_index_low,
            size_bytes=self.expected_size_bytes,
            last_write_time_filetime=self.baseline_last_write_time_filetime,
        )


def create_file_authorization(
    *,
    canonical_path: str,
    authorized_root: str,
    volume_serial_number: int,
    file_index_high: int,
    file_index_low: int,
    expected_size_bytes: int,
    baseline_last_write_time_filetime: int,
) -> FileAuthorization:
    """Validate and construct one FileAuthorization record."""
    return validate_file_authorization(
        {
            "canonical_path": canonical_path,
            "authorized_root": authorized_root,
            "volume_serial_number": volume_serial_number,
            "file_index_high": file_index_high,
            "file_index_low": file_index_low,
            "expected_size_bytes": expected_size_bytes,
            "baseline_last_write_time_filetime": (
                baseline_last_write_time_filetime
            ),
        }
    )


def validate_file_authorization(payload: Any) -> FileAuthorization:
    """Validate an exact-key nested file authorization object."""
    if not isinstance(payload, dict):
        raise ToolProcessError("file_authorization must be an object.")
    keys = frozenset(payload)
    if keys != FILE_AUTHORIZATION_KEYS:
        raise ToolProcessError("file_authorization keys are not an exact match.")

    canonical_path = _require_authorization_path(
        payload["canonical_path"],
        field_name="canonical_path",
    )
    authorized_root = _require_authorization_path(
        payload["authorized_root"],
        field_name="authorized_root",
    )
    volume = _require_nonnegative_int(
        payload["volume_serial_number"],
        field_name="volume_serial_number",
    )
    index_high = _require_nonnegative_int(
        payload["file_index_high"],
        field_name="file_index_high",
    )
    index_low = _require_nonnegative_int(
        payload["file_index_low"],
        field_name="file_index_low",
    )
    expected_size = _require_nonnegative_int(
        payload["expected_size_bytes"],
        field_name="expected_size_bytes",
    )
    if expected_size > MAX_TOOL_FILE_BYTES:
        raise FileTooLarge("File exceeds the maximum permitted size.")
    baseline_write = _require_nonnegative_int(
        payload["baseline_last_write_time_filetime"],
        field_name="baseline_last_write_time_filetime",
    )
    return FileAuthorization(
        canonical_path=canonical_path,
        authorized_root=authorized_root,
        volume_serial_number=volume,
        file_index_high=index_high,
        file_index_low=index_low,
        expected_size_bytes=expected_size,
        baseline_last_write_time_filetime=baseline_write,
    )


def validate_file_tool_process_parameters(
    implementation_identifier: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact child parameters for process-isolated file tools."""
    if implementation_identifier not in PROCESS_SAFE_FILE_IMPLEMENTATION_IDS:
        raise ToolProcessError(
            "Implementation is not an approved process-isolated file tool."
        )
    if not isinstance(parameters, dict):
        raise ToolProcessError("normalized_parameters must be an object.")
    keys = frozenset(parameters)
    if any(key != "file_authorization" and key.startswith("file_authorization")
           for key in keys):
        raise ToolProcessError("Secondary file authorization objects are not permitted.")
    if "file_authorization_2" in keys or "secondary_file_authorization" in keys:
        raise ToolProcessError("Secondary file authorization objects are not permitted.")

    if implementation_identifier == "impl_file_sha256":
        if keys != _FILE_SHA256_PARAM_KEYS:
            raise ToolProcessError(
                "file-sha256 process parameters keys are not an exact match."
            )
        auth = validate_file_authorization(parameters["file_authorization"])
        return {"file_authorization": auth.to_exact_dict()}

    if implementation_identifier == "impl_compare_sha256":
        if keys != _COMPARE_SHA256_PARAM_KEYS:
            raise ToolProcessError(
                "compare-sha256 process parameters keys are not an exact match."
            )
        auth = validate_file_authorization(parameters["file_authorization"])
        expected = validate_sha256_digest(str(parameters["expected_sha256"]))
        return {
            "file_authorization": auth.to_exact_dict(),
            "expected_sha256": expected,
        }

    raise ToolProcessError("Unsupported process-isolated file tool.")


def capture_parent_file_authorization(
    *,
    path_value: str,
    scope: AuthorizedScope,
) -> FileAuthorization:
    """Capture Windows file identity after parent-side scope authorization."""
    assert_file_tool_isolation_readiness()
    resolved = assert_path_authorized(scope, path_value)
    authorized_root = _matching_authorized_root(scope, resolved)
    path_text = validate_windows_path_format(str(resolved))
    root_text = validate_windows_path_format(authorized_root)
    identity = capture_windows_file_identity(path_text)
    if identity.size_bytes > MAX_TOOL_FILE_BYTES:
        raise FileTooLarge("File exceeds the maximum permitted size.")
    # Ensure the handle-derived path remains under the authorized root.
    if not _path_under_root(identity.canonical_path, root_text):
        raise SafeOpenError("Path is outside the authorized root.")
    return create_file_authorization(
        canonical_path=identity.canonical_path,
        authorized_root=root_text,
        volume_serial_number=identity.volume_serial_number,
        file_index_high=identity.file_index_high,
        file_index_low=identity.file_index_low,
        expected_size_bytes=identity.size_bytes,
        baseline_last_write_time_filetime=identity.last_write_time_filetime,
    )


def build_isolated_file_tool_parameters(
    *,
    definition: DefensiveToolDefinition,
    request: ToolExecutionRequest,
    scope: AuthorizedScope,
) -> dict[str, Any]:
    """Build exact child parameters after parent identity capture succeeds."""
    implementation = definition.implementation_identifier
    if implementation not in PROCESS_SAFE_FILE_IMPLEMENTATION_IDS:
        raise ToolProcessError(
            "Implementation is not an approved process-isolated file tool."
        )
    path_value = str(request.normalized_parameters["path"])
    authorization = capture_parent_file_authorization(
        path_value=path_value,
        scope=scope,
    )
    if implementation == "impl_file_sha256":
        return validate_file_tool_process_parameters(
            implementation,
            {"file_authorization": authorization.to_exact_dict()},
        )
    expected = validate_sha256_digest(
        str(request.normalized_parameters["expected_sha256"])
    )
    return validate_file_tool_process_parameters(
        implementation,
        {
            "file_authorization": authorization.to_exact_dict(),
            "expected_sha256": expected,
        },
    )


def safe_filename_from_authorization(authorization: FileAuthorization) -> str:
    """Return basename-only display name from authorization data."""
    return filename_only(authorization.canonical_path)


def _matching_authorized_root(scope: AuthorizedScope, resolved: Path) -> str:
    for root in scope.allowed_local_path_roots:
        try:
            root_path = Path(root)
            if root_path.is_symlink():
                continue
            root_resolved = root_path.resolve(strict=False)
            resolved.relative_to(root_resolved)
            return str(root_resolved)
        except (OSError, ValueError):
            continue
    raise ToolValidationError("Target path is outside the authorized scope roots.")


def _require_authorization_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolProcessError(f"{field_name} must be a non-empty string.")
    if len(value) > MAX_PROCESS_FILE_AUTHORIZATION_PATH_CHARS:
        raise ToolProcessError(f"{field_name} exceeds the maximum length.")
    return validate_windows_path_format(value)


def _require_nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolProcessError(f"{field_name} must be an integer.")
    if value < 0 or value > _MAX_IDENTITY_UINT:
        raise ToolProcessError(f"{field_name} is outside the allowed range.")
    return value


def _path_under_root(path_value: str, root_value: str) -> bool:
    path_norm = path_value.replace("/", "\\").rstrip("\\").casefold()
    root_norm = root_value.replace("/", "\\").rstrip("\\").casefold()
    if path_norm == root_norm:
        return True
    return path_norm.startswith(root_norm + "\\")
