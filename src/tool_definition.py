"""Immutable defensive tool definitions and typed parameter schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    MAX_TOOL_DESCRIPTION_LENGTH,
    MAX_TOOL_NAME_LENGTH,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TOOL_PARAMETER_STRING_LENGTH,
    MAX_TOOL_TIMEOUT_SECONDS,
)
from src.tool_common import (
    GATED_TOOL_CAPABILITY_CLASSES,
    RESERVED_TOOL_CAPABILITY_CLASSES,
    TOOL_CAPABILITY_CLASSES,
    TOOL_CATEGORIES,
    TOOL_EXECUTION_MODES,
    TOOL_OBJECTIVE_TYPES,
    TOOL_PARAMETER_TYPES,
    TOOL_PROCESS_ISOLATION_VALUES,
    TOOL_RISK_LEVELS,
    TOOL_TARGET_TYPES,
    BlankToolFieldError,
    InvalidToolEnumError,
    ToolCapabilityClass,
    ToolCategory,
    ToolExecutionMode,
    ToolObjectiveType,
    ToolParameterError,
    ToolParameterType,
    ToolProcessIsolation,
    ToolRiskLevel,
    ToolTargetType,
    ToolValidationError,
    normalize_string_tuple,
    require_non_blank_text,
    validate_controlled_value,
    validate_implementation_id,
    validate_sha256_digest,
    validate_tool_id,
    validate_uuid,
)

FORBIDDEN_PARAMETER_NAMES: frozenset[str] = frozenset(
    {
        "command",
        "shell",
        "cmd",
        "powershell",
        "bash",
        "script",
        "argv",
        "subprocess",
        "os_system",
    }
)


@dataclass(frozen=True)
class ToolParameterDefinition:
    """Typed parameter definition for a defensive tool."""

    name: str
    parameter_type: ToolParameterType
    required: bool
    description: str
    minimum_length: int | None = None
    maximum_length: int | None = None
    minimum_value: int | None = None
    maximum_value: int | None = None
    choices: tuple[str, ...] = ()
    sensitive: bool = False


@dataclass(frozen=True)
class DefensiveToolDefinition:
    """Immutable registered defensive tool definition."""

    tool_id: str
    name: str
    description: str
    category: ToolCategory
    version: str
    risk_level: ToolRiskLevel
    execution_mode: ToolExecutionMode
    supported_objective_types: tuple[ToolObjectiveType, ...]
    supported_target_types: tuple[ToolTargetType, ...]
    parameter_schema: tuple[ToolParameterDefinition, ...]
    timeout_seconds: int
    maximum_output_characters: int
    requires_approval: bool
    supports_dry_run: bool
    enabled: bool
    implementation_identifier: str
    process_isolation: ToolProcessIsolation
    capability_class: ToolCapabilityClass


def create_parameter_definition(
    *,
    name: str,
    parameter_type: str,
    required: bool,
    description: str,
    minimum_length: int | None = None,
    maximum_length: int | None = None,
    minimum_value: int | None = None,
    maximum_value: int | None = None,
    choices: list[str] | tuple[str, ...] | None = None,
    sensitive: bool = False,
) -> ToolParameterDefinition:
    """Create one validated immutable parameter definition."""
    cleaned_name = require_non_blank_text(
        name,
        field_name="Parameter name",
        max_length=MAX_TOOL_NAME_LENGTH,
    ).lower()
    if cleaned_name in FORBIDDEN_PARAMETER_NAMES:
        raise ToolValidationError(
            f"Parameter name '{cleaned_name}' is not allowed."
        )
    if not cleaned_name.replace("_", "").isalnum():
        raise ToolValidationError(
            "Parameter name must use lowercase letters, digits, or underscores."
        )

    normalized_type = validate_controlled_value(
        parameter_type,
        field_name="parameter type",
        allowed=TOOL_PARAMETER_TYPES,
    )
    cleaned_description = require_non_blank_text(
        description,
        field_name="Parameter description",
        max_length=MAX_TOOL_DESCRIPTION_LENGTH,
    )

    normalized_choices = normalize_string_tuple(
        choices,
        field_name="Parameter choices",
        allow_empty=True,
    )
    if normalized_type == "choice" and not normalized_choices:
        raise ToolValidationError("Choice parameters require at least one choice.")
    if normalized_type != "choice" and normalized_choices:
        raise ToolValidationError(
            "Only choice parameters may declare controlled choices."
        )

    if minimum_length is not None and minimum_length < 0:
        raise ToolValidationError("minimum_length cannot be negative.")
    if maximum_length is not None and maximum_length < 1:
        raise ToolValidationError("maximum_length must be at least 1.")
    if (
        minimum_length is not None
        and maximum_length is not None
        and minimum_length > maximum_length
    ):
        raise ToolValidationError("minimum_length cannot exceed maximum_length.")
    if (
        minimum_value is not None
        and maximum_value is not None
        and minimum_value > maximum_value
    ):
        raise ToolValidationError("minimum_value cannot exceed maximum_value.")

    return ToolParameterDefinition(
        name=cleaned_name,
        parameter_type=normalized_type,  # type: ignore[arg-type]
        required=bool(required),
        description=cleaned_description,
        minimum_length=minimum_length,
        maximum_length=maximum_length,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        choices=normalized_choices,
        sensitive=bool(sensitive),
    )


def create_tool_definition(
    *,
    tool_id: str,
    name: str,
    description: str,
    category: str,
    version: str,
    risk_level: str,
    execution_mode: str,
    supported_objective_types: list[str] | tuple[str, ...],
    supported_target_types: list[str] | tuple[str, ...],
    parameter_schema: list[ToolParameterDefinition] | tuple[ToolParameterDefinition, ...],
    timeout_seconds: int = DEFAULT_TOOL_TIMEOUT_SECONDS,
    maximum_output_characters: int = MAX_TOOL_OUTPUT_CHARS,
    requires_approval: bool,
    supports_dry_run: bool = True,
    enabled: bool = True,
    implementation_identifier: str,
    process_isolation: str = "prohibited",
    capability_class: str = "internal-readonly",
) -> DefensiveToolDefinition:
    """Create one validated immutable defensive tool definition."""
    cleaned_tool_id = validate_tool_id(tool_id)
    cleaned_name = require_non_blank_text(
        name,
        field_name="Tool name",
        max_length=MAX_TOOL_NAME_LENGTH,
    )
    cleaned_description = require_non_blank_text(
        description,
        field_name="Tool description",
        max_length=MAX_TOOL_DESCRIPTION_LENGTH,
    )
    cleaned_version = require_non_blank_text(
        version,
        field_name="Tool version",
        max_length=32,
    )
    cleaned_category = validate_controlled_value(
        category,
        field_name="tool category",
        allowed=TOOL_CATEGORIES,
    )
    cleaned_risk = validate_controlled_value(
        risk_level,
        field_name="tool risk level",
        allowed=TOOL_RISK_LEVELS,
    )
    cleaned_mode = validate_controlled_value(
        execution_mode,
        field_name="tool execution mode",
        allowed=TOOL_EXECUTION_MODES,
    )
    objectives = normalize_string_tuple(
        supported_objective_types,
        field_name="supported objective types",
        allowed=TOOL_OBJECTIVE_TYPES,
        allow_empty=False,
    )
    targets = normalize_string_tuple(
        supported_target_types,
        field_name="supported target types",
        allowed=TOOL_TARGET_TYPES,
        allow_empty=False,
    )
    implementation = validate_implementation_id(implementation_identifier)

    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise ToolValidationError("timeout_seconds must be an integer.")
    if timeout_seconds < 1 or timeout_seconds > MAX_TOOL_TIMEOUT_SECONDS:
        raise ToolValidationError(
            f"timeout_seconds must be between 1 and {MAX_TOOL_TIMEOUT_SECONDS}."
        )
    if (
        not isinstance(maximum_output_characters, int)
        or isinstance(maximum_output_characters, bool)
    ):
        raise ToolValidationError("maximum_output_characters must be an integer.")
    if maximum_output_characters < 1 or maximum_output_characters > MAX_TOOL_OUTPUT_CHARS:
        raise ToolValidationError(
            "maximum_output_characters is outside the allowed range."
        )

    if not supports_dry_run:
        raise ToolValidationError(
            "All Milestone 9 tools must support dry-run planning."
        )

    if cleaned_risk == "prohibited" and enabled:
        raise ToolValidationError("Prohibited tools cannot be enabled.")

    if cleaned_mode == "future-external" and enabled:
        # Future-external tools may be registered for compatibility but cannot run.
        # They may appear as disabled definitions only.
        raise ToolValidationError(
            "Future-external tools cannot be enabled in this milestone."
        )

    cleaned_capability = validate_controlled_value(
        capability_class,
        field_name="tool capability class",
        allowed=TOOL_CAPABILITY_CLASSES,
    )
    if cleaned_capability in GATED_TOOL_CAPABILITY_CLASSES and not requires_approval:
        raise ToolValidationError(
            "Gated capability classes require explicit human approval."
        )
    if cleaned_capability == "internal-mutating" and not requires_approval:
        raise ToolValidationError(
            "Side-effecting internal tools require explicit human approval."
        )
    if cleaned_capability in RESERVED_TOOL_CAPABILITY_CLASSES and enabled:
        raise ToolValidationError(
            "Reserved capability classes cannot be enabled."
        )

    cleaned_isolation = validate_controlled_value(
        process_isolation,
        field_name="process isolation",
        allowed=TOOL_PROCESS_ISOLATION_VALUES,
    )
    if cleaned_isolation in {"eligible", "required"}:
        from src.tool_process_common import (
            PROCESS_SAFE_FILE_IMPLEMENTATION_IDS,
            PROCESS_SAFE_IMPLEMENTATION_IDS,
        )

        if implementation not in PROCESS_SAFE_IMPLEMENTATION_IDS:
            raise ToolValidationError(
                "Only approved process-safe implementations may be "
                "process-isolation eligible or required."
            )
        # Only the reviewed Milestone 15 file-integrity implementations may
        # combine file-path parameters with process isolation.
        has_file_path = any(
            item.parameter_type == "file-path" for item in parameter_schema
        )
        if has_file_path and implementation not in PROCESS_SAFE_FILE_IMPLEMENTATION_IDS:
            raise ToolValidationError(
                "File-touching tools cannot be process-isolation eligible."
            )
        if (
            implementation in PROCESS_SAFE_FILE_IMPLEMENTATION_IDS
            and not has_file_path
        ):
            raise ToolValidationError(
                "Process-isolated file tools require a file-path parameter."
            )

    schema = tuple(parameter_schema)
    seen_names: set[str] = set()
    for parameter in schema:
        if not isinstance(parameter, ToolParameterDefinition):
            raise ToolValidationError(
                "parameter_schema entries must be ToolParameterDefinition objects."
            )
        if parameter.name in seen_names:
            raise ToolValidationError(
                f"Duplicate parameter name '{parameter.name}'."
            )
        seen_names.add(parameter.name)

    return DefensiveToolDefinition(
        tool_id=cleaned_tool_id,
        name=cleaned_name,
        description=cleaned_description,
        category=cleaned_category,  # type: ignore[arg-type]
        version=cleaned_version,
        risk_level=cleaned_risk,  # type: ignore[arg-type]
        execution_mode=cleaned_mode,  # type: ignore[arg-type]
        supported_objective_types=objectives,  # type: ignore[arg-type]
        supported_target_types=targets,  # type: ignore[arg-type]
        parameter_schema=schema,
        timeout_seconds=timeout_seconds,
        maximum_output_characters=maximum_output_characters,
        requires_approval=bool(requires_approval),
        supports_dry_run=True,
        enabled=bool(enabled),
        implementation_identifier=implementation,
        process_isolation=cleaned_isolation,  # type: ignore[arg-type]
        capability_class=cleaned_capability,  # type: ignore[arg-type]
    )


def validate_parameters_against_schema(
    raw_parameters: dict[str, Any],
    schema: tuple[ToolParameterDefinition, ...],
) -> dict[str, Any]:
    """Validate and normalize parameters against a tool parameter schema."""
    if not isinstance(raw_parameters, dict):
        raise ToolParameterError("Parameters must be a JSON object.")

    known = {item.name: item for item in schema}
    unknown = sorted(set(raw_parameters) - set(known))
    if unknown:
        raise ToolParameterError(
            f"Unknown parameter(s): {', '.join(unknown)}."
        )

    forbidden = sorted(set(raw_parameters) & FORBIDDEN_PARAMETER_NAMES)
    if forbidden:
        raise ToolParameterError(
            f"Forbidden parameter(s): {', '.join(forbidden)}."
        )

    normalized: dict[str, Any] = {}
    for definition in schema:
        if definition.name not in raw_parameters:
            if definition.required:
                raise ToolParameterError(
                    f"Missing required parameter '{definition.name}'."
                )
            continue
        normalized[definition.name] = _normalize_parameter_value(
            raw_parameters[definition.name],
            definition,
        )
    return normalized


def redact_parameters_for_display(
    parameters: dict[str, Any],
    schema: tuple[ToolParameterDefinition, ...],
) -> dict[str, Any]:
    """Return parameters with sensitive values and path details redacted."""
    by_name = {item.name: item for item in schema}
    redacted: dict[str, Any] = {}
    for key, value in parameters.items():
        definition = by_name.get(key)
        if definition is None:
            redacted[key] = "[redacted]"
            continue
        if definition.sensitive:
            redacted[key] = "[redacted]"
        elif definition.parameter_type == "file-path" and isinstance(value, str):
            redacted[key] = _filename_only(value)
        else:
            redacted[key] = value
    return redacted


def _normalize_parameter_value(
    value: Any,
    definition: ToolParameterDefinition,
) -> Any:
    """Normalize one parameter value according to its typed definition."""
    parameter_type = definition.parameter_type

    if parameter_type == "string":
        if not isinstance(value, str):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be a string."
            )
        cleaned = value.strip()
        if definition.required and not cleaned:
            raise BlankToolFieldError(
                f"Parameter '{definition.name}' cannot be blank."
            )
        _enforce_length(cleaned, definition)
        return cleaned

    if parameter_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be an integer."
            )
        if definition.minimum_value is not None and value < definition.minimum_value:
            raise ToolParameterError(
                f"Parameter '{definition.name}' is below the minimum value."
            )
        if definition.maximum_value is not None and value > definition.maximum_value:
            raise ToolParameterError(
                f"Parameter '{definition.name}' is above the maximum value."
            )
        return value

    if parameter_type == "boolean":
        if not isinstance(value, bool):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be a boolean."
            )
        return value

    if parameter_type == "choice":
        if not isinstance(value, str):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be a string choice."
            )
        cleaned = value.strip().lower()
        allowed = {choice.lower() for choice in definition.choices}
        if cleaned not in allowed:
            raise InvalidToolEnumError(
                f"Parameter '{definition.name}' is not an allowed choice."
            )
        for choice in definition.choices:
            if choice.lower() == cleaned:
                return choice
        raise InvalidToolEnumError(
            f"Parameter '{definition.name}' is not an allowed choice."
        )

    if parameter_type == "file-path":
        if not isinstance(value, str):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be a file path string."
            )
        cleaned = value.strip()
        if not cleaned:
            raise BlankToolFieldError(
                f"Parameter '{definition.name}' cannot be blank."
            )
        _enforce_length(cleaned, definition)
        # Path safety is enforced later by the safe-file service and scope checks.
        return cleaned

    if parameter_type == "sha256":
        if not isinstance(value, str):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be a SHA-256 string."
            )
        return validate_sha256_digest(value, field_name=f"Parameter '{definition.name}'")

    if parameter_type == "record-id":
        if not isinstance(value, str):
            raise ToolParameterError(
                f"Parameter '{definition.name}' must be a record ID string."
            )
        return validate_uuid(value, field_name=f"Parameter '{definition.name}'")

    raise ToolParameterError(
        f"Unsupported parameter type '{parameter_type}'."
    )


def _enforce_length(value: str, definition: ToolParameterDefinition) -> None:
    """Enforce optional minimum and maximum string lengths."""
    maximum = definition.maximum_length or MAX_TOOL_PARAMETER_STRING_LENGTH
    if len(value) > maximum:
        raise ToolParameterError(
            f"Parameter '{definition.name}' exceeds the maximum length."
        )
    if definition.minimum_length is not None and len(value) < definition.minimum_length:
        raise ToolParameterError(
            f"Parameter '{definition.name}' is shorter than the minimum length."
        )


def _filename_only(path_value: str) -> str:
    """Return only the final path segment for display/redaction."""
    cleaned = path_value.replace("\\", "/").rstrip("/")
    if not cleaned:
        return "[filename]"
    return cleaned.rsplit("/", 1)[-1]
