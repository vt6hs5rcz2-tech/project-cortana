"""Trusted in-code registry for defensive tool definitions."""

from __future__ import annotations

from src.config import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    MAX_TOOL_FILE_BYTES,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TOOL_TEXT_SEARCH_MATCHES,
    MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS,
    MAX_TOOL_TEXT_SEARCH_QUERY_CHARS,
)
from src.tool_common import ToolValidationError, UnknownToolIdError, validate_tool_id
from src.tool_definition import (
    DefensiveToolDefinition,
    create_parameter_definition,
    create_tool_definition,
)


class ToolRegistry:
    """Typed registry of trusted DefensiveToolDefinition objects."""

    def __init__(self) -> None:
        self._by_id: dict[str, DefensiveToolDefinition] = {}
        self._order: list[str] = []
        self._implementation_ids: set[str] = set()

    def register(self, definition: DefensiveToolDefinition) -> None:
        """Register one immutable tool definition from trusted application code."""
        if not isinstance(definition, DefensiveToolDefinition):
            raise ToolValidationError(
                "Only DefensiveToolDefinition objects may be registered."
            )
        if definition.tool_id in self._by_id:
            raise ToolValidationError(
                f"Duplicate tool ID '{definition.tool_id}'."
            )
        if definition.implementation_identifier in self._implementation_ids:
            raise ToolValidationError(
                "Duplicate implementation identifier "
                f"'{definition.implementation_identifier}'."
            )
        if definition.risk_level == "prohibited" and definition.enabled:
            raise ToolValidationError("Prohibited tools cannot be enabled.")
        if (
            definition.execution_mode == "future-external"
            and definition.enabled
        ):
            raise ToolValidationError(
                "Future-external tools cannot be enabled."
            )

        self._by_id[definition.tool_id] = definition
        self._order.append(definition.tool_id)
        self._implementation_ids.add(definition.implementation_identifier)

    def get(self, tool_id: str) -> DefensiveToolDefinition | None:
        """Return one tool definition by ID, or None when missing."""
        cleaned = validate_tool_id(tool_id)
        return self._by_id.get(cleaned)

    def require(self, tool_id: str) -> DefensiveToolDefinition:
        """Return one tool definition or raise when missing."""
        definition = self.get(tool_id)
        if definition is None:
            raise UnknownToolIdError(f"Unknown tool ID '{tool_id}'.")
        return definition

    def list_all(self) -> list[DefensiveToolDefinition]:
        """Return all registered tools in deterministic registration order."""
        return [self._by_id[tool_id] for tool_id in self._order]

    def list_enabled(self) -> list[DefensiveToolDefinition]:
        """Return enabled tools in deterministic order."""
        return [
            definition
            for definition in self.list_all()
            if definition.enabled
        ]

    def filter(
        self,
        *,
        category: str | None = None,
        risk_level: str | None = None,
        enabled_only: bool = False,
    ) -> list[DefensiveToolDefinition]:
        """Filter registered tools by optional category and risk level."""
        results: list[DefensiveToolDefinition] = []
        for definition in self.list_all():
            if enabled_only and not definition.enabled:
                continue
            if category is not None and definition.category != category.strip().lower():
                continue
            if (
                risk_level is not None
                and definition.risk_level != risk_level.strip().lower()
            ):
                continue
            results.append(definition)
        return results

    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._order)

    def enabled_count(self) -> int:
        """Return the number of enabled tools."""
        return len(self.list_enabled())

    def prohibited_tool_ids(self) -> frozenset[str]:
        """Return tool IDs registered as prohibited."""
        return frozenset(
            definition.tool_id
            for definition in self.list_all()
            if definition.risk_level == "prohibited"
        )


def build_default_tool_registry() -> ToolRegistry:
    """Create the trusted Milestone 9 built-in tool registry."""
    registry = ToolRegistry()
    for definition in _builtin_tool_definitions():
        registry.register(definition)
    assert_default_process_isolation_consistency(registry)
    return registry


def assert_default_process_isolation_consistency(registry: ToolRegistry) -> None:
    """Assert registry eligible/required IDs match the child dispatch table exactly."""
    from src.tool_process_common import (
        assert_process_isolation_tables_match,
        registry_process_isolation_implementation_ids,
    )
    from src.tool_process_runner import CHILD_DISPATCH_IMPLEMENTATION_IDS

    assert_process_isolation_tables_match(
        registry_implementation_ids=registry_process_isolation_implementation_ids(
            registry.list_all()
        ),
        child_dispatch_ids=CHILD_DISPATCH_IMPLEMENTATION_IDS,
    )


def _builtin_tool_definitions() -> list[DefensiveToolDefinition]:
    """Return the immutable built-in defensive tool definitions."""
    return [
        create_tool_definition(
            tool_id="system-summary",
            name="System Summary",
            description=(
                "Returns safe local platform information without exposing "
                "usernames, hostnames, environment values, or network addresses."
            ),
            category="system-information",
            version="1.0.0",
            risk_level="informational",
            execution_mode="internal-python",
            supported_objective_types=("inspect", "summarize"),
            supported_target_types=("system-summary", "none"),
            parameter_schema=(),
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            maximum_output_characters=MAX_TOOL_OUTPUT_CHARS,
            requires_approval=False,
            supports_dry_run=True,
            enabled=True,
            implementation_identifier="impl_system_summary",
            process_isolation="eligible",
            capability_class="internal-readonly",
        ),
        create_tool_definition(
            tool_id="file-sha256",
            name="File SHA-256",
            description=(
                "Calculates SHA-256 for an explicitly supplied local regular file "
                "using streaming reads. Rejects symlinks and oversized files."
            ),
            category="file-integrity",
            version="1.0.0",
            risk_level="low",
            execution_mode="internal-python",
            supported_objective_types=("hash", "inspect"),
            supported_target_types=("local-file",),
            parameter_schema=(
                create_parameter_definition(
                    name="path",
                    parameter_type="file-path",
                    required=True,
                    description="Local regular file path to hash.",
                    maximum_length=4096,
                    sensitive=False,
                ),
            ),
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            maximum_output_characters=MAX_TOOL_OUTPUT_CHARS,
            requires_approval=True,
            supports_dry_run=True,
            enabled=True,
            implementation_identifier="impl_file_sha256",
            process_isolation="eligible",
            capability_class="internal-readonly",
        ),
        create_tool_definition(
            tool_id="text-search",
            name="Text Search",
            description=(
                "Searches an explicitly supplied user-owned text file for a "
                "literal string. Read-only, no regular expressions."
            ),
            category="log-analysis",
            version="1.0.0",
            risk_level="low",
            execution_mode="internal-python",
            supported_objective_types=("search", "inspect"),
            supported_target_types=("local-file",),
            parameter_schema=(
                create_parameter_definition(
                    name="path",
                    parameter_type="file-path",
                    required=True,
                    description="Local regular text file path to search.",
                    maximum_length=4096,
                ),
                create_parameter_definition(
                    name="query",
                    parameter_type="string",
                    required=True,
                    description="Literal search string.",
                    minimum_length=1,
                    maximum_length=MAX_TOOL_TEXT_SEARCH_QUERY_CHARS,
                    sensitive=True,
                ),
                create_parameter_definition(
                    name="max_matches",
                    parameter_type="integer",
                    required=False,
                    description="Maximum matches to return.",
                    minimum_value=1,
                    maximum_value=MAX_TOOL_TEXT_SEARCH_MATCHES,
                ),
            ),
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            maximum_output_characters=MAX_TOOL_OUTPUT_CHARS,
            requires_approval=True,
            supports_dry_run=True,
            enabled=True,
            implementation_identifier="impl_text_search",
            process_isolation="eligible",
            capability_class="internal-readonly",
        ),
        create_tool_definition(
            tool_id="compare-sha256",
            name="Compare SHA-256",
            description=(
                "Compares a calculated SHA-256 for a local file against an "
                "expected digest. Read-only; does not alter records."
            ),
            category="file-integrity",
            version="1.0.0",
            risk_level="low",
            execution_mode="internal-python",
            supported_objective_types=("compare", "hash"),
            supported_target_types=("local-file",),
            parameter_schema=(
                create_parameter_definition(
                    name="path",
                    parameter_type="file-path",
                    required=True,
                    description="Local regular file path to hash.",
                    maximum_length=4096,
                ),
                create_parameter_definition(
                    name="expected_sha256",
                    parameter_type="sha256",
                    required=True,
                    description="Expected SHA-256 hex digest.",
                ),
            ),
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            maximum_output_characters=MAX_TOOL_OUTPUT_CHARS,
            requires_approval=True,
            supports_dry_run=True,
            enabled=True,
            implementation_identifier="impl_compare_sha256",
            process_isolation="eligible",
            capability_class="internal-readonly",
        ),
        create_tool_definition(
            tool_id="incident-summary",
            name="Incident Summary",
            description=(
                "Produces a deterministic local summary from an existing incident "
                "record without using the AI or exposing note/evidence text."
            ),
            category="incident-support",
            version="1.0.0",
            risk_level="informational",
            execution_mode="internal-python",
            supported_objective_types=("summarize", "inspect"),
            supported_target_types=("incident",),
            parameter_schema=(
                create_parameter_definition(
                    name="incident_id",
                    parameter_type="record-id",
                    required=True,
                    description="Existing local incident ID.",
                ),
            ),
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            maximum_output_characters=MAX_TOOL_OUTPUT_CHARS,
            requires_approval=False,
            supports_dry_run=True,
            enabled=True,
            implementation_identifier="impl_incident_summary",
            capability_class="internal-readonly",
        ),
        create_tool_definition(
            tool_id="simulated-log-check",
            name="Simulated Log Check",
            description=(
                "Demonstrates the tool lifecycle against controlled mock log input. "
                "Output is always labeled as simulation."
            ),
            category="simulation",
            version="1.0.0",
            risk_level="informational",
            execution_mode="simulated",
            supported_objective_types=("simulate", "search"),
            supported_target_types=("mock-log",),
            parameter_schema=(
                create_parameter_definition(
                    name="fixture",
                    parameter_type="choice",
                    required=True,
                    description="Controlled mock log fixture identifier.",
                    choices=("auth-noise", "malware-keyword", "empty"),
                ),
                create_parameter_definition(
                    name="needle",
                    parameter_type="string",
                    required=False,
                    description="Optional literal needle for the mock fixture.",
                    maximum_length=100,
                    sensitive=True,
                ),
            ),
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            maximum_output_characters=MAX_TOOL_OUTPUT_CHARS,
            requires_approval=False,
            supports_dry_run=True,
            enabled=True,
            implementation_identifier="impl_simulated_log_check",
            process_isolation="eligible",
            capability_class="internal-readonly",
        ),
    ]


# Keep module-level constants discoverable for tests without exposing file limits as secrets.
BUILTIN_TOOL_FILE_BYTE_LIMIT = MAX_TOOL_FILE_BYTES
BUILTIN_TOOL_TEXT_PREVIEW_LIMIT = MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS
