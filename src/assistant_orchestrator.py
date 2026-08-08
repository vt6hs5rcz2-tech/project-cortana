"""Deterministic natural-language routing for a small safe capability set.

Milestone 18 adds high-confidence whole-message matching only. Slash commands
remain the privileged control plane. Ambiguous input falls through to ordinary
conversation. This module does not classify intent with AI and does not execute
tools, workflows, or incident analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import (
    MAX_MEMORY_TEXT_LENGTH,
    MAX_SEARCH_DOCS_RESULTS,
    MAX_SEARCH_RESULT_PREVIEW_CHARS,
)
from src.document_retrieval import (
    BlankRetrievalQueryError,
    DocumentRetrievalError,
    LexicalDocumentRetriever,
    RetrievalResult,
)
from src.document_vault import DocumentStorageError, DocumentVault
from src.incident_repository import IncidentRepository, IncidentStorageError
from src.memory import MemoryTextTooLongError, MemoryValidationError
from src.memory_store import MemoryStorageError, MemoryStore
from src.security_common import InvalidSecurityIdError, validate_security_id
from src.security_incident import SecurityIncident

_MEMORY_WRITE_PATTERN = re.compile(
    r"^remember\s*[:\-]?\s+(?P<text>.+)$",
    re.IGNORECASE,
)
_DOCUMENT_SEARCH_PATTERN = re.compile(
    r"^(?:find\s+documents?\s+about|search\s+(?:my\s+)?documents?\s+for)"
    r"\s+(?P<query>.+)$",
    re.IGNORECASE,
)
_INCIDENT_SHOW_PATTERN = re.compile(
    r"^show\s+incident\s+(?P<incident_id>[0-9a-fA-F-]{36})$",
    re.IGNORECASE,
)

_MEMORY_READ_PHRASES = frozenset(
    {
        "list my memories",
        "list memories",
        "show my memories",
    }
)
_EVIDENCE_SEARCH_GUIDANCE_PHRASES = frozenset(
    {
        "search evidence",
        "search incident evidence",
    }
)
_TOOL_GUIDANCE_PHRASES = frozenset(
    {
        "run a tool",
        "scan this file",
    }
)
_WORKFLOW_GUIDANCE_PHRASES = frozenset(
    {
        "run the playbook",
        "execute the workflow",
    }
)
_ANALYST_GUIDANCE_PHRASES = frozenset(
    {
        "run ai analysis",
        "analyze this incident with ai",
    }
)
_MEMORY_READ_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _MEMORY_READ_PHRASES
)
_EVIDENCE_SEARCH_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _EVIDENCE_SEARCH_GUIDANCE_PHRASES
)
_TOOL_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _TOOL_GUIDANCE_PHRASES
)
_WORKFLOW_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _WORKFLOW_GUIDANCE_PHRASES
)
_ANALYST_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _ANALYST_GUIDANCE_PHRASES
)

_MEMORY_MISSING_TEXT = (
    "Cortana: Please provide text to remember. "
    "Usage: remember <text>"
)
_MEMORY_TOO_LONG = (
    "Cortana: Memory text is too long. "
    f"Maximum length is {MAX_MEMORY_TEXT_LENGTH} characters."
)
_MEMORIES_EMPTY = "Cortana: No saved memories."
_SEARCH_DOCS_EMPTY_VAULT = (
    "Cortana: No documents are stored in the Knowledge Vault."
)
_SEARCH_DOCS_NO_RESULTS = (
    "Cortana: No matching document passages were found."
)
_SEARCH_DOCS_MISSING_QUERY = (
    "Cortana: Please provide a search query."
)
_INCIDENT_NOT_FOUND_TEMPLATE = (
    "Cortana: No saved incident found with ID '{incident_id}'."
)
_INCIDENT_UNAVAILABLE = (
    "Cortana: Security incident repository is unavailable."
)

_EVIDENCE_SEARCH_GUIDANCE = (
    "Cortana: Evidence search is available only through the controlled "
    "/evidence-search command with the required incident ID, evidence ID, "
    "scope ID, and query. Continue with /tool-dry-run, /tool-approve, and "
    "/tool-run. Natural-language input cannot create or run evidence searches."
)
_TOOL_GUIDANCE = (
    "Cortana: Defensive tools run only through the controlled /tool-* flow "
    "(/tool-request, /tool-dry-run, /tool-approve, /tool-run). "
    "Natural-language input cannot request, approve, or execute tools."
)
_WORKFLOW_GUIDANCE = (
    "Cortana: Playbooks and workflows run only through the controlled "
    "/playbook-* flow (/playbook-show, /playbook-run, /playbook-status). "
    "Natural-language input cannot execute workflows."
)
_ANALYST_GUIDANCE = (
    "Cortana: AI incident analysis is available only through the controlled "
    "/incident-analysis-* flow when analysis features are enabled. "
    "Natural-language input cannot prepare or run AI analysis."
)


@dataclass(frozen=True)
class OrchestrationResult:
    """Bounded safe outcome for one high-confidence orchestration match."""

    domain: str
    action: str
    confidence: str
    missing_fields: tuple[str, ...]
    safe_user_message: str


class UnifiedAssistantOrchestrator:
    """Route a tiny set of explicit ordinary-language requests to existing APIs."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        document_vault: DocumentVault,
        document_retriever: LexicalDocumentRetriever,
        incident_repository: IncidentRepository | None,
    ) -> None:
        """Initialize with only the dependencies required by operational routes."""
        self._memory_store = memory_store
        self._document_vault = document_vault
        self._document_retriever = document_retriever
        self._incident_repository = incident_repository

    def try_handle(self, user_message: str) -> OrchestrationResult | None:
        """Return a handled result for one full-message match, else None."""
        message = user_message.strip()
        if not message:
            return None

        handlers = (
            self._try_memory_write,
            self._try_memory_read,
            self._try_document_search,
            self._try_incident_read,
            self._try_evidence_search_guidance,
            self._try_tool_guidance,
            self._try_workflow_guidance,
            self._try_analyst_guidance,
        )
        for handler in handlers:
            result = handler(message)
            if result is not None:
                return result
        return None

    def _try_memory_write(self, message: str) -> OrchestrationResult | None:
        match = _MEMORY_WRITE_PATTERN.fullmatch(message)
        if match is None:
            return None

        text = match.group("text")
        try:
            record = self._memory_store.add_memory(text)
        except MemoryTextTooLongError:
            return OrchestrationResult(
                domain="memory",
                action="write",
                confidence="high",
                missing_fields=(),
                safe_user_message=_MEMORY_TOO_LONG,
            )
        except MemoryValidationError:
            return OrchestrationResult(
                domain="memory",
                action="write",
                confidence="high",
                missing_fields=("text",),
                safe_user_message=_MEMORY_MISSING_TEXT,
            )
        except MemoryStorageError as error:
            return OrchestrationResult(
                domain="memory",
                action="write",
                confidence="high",
                missing_fields=(),
                safe_user_message=error.user_message,
            )

        return OrchestrationResult(
            domain="memory",
            action="write",
            confidence="high",
            missing_fields=(),
            safe_user_message=f"Cortana: Memory saved ({record.id}).",
        )

    def _try_memory_read(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _MEMORY_READ_PHRASES_NORMALIZED:
            return None

        try:
            memories = self._memory_store.list_memories()
        except MemoryStorageError as error:
            return OrchestrationResult(
                domain="memory",
                action="list",
                confidence="high",
                missing_fields=(),
                safe_user_message=error.user_message,
            )

        if not memories:
            return OrchestrationResult(
                domain="memory",
                action="list",
                confidence="high",
                missing_fields=(),
                safe_user_message=_MEMORIES_EMPTY,
            )

        lines = ["Cortana: Saved memories:"]
        for memory in memories:
            lines.append(f"  [{memory.id}] {memory.created_at}")
            lines.append(f"    {memory.text}")
        return OrchestrationResult(
            domain="memory",
            action="list",
            confidence="high",
            missing_fields=(),
            safe_user_message="\n".join(lines),
        )

    def _try_document_search(self, message: str) -> OrchestrationResult | None:
        match = _DOCUMENT_SEARCH_PATTERN.fullmatch(message)
        if match is None:
            return None

        query = match.group("query").strip()
        if not query:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=("query",),
                safe_user_message=_SEARCH_DOCS_MISSING_QUERY,
            )

        try:
            documents = self._document_vault.list_documents()
        except DocumentStorageError as error:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=(),
                safe_user_message=error.user_message,
            )

        if not documents:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=(),
                safe_user_message=_SEARCH_DOCS_EMPTY_VAULT,
            )

        try:
            results = self._document_retriever.search_vault(
                query,
                self._document_vault,
                max_results=MAX_SEARCH_DOCS_RESULTS,
            )
        except BlankRetrievalQueryError:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=("query",),
                safe_user_message=_SEARCH_DOCS_MISSING_QUERY,
            )
        except DocumentRetrievalError as error:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=(),
                safe_user_message=f"Cortana: {error}",
            )

        if not results:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=(),
                safe_user_message=_SEARCH_DOCS_NO_RESULTS,
            )

        return OrchestrationResult(
            domain="documents",
            action="search",
            confidence="high",
            missing_fields=(),
            safe_user_message=_format_search_results(results),
        )

    def _try_incident_read(self, message: str) -> OrchestrationResult | None:
        match = _INCIDENT_SHOW_PATTERN.fullmatch(message)
        if match is None:
            return None

        incident_id_argument = match.group("incident_id")
        if self._incident_repository is None:
            return OrchestrationResult(
                domain="security",
                action="incident_read",
                confidence="high",
                missing_fields=(),
                safe_user_message=_INCIDENT_UNAVAILABLE,
            )

        try:
            incident_id = validate_security_id(
                incident_id_argument,
                field_name="Incident ID",
            )
            incident = self._incident_repository.get_incident(incident_id)
        except InvalidSecurityIdError:
            return OrchestrationResult(
                domain="security",
                action="incident_read",
                confidence="high",
                missing_fields=("incident_id",),
                safe_user_message=_INCIDENT_NOT_FOUND_TEMPLATE.format(
                    incident_id=incident_id_argument
                ),
            )
        except IncidentStorageError as error:
            return OrchestrationResult(
                domain="security",
                action="incident_read",
                confidence="high",
                missing_fields=(),
                safe_user_message=error.user_message,
            )

        if incident is None:
            return OrchestrationResult(
                domain="security",
                action="incident_read",
                confidence="high",
                missing_fields=("incident_id",),
                safe_user_message=_INCIDENT_NOT_FOUND_TEMPLATE.format(
                    incident_id=incident_id_argument
                ),
            )

        return OrchestrationResult(
            domain="security",
            action="incident_read",
            confidence="high",
            missing_fields=(),
            safe_user_message=_format_incident(incident),
        )

    def _try_evidence_search_guidance(
        self,
        message: str,
    ) -> OrchestrationResult | None:
        if message.casefold() not in _EVIDENCE_SEARCH_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="evidence_search",
            confidence="high",
            missing_fields=(),
            safe_user_message=_EVIDENCE_SEARCH_GUIDANCE,
        )

    def _try_tool_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _TOOL_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="tools",
            confidence="high",
            missing_fields=(),
            safe_user_message=_TOOL_GUIDANCE,
        )

    def _try_workflow_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _WORKFLOW_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="workflow",
            confidence="high",
            missing_fields=(),
            safe_user_message=_WORKFLOW_GUIDANCE,
        )

    def _try_analyst_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _ANALYST_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="analyst_assistance",
            confidence="high",
            missing_fields=(),
            safe_user_message=_ANALYST_GUIDANCE,
        )


def _format_search_results(results: list[RetrievalResult]) -> str:
    """Format ranked local search results with bounded previews."""
    lines = ["Cortana: Document search results:"]
    for result in results:
        preview = _bounded_preview(result.chunk.text)
        lines.append(
            f"  {result.citation_label} filename={result.chunk.document_filename} "
            f"chunk_index={result.chunk.chunk_index}"
        )
        lines.append(f"    {preview}")
    return "\n".join(lines)


def _bounded_preview(text: str) -> str:
    """Return a safely bounded preview of chunk text for local display."""
    limit = MAX_SEARCH_RESULT_PREVIEW_CHARS
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def _format_incident(incident: SecurityIncident) -> str:
    """Format one incident using the existing /incident display shape."""
    closed = incident.closed_at or "(none)"
    return (
        "Cortana: Security incident\n"
        f"  ID: {incident.incident_id}\n"
        f"  Title: {incident.title}\n"
        f"  Severity: {incident.severity}\n"
        f"  Status: {incident.status}\n"
        f"  Created at: {incident.created_at}\n"
        f"  Updated at: {incident.updated_at}\n"
        f"  Opened at: {incident.opened_at}\n"
        f"  Closed at: {closed}\n"
        f"  Event IDs: {', '.join(incident.event_ids) if incident.event_ids else '(none)'}\n"
        f"  Evidence IDs: {', '.join(incident.evidence_ids) if incident.evidence_ids else '(none)'}\n"
        f"  Indicator IDs: {', '.join(incident.indicator_ids) if incident.indicator_ids else '(none)'}\n"
        f"  Note IDs: {', '.join(incident.note_ids) if incident.note_ids else '(none)'}\n"
        f"  Tags: {', '.join(incident.tags) if incident.tags else '(none)'}\n"
        "  Summary:\n"
        f"{incident.summary}"
    )
