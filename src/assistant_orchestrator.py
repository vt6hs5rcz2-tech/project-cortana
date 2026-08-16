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
    LOCAL_DOCUMENT_RETRIEVAL_ENABLED,
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
from src.memory_store import MemoryCountLimitError, MemoryStorageError, MemoryStore
from src.security_common import InvalidSecurityIdError, validate_security_id
from src.security_incident import SecurityIncident

_MEMORY_WRITE_PATTERN = re.compile(
    r"^remember\s*[:\-]?\s+(?P<text>.+)$",
    re.IGNORECASE,
)
_MEMORY_FILLER_SUFFIX = re.compile(
    r"(?:\s+(?:please|forever|for\s+me|right\s+now))+$",
    re.IGNORECASE,
)
_MEMORY_POSSESSIVE = re.compile(r"['’]s\b")
_MEMORY_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
_MEMORY_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|equals|called)\b|=",
    re.IGNORECASE,
)
_MEMORY_MY_FACT = re.compile(
    r"^(?:that\s+)?(?:my|our)\s+\w+",
    re.IGNORECASE,
)
_MEMORY_THE_X_IS_Y = re.compile(
    r"^(?:that\s+)?the\s+\w+(?:\s+\w+){0,3}\s+(?:is|are|was|were|equals)\s+\S+",
    re.IGNORECASE,
)
# Discourse / anaphoric nouns are not factual values.
_MEMORY_EMPTY_REFERENCE = frozenset(
    {
        "this",
        "that",
        "it",
        "these",
        "those",
        "thing",
        "things",
        "stuff",
        "one",
        "ones",
        "fact",
        "facts",
        "item",
        "items",
        "part",
        "parts",
        "last",
        "details",
        "specifics",
        "situation",
        "issue",
        "topic",
        "matter",
        "point",
        "subject",
        "context",
        "info",
        "information",
        "conversation",
        "discussion",
        "aforementioned",
        "particular",
        "earlier",
        "previous",
        "before",
        "ago",
        "minute",
        "moment",
        "above",
        "whatever",
        "all",
    }
)
_MEMORY_FUNCTION_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "your",
        "our",
        "his",
        "her",
        "their",
        "i",
        "we",
        "you",
        "me",
        "us",
        "he",
        "she",
        "they",
        "what",
        "which",
        "who",
        "said",
        "say",
        "told",
        "mentioned",
        "discussed",
        "discuss",
        "talked",
        "talk",
        "just",
        "previously",
        "from",
        "know",
        "yeah",
        "well",
        "like",
        "kinda",
        "sort",
        "please",
        "forever",
        "now",
        "right",
        "was",
        "were",
        "is",
        "are",
        "been",
        "be",
        "am",
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "about",
        "with",
        "here",
        "there",
        "and",
        "or",
        "we",
        "another",
    }
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
_REMINDER_SET_GUIDANCE_PHRASES = frozenset(
    {
        "set a reminder",
    }
)
_REMINDER_LIST_GUIDANCE_PHRASES = frozenset(
    {
        "list reminders",
    }
)
_CALENDAR_SHOW_GUIDANCE_PHRASES = frozenset(
    {
        "show my calendar",
    }
)
_CALENDAR_SCHEDULE_GUIDANCE_PHRASES = frozenset(
    {
        "schedule a meeting",
    }
)
_STUDY_GUIDANCE_PHRASES = frozenset(
    {
        "help me study",
        "quiz me",
    }
)
_STUDY_GUIDANCE = (
    "Cortana: Study Partner uses explicit slash commands. "
    "Start with /study-start <doc-id>[,<doc-id>...], then use "
    "/study-explain, /study-question, /study-answer, /study-progress, "
    "and /study-end. Run /help for the full list."
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
_REMINDER_SET_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _REMINDER_SET_GUIDANCE_PHRASES
)
_REMINDER_LIST_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _REMINDER_LIST_GUIDANCE_PHRASES
)
_CALENDAR_SHOW_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _CALENDAR_SHOW_GUIDANCE_PHRASES
)
_CALENDAR_SCHEDULE_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _CALENDAR_SCHEDULE_GUIDANCE_PHRASES
)
_STUDY_GUIDANCE_PHRASES_NORMALIZED = frozenset(
    phrase.casefold() for phrase in _STUDY_GUIDANCE_PHRASES
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
_REMINDER_SET_GUIDANCE = (
    "Cortana: Reminders are created only through the controlled "
    "/reminder-add command with an explicit local time, IANA timezone, "
    "recurrence, and message fields. Natural-language input cannot create "
    "or schedule reminders."
)
_REMINDER_LIST_GUIDANCE = (
    "Cortana: Scheduled reminders are listed only through the controlled "
    "/reminders command. Natural-language input cannot list or modify "
    "reminders."
)
_CALENDAR_SHOW_GUIDANCE = (
    "Cortana: Calendar events are listed only through the controlled "
    "/calendar-events command. Natural-language input cannot read or "
    "modify calendar data."
)
_CALENDAR_SCHEDULE_GUIDANCE = (
    "Cortana: Calendar events are created only through the controlled "
    "/calendar-create command followed by explicit /calendar-confirm. "
    "Natural-language input cannot schedule or modify calendar events."
)


def _strip_memory_filler_suffixes(text: str) -> str:
    """Remove trailing polite/duration fillers from a remember payload."""
    cleaned = text.strip().rstrip(".,!?").strip()
    while True:
        next_text = _MEMORY_FILLER_SUFFIX.sub("", cleaned).strip().rstrip(".,!?").strip()
        if next_text == cleaned:
            return next_text
        cleaned = next_text


def _memory_content_tokens(text: str) -> tuple[str, ...]:
    """Normalize a remember payload into lowercase tokens."""
    cleaned = _strip_memory_filler_suffixes(text)
    cleaned = _MEMORY_POSSESSIVE.sub("", cleaned.casefold())
    return tuple(token for token in _MEMORY_TOKEN_SPLIT.split(cleaned) if token)


def _token_has_digit(token: str) -> bool:
    return any(char.isdigit() for char in token)


def _is_content_memory_token(token: str) -> bool:
    """Return True for a token that can carry a factual value."""
    if _token_has_digit(token):
        return True
    if token in _MEMORY_EMPTY_REFERENCE or token in _MEMORY_FUNCTION_WORDS:
        return False
    return True


def _has_factual_memory_content(text: str) -> bool:
    """Return True when the payload has deterministic factual structure.

    Positive signals: digits/identifiers, ``X is Y`` / ``X = Y``,
    ``my/our X ...``, or ``the X is Y``. Ambiguous discourse is rejected.
    Slash ``/remember`` is not filtered by this function.
    """
    tokens = _memory_content_tokens(text)
    if not tokens:
        return False
    if any(_token_has_digit(token) for token in tokens):
        return True
    joined = " ".join(tokens)
    content = [token for token in tokens if _is_content_memory_token(token)]
    if len(content) >= 2:
        return True
    if len(content) >= 1 and _MEMORY_PREDICATE.search(joined):
        return True
    if _MEMORY_THE_X_IS_Y.search(joined) and content:
        return True
    if _MEMORY_MY_FACT.search(joined) and content:
        return True
    return False


def _is_deictic_memory_text(text: str) -> bool:
    """Return True when NL remember text is too ambiguous to persist."""
    return not _has_factual_memory_content(text)


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
            self._try_reminder_set_guidance,
            self._try_reminder_list_guidance,
            self._try_calendar_show_guidance,
            self._try_calendar_schedule_guidance,
            self._try_study_guidance,
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
        if len(text) > MAX_MEMORY_TEXT_LENGTH:
            pass
        elif _is_deictic_memory_text(text):
            return None

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
        except MemoryCountLimitError as error:
            return OrchestrationResult(
                domain="memory",
                action="write",
                confidence="high",
                missing_fields=(),
                safe_user_message=error.user_message,
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

        if not LOCAL_DOCUMENT_RETRIEVAL_ENABLED:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=(),
                safe_user_message=(
                    "Cortana: Local document retrieval is currently disabled."
                ),
            )

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
        except DocumentRetrievalError:
            return OrchestrationResult(
                domain="documents",
                action="search",
                confidence="high",
                missing_fields=(),
                safe_user_message="Cortana: I couldn't search the document vault.",
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

    def _try_reminder_set_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _REMINDER_SET_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="reminder_set",
            confidence="high",
            missing_fields=(),
            safe_user_message=_REMINDER_SET_GUIDANCE,
        )

    def _try_reminder_list_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _REMINDER_LIST_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="reminder_list",
            confidence="high",
            missing_fields=(),
            safe_user_message=_REMINDER_LIST_GUIDANCE,
        )

    def _try_calendar_show_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _CALENDAR_SHOW_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="calendar_show",
            confidence="high",
            missing_fields=(),
            safe_user_message=_CALENDAR_SHOW_GUIDANCE,
        )

    def _try_calendar_schedule_guidance(
        self,
        message: str,
    ) -> OrchestrationResult | None:
        if message.casefold() not in _CALENDAR_SCHEDULE_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="calendar_schedule",
            confidence="high",
            missing_fields=(),
            safe_user_message=_CALENDAR_SCHEDULE_GUIDANCE,
        )

    def _try_study_guidance(self, message: str) -> OrchestrationResult | None:
        if message.casefold() not in _STUDY_GUIDANCE_PHRASES_NORMALIZED:
            return None
        return OrchestrationResult(
            domain="guidance",
            action="study_partner",
            confidence="high",
            missing_fields=(),
            safe_user_message=_STUDY_GUIDANCE,
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
