"""Local slash-command framework for Project Cortana."""

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.active_memory import (
    ActiveMemoryCharLimitError,
    ActiveMemoryContext,
    ActiveMemoryCountLimitError,
    DuplicateActiveMemoryError,
    normalize_memory_id,
)
from src.ai_service import OpenAIClient, generate_response
from src.citation_validation import validate_response_citations
from src.config import (
    ACTIVE_MEMORY_PERSISTENCE_ENABLED,
    ALLOWED_DOCUMENT_EXTENSIONS,
    ARBITRARY_SHELL_EXECUTION_ENABLED,
    AUTOMATED_RESPONSE_ENABLED,
    AUTONOMOUS_REMEDIATION_ENABLED,
    CHAIN_OF_CUSTODY_ENABLED,
    DEFENSIVE_TOOL_FRAMEWORK_ENABLED,
    DOCUMENT_CONTEXT_INJECTION_ENABLED,
    EVIDENCE_COPY_ENABLED,
    EXPLICIT_PERSISTENT_MEMORY_ENABLED,
    EXTERNAL_THREAT_INTELLIGENCE_LOOKUPS_ENABLED,
    EXTERNAL_TOOL_EXECUTION_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    INCIDENT_AI_CONTEXT_INJECTION_ENABLED,
    INCIDENT_REPOSITORY_ENABLED,
    INCIDENT_REPOSITORY_PERSISTENCE_ENABLED,
    INCIDENT_SINGLE_INSTANCE_COORDINATION_ENABLED,
    KNOWLEDGE_VAULT_ENABLED,
    LOCAL_DOCUMENT_RETRIEVAL_ENABLED,
    MAX_ACTIVE_MEMORIES,
    MAX_ACTIVE_MEMORY_CHARS,
    MAX_MEMORY_TEXT_LENGTH,
    MAX_RETRIEVED_CHUNKS,
    MAX_RETRIEVED_CONTEXT_CHARS,
    MAX_SEARCH_DOCS_RESULTS,
    MAX_SEARCH_RESULT_PREVIEW_CHARS,
    MAX_STORED_DOCUMENTS,
    SEMANTIC_RETRIEVAL_ENABLED,
    SOURCE_MANIFEST_PERSISTENCE_ENABLED,
    DEFENSIVE_WORKFLOW_ORCHESTRATION_ENABLED,
    TOOL_AUDIT_PERSISTENCE_ENABLED,
    TOOL_DRY_RUN_ENFORCEMENT_ENABLED,
    TOOL_HUMAN_APPROVAL_ENABLED,
    TOOL_SCOPE_ENFORCEMENT_ENABLED,
    TOOL_SINGLE_INSTANCE_COORDINATION_ENABLED,
    WORKFLOW_AI_CONTEXT_INJECTION_ENABLED,
    WORKFLOW_BACKGROUND_EXECUTION_ENABLED,
    WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED,
    WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED,
    WORKFLOW_NESTED_PLAYBOOKS_ENABLED,
    WORKFLOW_PARALLEL_EXECUTION_ENABLED,
)
from src.conversation import ConversationHistory
from src.document import (
    DocumentRecord,
    DocumentValidationError,
    validate_document_id,
)
from src.document_chunker import DocumentChunkingError
from src.document_extractor import DocumentExtractionError, TextExtractor
from src.document_ingestion import DocumentIngestionError, ingest_local_document
from src.document_retrieval import (
    BlankRetrievalQueryError,
    DocumentRetrievalError,
    LexicalDocumentRetriever,
    RetrievalContextLimitError,
    RetrievalResult,
    build_source_manifest,
)
from src.document_vault import (
    DocumentCountLimitError,
    DocumentStorageError,
    DocumentVault,
    DuplicateDocumentHashError,
)
from src.evidence_store import EvidenceStore, LocalEvidenceStore
from src.incident_repository import (
    IncidentRepository,
    IncidentStorageError,
    JsonIncidentRepository,
)
from src.memory import MemoryRecord, MemoryTextTooLongError, MemoryValidationError
from src.memory_store import MemoryStorageError, MemoryStore
from src.retrieval_session import RetrievalSession
from src.security_commands import (
    SECURITY_COMMAND_NAMES,
    SecurityCommandContext,
    handle_security_command,
)
from src.settings import Settings
from src.tool_commands import (
    TOOL_COMMAND_NAMES,
    ToolCommandContext,
    create_default_tool_services,
    handle_tool_command,
)
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry, build_default_tool_registry
from src.tool_repository import (
    JsonToolControlRepository,
    ToolControlRepository,
    ToolStorageError,
)
from src.workflow_commands import (
    WORKFLOW_COMMAND_NAMES,
    WorkflowCommandContext,
    create_default_workflow_services,
    handle_workflow_command,
)
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import WorkflowRunRepository

logger = logging.getLogger("ProjectCortana")

COMMAND_HELP = "help"
COMMAND_STATUS = "status"
COMMAND_CLEAR = "clear"
COMMAND_ABOUT = "about"
COMMAND_EXIT = "exit"
COMMAND_REMEMBER = "remember"
COMMAND_MEMORIES = "memories"
COMMAND_FORGET = "forget"
COMMAND_FORGET_ALL = "forget-all"
COMMAND_RECALL = "recall"
COMMAND_ACTIVE_MEMORIES = "active-memories"
COMMAND_RELEASE = "release"
COMMAND_RELEASE_ALL = "release-all"
COMMAND_ADD_DOCUMENT = "add-document"
COMMAND_DOCUMENTS = "documents"
COMMAND_DOCUMENT = "document"
COMMAND_REMOVE_DOCUMENT = "remove-document"
COMMAND_REMOVE_ALL_DOCUMENTS = "remove-all-documents"
COMMAND_SEARCH_DOCS = "search-docs"
COMMAND_ASK_DOCS = "ask-docs"
COMMAND_SOURCES = "sources"

FORGET_ALL_CONFIRM_TOKEN = "confirm"
REMOVE_ALL_DOCUMENTS_CONFIRM_TOKEN = "confirm"

SUPPORTED_COMMANDS = frozenset(
    {
        COMMAND_HELP,
        COMMAND_STATUS,
        COMMAND_CLEAR,
        COMMAND_ABOUT,
        COMMAND_EXIT,
        COMMAND_REMEMBER,
        COMMAND_MEMORIES,
        COMMAND_FORGET,
        COMMAND_FORGET_ALL,
        COMMAND_RECALL,
        COMMAND_ACTIVE_MEMORIES,
        COMMAND_RELEASE,
        COMMAND_RELEASE_ALL,
        COMMAND_ADD_DOCUMENT,
        COMMAND_DOCUMENTS,
        COMMAND_DOCUMENT,
        COMMAND_REMOVE_DOCUMENT,
        COMMAND_REMOVE_ALL_DOCUMENTS,
        COMMAND_SEARCH_DOCS,
        COMMAND_ASK_DOCS,
        COMMAND_SOURCES,
        *SECURITY_COMMAND_NAMES,
        *TOOL_COMMAND_NAMES,
        *WORKFLOW_COMMAND_NAMES,
    }
)

HELP_TEXT = """Cortana: Available commands:
  /help                 - List available commands and brief descriptions
  /status               - Show safe local session information
  /clear                - Clear in-memory conversation history for this session
  /remember             - Save one explicit persistent memory
  /memories             - List saved persistent memories
  /forget               - Delete one saved memory by ID
  /forget-all           - Delete all saved memories after confirmation
  /recall               - Activate one saved memory for temporary AI context
  /active-memories      - List memories currently active for AI context
  /release              - Remove one memory from active AI context
  /release-all          - Clear all active AI memory context for this session
  /add-document         - Ingest one local document into the Knowledge Vault
  /documents            - List stored Knowledge Vault documents
  /document             - Inspect one stored document by ID
  /remove-document      - Delete one stored document by ID
  /remove-all-documents - Delete all stored documents after confirmation
  /search-docs          - Search Knowledge Vault documents locally
  /ask-docs             - Ask a source-grounded question over retrieved documents
  /sources              - Show sources from the latest grounded answer
  /event-new            - Record one local security event
  /events               - List saved security events
  /event                - Show one security event by ID
  /event-status         - Update one security event status
  /incident-new         - Open one local security incident
  /incidents            - List saved security incidents
  /incident             - Show one security incident by ID
  /incident-status      - Update one security incident status
  /incident-link-event  - Link an event to an incident
  /incident-unlink-event - Unlink an event from an incident
  /indicator-add        - Record one local indicator
  /indicators           - List saved indicators
  /indicator            - Show one indicator by ID
  /evidence-register    - Register and copy local evidence bytes
  /evidence             - List saved evidence metadata
  /evidence-show        - Show one evidence record by ID
  /evidence-verify      - Verify a stored evidence copy by SHA-256
  /incident-add-note    - Add one analyst note to an incident
  /incident-notes       - List analyst notes for an incident
  /incident-timeline    - Show a derived incident timeline
  /tools                - List enabled defensive tools
  /tool                 - Show one defensive tool by ID
  /scope-new            - Create one authorized tool scope
  /scopes               - List authorized scopes
  /scope                - Show one authorized scope by ID
  /scope-disable        - Disable one authorized scope
  /tool-request         - Create one tool execution request
  /tool-requests        - List tool execution requests
  /tool-request-show    - Show one tool execution request
  /tool-dry-run         - Generate a dry-run plan for a request
  /tool-approve         - Approve one tool execution request
  /tool-reject          - Reject one tool execution request
  /tool-cancel          - Cancel one tool execution request
  /tool-run             - Execute one approved tool request
  /tool-result          - Show one tool execution result
  /tool-audit           - List tool-control audit entries
  /playbooks            - List enabled defensive playbooks
  /playbook-show        - Show one defensive playbook by name
  /playbook-run         - Dry-run or execute one trusted playbook
  /playbook-status      - Show one workflow run by ID
  /about                - Describe Project Cortana and this software milestone
  /exit                 - End the session cleanly"""

ABOUT_TEXT = (
    "Cortana: Project Cortana is an AI-powered authorized cybersecurity and "
    "defensive-operations assistant. This build is an early software milestone "
    "focused on identity, local commands, in-session conversation, explicit "
    "user-controlled persistent memory, temporary active memory context, a "
    "local Knowledge Vault, source-grounded document questions, a local "
    "human-controlled security event, incident, indicator, evidence, and "
    "chain-of-custody foundation, a human-supervised defensive tool "
    "framework with scope controls and approval, and trusted defensive "
    "playbook orchestration over allowlisted tools."
)

CLEAR_CONFIRMATION = (
    "Cortana: Conversation history and the latest grounded source manifest "
    "have been cleared. Incident records and evidence were left unchanged."
)
CLEAR_ALREADY_EMPTY = (
    "Cortana: Conversation history is already empty. "
    "Any grounded source manifest has also been cleared. "
    "Incident records and evidence were left unchanged."
)

REMEMBER_MISSING_TEXT = "Cortana: Please provide text to remember. Usage: /remember <text>"
REMEMBER_TOO_LONG = (
    "Cortana: Memory text is too long. "
    f"Maximum length is {MAX_MEMORY_TEXT_LENGTH} characters."
)
MEMORIES_EMPTY = "Cortana: No saved memories."
FORGET_MISSING_ID = "Cortana: Please provide a memory ID. Usage: /forget <memory-id>"
FORGET_NOT_FOUND_TEMPLATE = "Cortana: No saved memory found with ID '{memory_id}'."
FORGET_SUCCESS_TEMPLATE = "Cortana: Deleted memory '{memory_id}'."
FORGET_ALL_PROMPT = (
    "Cortana: This will permanently delete all saved memories. "
    "Type /forget-all confirm to proceed."
)
FORGET_ALL_SUCCESS = "Cortana: All saved memories have been deleted."
FORGET_ALL_ALREADY_EMPTY = "Cortana: There are no saved memories to delete."
FORGET_ALL_NOT_CONFIRMED = (
    "Cortana: Forget-all was not confirmed. Saved memories were left unchanged. "
    "Type /forget-all confirm to permanently delete all saved memories."
)

RECALL_MISSING_ID = "Cortana: Please provide a memory ID. Usage: /recall <memory-id>"
RECALL_NOT_FOUND_TEMPLATE = "Cortana: No saved memory found with ID '{memory_id}'."
RECALL_SUCCESS_TEMPLATE = "Cortana: Memory '{memory_id}' is now active for this session."
RECALL_ALREADY_ACTIVE_TEMPLATE = (
    "Cortana: Memory '{memory_id}' is already active for this session."
)
RECALL_COUNT_LIMIT = (
    "Cortana: Active memory limit reached. "
    f"A maximum of {MAX_ACTIVE_MEMORIES} memories can be active at once."
)
RECALL_CHAR_LIMIT = (
    "Cortana: Active memory character limit reached. "
    f"A maximum of {MAX_ACTIVE_MEMORY_CHARS} characters of active memory text is allowed."
)
ACTIVE_MEMORIES_EMPTY = "Cortana: No memories are currently active for AI context."
RELEASE_MISSING_ID = "Cortana: Please provide a memory ID. Usage: /release <memory-id>"
RELEASE_NOT_ACTIVE_TEMPLATE = (
    "Cortana: Memory '{memory_id}' is not currently active."
)
RELEASE_SUCCESS_TEMPLATE = (
    "Cortana: Memory '{memory_id}' has been released from active context."
)
RELEASE_ALL_SUCCESS = "Cortana: All active memories have been released."
RELEASE_ALL_ALREADY_EMPTY = "Cortana: No active memories to release."

ADD_DOCUMENT_MISSING_PATH = (
    "Cortana: Please provide a file path. Usage: /add-document <path>"
)
ADD_DOCUMENT_SUCCESS_TEMPLATE = (
    "Cortana: Document ingested ({document_id}). "
    "Filename: {filename}. Extracted characters: {character_count}."
)
DOCUMENTS_EMPTY = "Cortana: No documents are stored in the Knowledge Vault."
DOCUMENT_MISSING_ID = (
    "Cortana: Please provide a document ID. Usage: /document <document-id>"
)
DOCUMENT_NOT_FOUND_TEMPLATE = (
    "Cortana: No stored document found with ID '{document_id}'."
)
REMOVE_DOCUMENT_MISSING_ID = (
    "Cortana: Please provide a document ID. Usage: /remove-document <document-id>"
)
REMOVE_DOCUMENT_NOT_FOUND_TEMPLATE = (
    "Cortana: No stored document found with ID '{document_id}'."
)
REMOVE_DOCUMENT_SUCCESS_TEMPLATE = (
    "Cortana: Deleted document '{document_id}'."
)
REMOVE_ALL_DOCUMENTS_PROMPT = (
    "Cortana: This will permanently delete all Knowledge Vault documents. "
    "Type /remove-all-documents confirm to proceed."
)
REMOVE_ALL_DOCUMENTS_SUCCESS = (
    "Cortana: All Knowledge Vault documents have been deleted."
)
REMOVE_ALL_DOCUMENTS_ALREADY_EMPTY = (
    "Cortana: There are no Knowledge Vault documents to delete."
)
REMOVE_ALL_DOCUMENTS_NOT_CONFIRMED = (
    "Cortana: Remove-all-documents was not confirmed. "
    "Stored documents were left unchanged. "
    "Type /remove-all-documents confirm to permanently delete all documents."
)

SEARCH_DOCS_MISSING_QUERY = (
    "Cortana: Please provide a search query. Usage: /search-docs <query>"
)
SEARCH_DOCS_EMPTY_VAULT = (
    "Cortana: No documents are stored in the Knowledge Vault."
)
SEARCH_DOCS_NO_RESULTS = (
    "Cortana: No matching document passages were found."
)
ASK_DOCS_MISSING_QUESTION = (
    "Cortana: Please provide a question. Usage: /ask-docs <question>"
)
ASK_DOCS_EMPTY_VAULT = (
    "Cortana: No documents are stored in the Knowledge Vault."
)
ASK_DOCS_NO_EVIDENCE = (
    "Cortana: No supporting document evidence was found for that question."
)
ASK_DOCS_AI_FAILURE = "Cortana: I could not complete that request."
ASK_DOCS_UNAVAILABLE = (
    "Cortana: Source-grounded document questions are unavailable right now."
)
SOURCES_EMPTY = (
    "Cortana: No grounded document sources are available in this session yet. "
    "Use /ask-docs to ask a source-grounded question first."
)

UNKNOWN_COMMAND_TEMPLATE = (
    "Cortana: Unknown command '{command}'. Type /help for available commands."
)


class CommandOutcome(Enum):
    """Result of handling a local slash command."""

    CONTINUE = "continue"
    EXIT = "exit"


@dataclass(frozen=True)
class CommandResult:
    """Output and session effect from a handled slash command."""

    outcome: CommandOutcome
    message: str | None = None


@dataclass(frozen=True)
class CommandContext:
    """Inputs available to local slash-command handlers."""

    message: str
    settings: Settings
    conversation_history: ConversationHistory
    memory_store: MemoryStore
    active_memory_context: ActiveMemoryContext
    document_vault: DocumentVault
    document_extractor: TextExtractor
    document_retriever: LexicalDocumentRetriever
    retrieval_session: RetrievalSession
    incident_repository: IncidentRepository
    evidence_store: EvidenceStore
    tool_registry: ToolRegistry
    tool_repository: ToolControlRepository
    tool_executor: DefensiveToolExecutor
    workflow_registry: WorkflowRegistry
    workflow_run_repository: WorkflowRunRepository
    workflow_executor: WorkflowExecutor
    client: OpenAIClient | None = None


CommandHandler = Callable[[CommandContext], CommandResult]


def _ephemeral_incident_services() -> tuple[IncidentRepository, EvidenceStore]:
    """Create disposable local stores for tests that omit Milestone 8 injection."""
    root = Path(tempfile.mkdtemp(prefix="cortana-incident-"))
    return (
        JsonIncidentRepository(root / "incidents.json"),
        LocalEvidenceStore(root / "evidence"),
    )


def _ephemeral_tool_services(
    incident_repository: IncidentRepository,
) -> tuple[ToolRegistry, ToolControlRepository, DefensiveToolExecutor]:
    """Create disposable tool services for tests that omit Milestone 9 injection."""
    root = Path(tempfile.mkdtemp(prefix="cortana-tools-"))
    repository = JsonToolControlRepository(root / "tool_control.json")
    registry, executor = create_default_tool_services(
        tool_repository=repository,
        incident_repository=incident_repository,
    )
    return registry, repository, executor


def _ephemeral_workflow_services(
    *,
    tool_registry: ToolRegistry,
    tool_repository: ToolControlRepository,
    tool_executor: DefensiveToolExecutor,
) -> tuple[WorkflowRegistry, WorkflowRunRepository, WorkflowExecutor]:
    """Create disposable workflow services for tests that omit Milestone 10 injection."""
    return create_default_workflow_services(
        tool_registry=tool_registry,
        tool_repository=tool_repository,
        tool_executor=tool_executor,
    )


def parse_slash_input(message: str) -> str | None:
    """Return a normalized command name for Cortana slash input, or None for AI content.

    Path-like leading-slash content (for example ``/etc/passwd``) returns ``None`` so
    it continues through the normal conversation path.
    """
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None

    command_token = stripped.split(maxsplit=1)[0]
    command_body = command_token.lstrip("/")

    if "/" in command_body:
        return None

    return command_body.lower()


def normalize_command_name(message: str) -> str:
    """Return the lowercase command name from a slash-command message."""
    parsed_command = parse_slash_input(message)
    if parsed_command is None:
        stripped = message.strip()
        command_token = stripped.split(maxsplit=1)[0]
        return command_token.lstrip("/").lower()
    return parsed_command


def extract_command_argument(message: str) -> str:
    """Return the raw argument text after the command token, preserving capitalization."""
    stripped = message.strip()
    parts = stripped.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1]


def handle_slash_command(
    message: str,
    *,
    settings: Settings,
    conversation_history: ConversationHistory,
    memory_store: MemoryStore,
    active_memory_context: ActiveMemoryContext,
    document_vault: DocumentVault,
    document_extractor: TextExtractor,
    document_retriever: LexicalDocumentRetriever | None = None,
    retrieval_session: RetrievalSession | None = None,
    incident_repository: IncidentRepository | None = None,
    evidence_store: EvidenceStore | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_repository: ToolControlRepository | None = None,
    tool_executor: DefensiveToolExecutor | None = None,
    workflow_registry: WorkflowRegistry | None = None,
    workflow_run_repository: WorkflowRunRepository | None = None,
    workflow_executor: WorkflowExecutor | None = None,
    client: OpenAIClient | None = None,
) -> CommandResult:
    """Handle a slash command locally.

    Most commands never call the AI service. ``/ask-docs`` is the explicit
    exception and requires an injected client. Milestone 8 security commands,
    Milestone 9 tool commands, and Milestone 10 workflow commands are always
    local and never call the AI service.
    """
    command_name = normalize_command_name(message)

    if incident_repository is None or evidence_store is None:
        ephemeral_repository, ephemeral_store = _ephemeral_incident_services()
        incident_repository = incident_repository or ephemeral_repository
        evidence_store = evidence_store or ephemeral_store

    if tool_registry is None or tool_repository is None or tool_executor is None:
        ephemeral_registry, ephemeral_tools, ephemeral_executor = (
            _ephemeral_tool_services(incident_repository)
        )
        tool_registry = tool_registry or ephemeral_registry
        tool_repository = tool_repository or ephemeral_tools
        tool_executor = tool_executor or ephemeral_executor

    # Workflow services are all-or-nothing: a partial injection can otherwise
    # pair an executor with a different run repository than /playbook-status.
    if (
        workflow_registry is None
        or workflow_run_repository is None
        or workflow_executor is None
    ):
        (
            workflow_registry,
            workflow_run_repository,
            workflow_executor,
        ) = _ephemeral_workflow_services(
            tool_registry=tool_registry,
            tool_repository=tool_repository,
            tool_executor=tool_executor,
        )

    if command_name in SECURITY_COMMAND_NAMES:
        security_result = handle_security_command(
            command_name,
            SecurityCommandContext(
                message=message,
                incident_repository=incident_repository,
                evidence_store=evidence_store,
            ),
        )
        if security_result is not None:
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=security_result.message,
            )

    if command_name in TOOL_COMMAND_NAMES:
        tool_result = handle_tool_command(
            command_name,
            ToolCommandContext(
                message=message,
                tool_registry=tool_registry,
                tool_repository=tool_repository,
                tool_executor=tool_executor,
                incident_repository=incident_repository,
            ),
        )
        if tool_result is not None:
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=tool_result.message,
            )

    if command_name in WORKFLOW_COMMAND_NAMES:
        workflow_result = handle_workflow_command(
            command_name,
            WorkflowCommandContext(
                message=message,
                tool_registry=tool_registry,
                tool_repository=tool_repository,
                tool_executor=tool_executor,
                incident_repository=incident_repository,
                workflow_registry=workflow_registry,
                workflow_run_repository=workflow_run_repository,
                workflow_executor=workflow_executor,
            ),
        )
        if workflow_result is not None:
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=workflow_result.message,
            )

    handler = COMMAND_HANDLERS.get(command_name)

    if handler is None:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=UNKNOWN_COMMAND_TEMPLATE.format(command=f"/{command_name}"),
        )

    context = CommandContext(
        message=message,
        settings=settings,
        conversation_history=conversation_history,
        memory_store=memory_store,
        active_memory_context=active_memory_context,
        document_vault=document_vault,
        document_extractor=document_extractor,
        document_retriever=document_retriever or LexicalDocumentRetriever(),
        retrieval_session=retrieval_session or RetrievalSession(),
        incident_repository=incident_repository,
        evidence_store=evidence_store,
        tool_registry=tool_registry,
        tool_repository=tool_repository,
        tool_executor=tool_executor,
        workflow_registry=workflow_registry,
        workflow_run_repository=workflow_run_repository,
        workflow_executor=workflow_executor,
        client=client,
    )
    return handler(context)


def _handle_help(_context: CommandContext) -> CommandResult:
    """Return the local help text."""
    return CommandResult(outcome=CommandOutcome.CONTINUE, message=HELP_TEXT)


def _handle_status(context: CommandContext) -> CommandResult:
    """Return safe local session status."""
    try:
        status_text = format_status(
            context.settings,
            context.conversation_history,
            context.memory_store,
            context.active_memory_context,
            context.document_vault,
            context.retrieval_session,
            context.incident_repository,
            context.tool_registry,
            context.tool_repository,
            context.workflow_registry,
            context.workflow_run_repository,
        )
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)
    except IncidentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)
    except ToolStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    return CommandResult(outcome=CommandOutcome.CONTINUE, message=status_text)


def _handle_clear(context: CommandContext) -> CommandResult:
    """Clear temporary conversation history and the grounded source manifest."""
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=clear_conversation_history(
            context.conversation_history,
            retrieval_session=context.retrieval_session,
        ),
    )


def _handle_about(_context: CommandContext) -> CommandResult:
    """Return the milestone description."""
    return CommandResult(outcome=CommandOutcome.CONTINUE, message=ABOUT_TEXT)


def _handle_exit(_context: CommandContext) -> CommandResult:
    """Signal clean session termination."""
    return CommandResult(outcome=CommandOutcome.EXIT)


def _handle_remember(context: CommandContext) -> CommandResult:
    """Save one explicit persistent memory without activating it."""
    memory_text = extract_command_argument(context.message)
    if not memory_text.strip():
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMEMBER_MISSING_TEXT,
        )

    try:
        record = context.memory_store.add_memory(memory_text)
    except MemoryTextTooLongError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMEMBER_TOO_LONG,
        )
    except MemoryValidationError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMEMBER_MISSING_TEXT,
        )
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    logger.info("Persistent memory saved id=%s", record.id)
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=f"Cortana: Memory saved ({record.id}).",
    )


def _handle_memories(context: CommandContext) -> CommandResult:
    """List saved persistent memories."""
    try:
        memories = context.memory_store.list_memories()
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not memories:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=MEMORIES_EMPTY)

    lines = ["Cortana: Saved memories:"]
    for memory in memories:
        lines.append(f"  [{memory.id}] {memory.created_at}")
        lines.append(f"    {memory.text}")

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message="\n".join(lines),
    )


def _handle_forget(context: CommandContext) -> CommandResult:
    """Delete one saved memory by ID and remove it from active context if present."""
    memory_id = extract_command_argument(context.message).strip()
    if not memory_id:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_MISSING_ID,
        )

    try:
        record = _find_saved_memory(context.memory_store, memory_id)
        if record is None:
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=FORGET_NOT_FOUND_TEMPLATE.format(memory_id=memory_id),
            )
        deleted = context.memory_store.delete_memory(record.id)
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not deleted:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_NOT_FOUND_TEMPLATE.format(memory_id=memory_id),
        )

    was_active = context.active_memory_context.deactivate(record.id)
    logger.info(
        "Persistent memory deleted id=%s was_active=%s",
        record.id,
        was_active,
    )
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=FORGET_SUCCESS_TEMPLATE.format(memory_id=record.id),
    )


def _handle_forget_all(context: CommandContext) -> CommandResult:
    """Require explicit confirmation before deleting all saved memories."""
    confirmation = extract_command_argument(context.message).strip().lower()

    if confirmation != FORGET_ALL_CONFIRM_TOKEN:
        if confirmation == "":
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=FORGET_ALL_PROMPT,
            )
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_ALL_NOT_CONFIRMED,
        )

    try:
        deleted_count = context.memory_store.delete_all_memories()
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    cleared_active = context.active_memory_context.clear()
    logger.info(
        "Persistent memories deleted count=%s active_cleared=%s",
        deleted_count,
        cleared_active,
    )

    if deleted_count == 0:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_ALL_ALREADY_EMPTY,
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=FORGET_ALL_SUCCESS,
    )


def _find_saved_memory(
    memory_store: MemoryStore,
    memory_id: str,
) -> MemoryRecord | None:
    """Find one saved memory using consistent ID comparison."""
    target_key = normalize_memory_id(memory_id)
    if not target_key:
        return None

    for memory in memory_store.list_memories():
        if normalize_memory_id(memory.id) == target_key:
            return memory
    return None


def _handle_recall(context: CommandContext) -> CommandResult:
    """Activate one saved memory for temporary session AI context."""
    memory_id = extract_command_argument(context.message).strip()
    if not memory_id:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RECALL_MISSING_ID,
        )

    try:
        record = _find_saved_memory(context.memory_store, memory_id)
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if record is None:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RECALL_NOT_FOUND_TEMPLATE.format(memory_id=memory_id),
        )

    try:
        context.active_memory_context.activate(record)
    except DuplicateActiveMemoryError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RECALL_ALREADY_ACTIVE_TEMPLATE.format(memory_id=record.id),
        )
    except ActiveMemoryCountLimitError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RECALL_COUNT_LIMIT,
        )
    except ActiveMemoryCharLimitError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RECALL_CHAR_LIMIT,
        )

    logger.info(
        "Memory activated id=%s active_count=%s",
        record.id,
        context.active_memory_context.active_count,
    )
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=RECALL_SUCCESS_TEMPLATE.format(memory_id=record.id),
    )


def _handle_active_memories(context: CommandContext) -> CommandResult:
    """List memories currently active for temporary AI context."""
    active_memories = context.active_memory_context.list_active()
    if not active_memories:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ACTIVE_MEMORIES_EMPTY,
        )

    lines = ["Cortana: Active memories:"]
    for memory in active_memories:
        lines.append(f"  [{memory.id}]")
        lines.append(f"    {memory.text}")

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message="\n".join(lines),
    )


def _handle_release(context: CommandContext) -> CommandResult:
    """Remove one memory from active session context without deleting storage."""
    memory_id = extract_command_argument(context.message).strip()
    if not memory_id:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RELEASE_MISSING_ID,
        )

    released = context.active_memory_context.deactivate(memory_id)
    if not released:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RELEASE_NOT_ACTIVE_TEMPLATE.format(memory_id=memory_id),
        )

    logger.info(
        "Memory released id=%s active_count=%s",
        memory_id,
        context.active_memory_context.active_count,
    )
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=RELEASE_SUCCESS_TEMPLATE.format(memory_id=memory_id),
    )


def _handle_release_all(context: CommandContext) -> CommandResult:
    """Clear all active session memory context without deleting storage."""
    cleared_count = context.active_memory_context.clear()
    logger.info("Active memories cleared count=%s", cleared_count)

    if cleared_count == 0:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=RELEASE_ALL_ALREADY_EMPTY,
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=RELEASE_ALL_SUCCESS,
    )


def _handle_add_document(context: CommandContext) -> CommandResult:
    """Ingest one explicitly supplied local document without calling the AI."""
    path_argument = extract_command_argument(context.message)
    if not path_argument.strip():
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ADD_DOCUMENT_MISSING_PATH,
        )

    try:
        record = ingest_local_document(
            path_argument,
            vault=context.document_vault,
            extractor=context.document_extractor,
        )
    except DuplicateDocumentHashError as error:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=error.user_message,
        )
    except DocumentCountLimitError as error:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=error.user_message,
        )
    except DocumentExtractionError as error:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=error.user_message,
        )
    except DocumentIngestionError as error:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=error.user_message,
        )
    except DocumentStorageError as error:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=error.user_message,
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=ADD_DOCUMENT_SUCCESS_TEMPLATE.format(
            document_id=record.id,
            filename=record.filename,
            character_count=record.extracted_text_length,
        ),
    )


def _handle_documents(context: CommandContext) -> CommandResult:
    """List stored Knowledge Vault documents without extracted text bodies."""
    try:
        documents = context.document_vault.list_documents()
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not documents:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=DOCUMENTS_EMPTY)

    lines = ["Cortana: Stored documents:"]
    for document in documents:
        lines.append(
            f"  [{document.id}] {document.filename} ({document.extension}) "
            f"size={document.source_size_bytes} chars={document.extracted_text_length} "
            f"ingested={document.ingested_at}"
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message="\n".join(lines),
    )


def _handle_document(context: CommandContext) -> CommandResult:
    """Display safe metadata and locally stored extracted text for one document."""
    document_id_argument = extract_command_argument(context.message).strip()
    if not document_id_argument:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=DOCUMENT_MISSING_ID,
        )

    try:
        document = _find_stored_document(context.document_vault, document_id_argument)
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if document is None:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=DOCUMENT_NOT_FOUND_TEMPLATE.format(
                document_id=document_id_argument
            ),
        )

    message = (
        "Cortana: Locally stored Knowledge Vault source content\n"
        f"  ID: {document.id}\n"
        f"  Filename: {document.filename}\n"
        f"  Extension: {document.extension}\n"
        f"  Source size (bytes): {document.source_size_bytes}\n"
        f"  Content hash: {document.content_hash}\n"
        f"  Extracted characters: {document.extracted_text_length}\n"
        f"  Ingested at: {document.ingested_at}\n"
        "  Extracted text:\n"
        f"{document.extracted_text}"
    )
    return CommandResult(outcome=CommandOutcome.CONTINUE, message=message)


def _handle_remove_document(context: CommandContext) -> CommandResult:
    """Delete one stored document by ID without affecting memories."""
    document_id_argument = extract_command_argument(context.message).strip()
    if not document_id_argument:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMOVE_DOCUMENT_MISSING_ID,
        )

    try:
        document = _find_stored_document(context.document_vault, document_id_argument)
        if document is None:
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=REMOVE_DOCUMENT_NOT_FOUND_TEMPLATE.format(
                    document_id=document_id_argument
                ),
            )
        deleted = context.document_vault.delete_document(document.id)
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not deleted:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMOVE_DOCUMENT_NOT_FOUND_TEMPLATE.format(
                document_id=document_id_argument
            ),
        )

    context.retrieval_session.remove_document(document.id)
    logger.info(
        "Document deleted id=%s filename=%s extension=%s",
        document.id,
        document.filename,
        document.extension,
    )
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=REMOVE_DOCUMENT_SUCCESS_TEMPLATE.format(document_id=document.id),
    )


def _handle_remove_all_documents(context: CommandContext) -> CommandResult:
    """Require exact confirmation before deleting all Knowledge Vault documents."""
    confirmation = extract_command_argument(context.message).strip().lower()

    if confirmation != REMOVE_ALL_DOCUMENTS_CONFIRM_TOKEN:
        if confirmation == "":
            return CommandResult(
                outcome=CommandOutcome.CONTINUE,
                message=REMOVE_ALL_DOCUMENTS_PROMPT,
            )
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMOVE_ALL_DOCUMENTS_NOT_CONFIRMED,
        )

    try:
        deleted_count = context.document_vault.delete_all_documents()
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    context.retrieval_session.clear()
    logger.info("Documents deleted count=%s", deleted_count)

    if deleted_count == 0:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=REMOVE_ALL_DOCUMENTS_ALREADY_EMPTY,
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=REMOVE_ALL_DOCUMENTS_SUCCESS,
    )


def _handle_search_docs(context: CommandContext) -> CommandResult:
    """Run local lexical document search without calling the AI service."""
    query = extract_command_argument(context.message)
    if not query.strip():
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=SEARCH_DOCS_MISSING_QUERY,
        )

    try:
        documents = context.document_vault.list_documents()
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not documents:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=SEARCH_DOCS_EMPTY_VAULT,
        )

    try:
        results = context.document_retriever.search(
            query,
            documents,
            max_results=MAX_SEARCH_DOCS_RESULTS,
        )
    except BlankRetrievalQueryError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=SEARCH_DOCS_MISSING_QUERY,
        )
    except (
        DocumentChunkingError,
        DocumentRetrievalError,
        RetrievalContextLimitError,
    ) as error:
        logger.error(
            "Local document search failed error_type=%s",
            type(error).__name__,
        )
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message="Cortana: Local document search could not be completed safely.",
        )

    logger.info("Local document search completed result_count=%s", len(results))

    if not results:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=SEARCH_DOCS_NO_RESULTS,
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=_format_search_docs_results(results),
    )


def _handle_ask_docs(context: CommandContext) -> CommandResult:
    """Retrieve local evidence and ask a source-grounded AI question."""
    question = extract_command_argument(context.message)
    if not question.strip():
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ASK_DOCS_MISSING_QUESTION,
        )

    if context.client is None:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ASK_DOCS_UNAVAILABLE,
        )

    try:
        documents = context.document_vault.list_documents()
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not documents:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ASK_DOCS_EMPTY_VAULT,
        )

    try:
        results = context.document_retriever.search(question, documents)
    except BlankRetrievalQueryError:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ASK_DOCS_MISSING_QUESTION,
        )
    except (
        DocumentChunkingError,
        DocumentRetrievalError,
        RetrievalContextLimitError,
    ) as error:
        logger.error(
            "Grounded document retrieval failed error_type=%s",
            type(error).__name__,
        )
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message="Cortana: Document evidence could not be prepared safely.",
        )

    if not results:
        logger.info("Grounded document question found no evidence result_count=0")
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ASK_DOCS_NO_EVIDENCE,
        )

    active_memories = context.active_memory_context.list_active()
    try:
        answer = generate_response(
            client=context.client,
            settings=context.settings,
            user_message=question.strip(),
            conversation_history=context.conversation_history,
            active_memories=active_memories or None,
            memory_boundary_token=(
                context.active_memory_context.boundary_token
                if active_memories
                else None
            ),
            document_results=results,
            document_boundary_token=context.retrieval_session.boundary_token,
        )
    except Exception as error:
        logger.error(
            "Grounded document AI request failed error_type=%s",
            type(error).__name__,
        )
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=ASK_DOCS_AI_FAILURE,
        )

    citation_result = validate_response_citations(
        answer,
        {result.citation_label for result in results},
    )
    final_answer = citation_result.sanitized_response
    logger.info(
        "Grounded document question completed result_count=%s "
        "citation_validation_succeeded=%s",
        len(results),
        citation_result.is_valid,
    )

    manifest = build_source_manifest(results)
    context.retrieval_session.record_grounded_result(
        query=question.strip(),
        source_manifest=manifest,
        citation_labels={result.citation_label for result in results},
    )

    context.conversation_history.add_user_message(question.strip())
    context.conversation_history.add_assistant_message(final_answer)

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=f"Cortana: {final_answer}",
    )


def _handle_sources(context: CommandContext) -> CommandResult:
    """Show the source manifest from the latest successful /ask-docs request."""
    manifest = context.retrieval_session.source_manifest
    if not manifest:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=SOURCES_EMPTY,
        )

    lines = ["Cortana: Latest grounded sources:"]
    for entry in manifest:
        lines.append(
            f"  {entry.citation_label} filename={entry.filename} "
            f"chunk_index={entry.chunk_index} "
            f"chars={entry.start_offset}-{entry.end_offset}"
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message="\n".join(lines),
    )


def _format_search_docs_results(results: list[RetrievalResult]) -> str:
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


def _find_stored_document(
    document_vault: DocumentVault,
    document_id: str,
) -> DocumentRecord | None:
    """Find one stored document using canonical UUID comparison when possible."""
    try:
        canonical_id = validate_document_id(document_id)
    except DocumentValidationError:
        canonical_id = document_id.strip()

    if not canonical_id:
        return None

    for document in document_vault.list_documents():
        if document.id == canonical_id:
            return document
    return None


COMMAND_HANDLERS: dict[str, CommandHandler] = {
    COMMAND_HELP: _handle_help,
    COMMAND_STATUS: _handle_status,
    COMMAND_CLEAR: _handle_clear,
    COMMAND_ABOUT: _handle_about,
    COMMAND_EXIT: _handle_exit,
    COMMAND_REMEMBER: _handle_remember,
    COMMAND_MEMORIES: _handle_memories,
    COMMAND_FORGET: _handle_forget,
    COMMAND_FORGET_ALL: _handle_forget_all,
    COMMAND_RECALL: _handle_recall,
    COMMAND_ACTIVE_MEMORIES: _handle_active_memories,
    COMMAND_RELEASE: _handle_release,
    COMMAND_RELEASE_ALL: _handle_release_all,
    COMMAND_ADD_DOCUMENT: _handle_add_document,
    COMMAND_DOCUMENTS: _handle_documents,
    COMMAND_DOCUMENT: _handle_document,
    COMMAND_REMOVE_DOCUMENT: _handle_remove_document,
    COMMAND_REMOVE_ALL_DOCUMENTS: _handle_remove_all_documents,
    COMMAND_SEARCH_DOCS: _handle_search_docs,
    COMMAND_ASK_DOCS: _handle_ask_docs,
    COMMAND_SOURCES: _handle_sources,
}


def format_status(
    settings: Settings,
    conversation_history: ConversationHistory,
    memory_store: MemoryStore,
    active_memory_context: ActiveMemoryContext,
    document_vault: DocumentVault,
    retrieval_session: RetrievalSession | None = None,
    incident_repository: IncidentRepository | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_repository: ToolControlRepository | None = None,
    workflow_registry: WorkflowRegistry | None = None,
    workflow_run_repository: WorkflowRunRepository | None = None,
) -> str:
    """Build safe local session status text for /status."""
    completed_turns = conversation_history.completed_turn_count
    max_turns = conversation_history.max_completed_turns
    saved_memory_count = len(memory_store.list_memories())
    active_count = active_memory_context.active_count
    active_characters = active_memory_context.total_character_usage
    stored_document_count = document_vault.document_count()
    supported_types = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
    session = retrieval_session
    has_manifest = session.has_source_manifest if session is not None else False
    document_injection = (
        "enabled (explicit /ask-docs only)"
        if DOCUMENT_CONTEXT_INJECTION_ENABLED
        else "disabled"
    )
    if incident_repository is None:
        saved_event_count = 0
        saved_incident_count = 0
        saved_indicator_count = 0
        saved_evidence_count = 0
    else:
        saved_event_count = incident_repository.event_count()
        saved_incident_count = incident_repository.incident_count()
        saved_indicator_count = incident_repository.indicator_count()
        saved_evidence_count = incident_repository.evidence_count()

    registry = tool_registry or build_default_tool_registry()
    registered_tool_count = registry.count()
    enabled_tool_count = registry.enabled_count()
    if tool_repository is None:
        active_scope_count = 0
        pending_approval_count = 0
    else:
        active_scope_count = tool_repository.active_scope_count()
        pending_approval_count = tool_repository.pending_approval_count()

    if workflow_registry is None:
        registered_playbook_count = 0
        enabled_playbook_count = 0
    else:
        registered_playbook_count = workflow_registry.count()
        enabled_playbook_count = workflow_registry.enabled_count()
    if workflow_run_repository is None:
        retained_workflow_run_count = 0
    else:
        retained_workflow_run_count = len(workflow_run_repository.list_runs())

    return (
        "Cortana: Session status\n"
        "  Status: online\n"
        f"  Model: {settings.openai_model}\n"
        f"  Retained completed turns: {completed_turns}\n"
        f"  Maximum retained turns: {max_turns}\n"
        "  History persistence: "
        f"{'enabled' if HISTORY_PERSISTENCE_ENABLED else 'disabled'}\n"
        "  Explicit persistent memory: "
        f"{'enabled' if EXPLICIT_PERSISTENT_MEMORY_ENABLED else 'disabled'}\n"
        f"  Saved memories: {saved_memory_count}\n"
        f"  Active memories: {active_count}\n"
        f"  Maximum active memories: {MAX_ACTIVE_MEMORIES}\n"
        f"  Active memory characters: {active_characters}\n"
        f"  Maximum active memory characters: {MAX_ACTIVE_MEMORY_CHARS}\n"
        "  Active memory persistence: "
        f"{'enabled' if ACTIVE_MEMORY_PERSISTENCE_ENABLED else 'disabled'}\n"
        "  Knowledge Vault: "
        f"{'enabled' if KNOWLEDGE_VAULT_ENABLED else 'disabled'}\n"
        f"  Stored documents: {stored_document_count}\n"
        f"  Maximum documents: {MAX_STORED_DOCUMENTS}\n"
        f"  Supported document types: {supported_types}\n"
        "  Local document retrieval: "
        f"{'enabled' if LOCAL_DOCUMENT_RETRIEVAL_ENABLED else 'disabled'}\n"
        "  Semantic retrieval: "
        f"{'enabled' if SEMANTIC_RETRIEVAL_ENABLED else 'disabled'}\n"
        f"  Document context injection: {document_injection}\n"
        f"  Maximum retrieved chunks: {MAX_RETRIEVED_CHUNKS}\n"
        "  Maximum retrieved context characters: "
        f"{MAX_RETRIEVED_CONTEXT_CHARS}\n"
        "  Current source manifest: "
        f"{'present' if has_manifest else 'absent'}\n"
        "  Source manifest persistence: "
        f"{'enabled' if SOURCE_MANIFEST_PERSISTENCE_ENABLED else 'disabled'}\n"
        "  Incident repository: "
        f"{'enabled' if INCIDENT_REPOSITORY_ENABLED else 'disabled'}\n"
        f"  Saved events: {saved_event_count}\n"
        f"  Saved incidents: {saved_incident_count}\n"
        f"  Saved indicators: {saved_indicator_count}\n"
        f"  Saved evidence: {saved_evidence_count}\n"
        "  Evidence-copy capability: "
        f"{'enabled' if EVIDENCE_COPY_ENABLED else 'disabled'}\n"
        "  Chain-of-custody: "
        f"{'enabled' if CHAIN_OF_CUSTODY_ENABLED else 'disabled'}\n"
        "  Automated response: "
        f"{'enabled' if AUTOMATED_RESPONSE_ENABLED else 'disabled'}\n"
        "  External threat-intelligence lookups: "
        f"{'enabled' if EXTERNAL_THREAT_INTELLIGENCE_LOOKUPS_ENABLED else 'disabled'}\n"
        "  Incident AI-context injection: "
        f"{'enabled' if INCIDENT_AI_CONTEXT_INJECTION_ENABLED else 'disabled'}\n"
        "  Repository persistence: "
        f"{'enabled' if INCIDENT_REPOSITORY_PERSISTENCE_ENABLED else 'disabled'}\n"
        "  Single-instance coordination: "
        f"{'enabled' if INCIDENT_SINGLE_INSTANCE_COORDINATION_ENABLED else 'disabled'}\n"
        "  Defensive tool framework: "
        f"{'enabled' if DEFENSIVE_TOOL_FRAMEWORK_ENABLED else 'disabled'}\n"
        f"  Registered tools: {registered_tool_count}\n"
        f"  Enabled tools: {enabled_tool_count}\n"
        "  Scope enforcement: "
        f"{'enabled' if TOOL_SCOPE_ENFORCEMENT_ENABLED else 'disabled'}\n"
        "  Human approval: "
        f"{'enabled' if TOOL_HUMAN_APPROVAL_ENABLED else 'disabled'}\n"
        "  Dry-run enforcement: "
        f"{'enabled' if TOOL_DRY_RUN_ENFORCEMENT_ENABLED else 'disabled'}\n"
        "  Arbitrary shell execution: "
        f"{'enabled' if ARBITRARY_SHELL_EXECUTION_ENABLED else 'disabled'}\n"
        "  External tool execution: "
        f"{'enabled' if EXTERNAL_TOOL_EXECUTION_ENABLED else 'disabled'}\n"
        "  Autonomous remediation: "
        f"{'enabled' if AUTONOMOUS_REMEDIATION_ENABLED else 'disabled'}\n"
        f"  Active scopes: {active_scope_count}\n"
        f"  Pending approvals: {pending_approval_count}\n"
        "  Tool audit persistence: "
        f"{'enabled' if TOOL_AUDIT_PERSISTENCE_ENABLED else 'disabled'}\n"
        "  Tool single-instance coordination: "
        f"{'enabled' if TOOL_SINGLE_INSTANCE_COORDINATION_ENABLED else 'disabled'}\n"
        "  Defensive workflow orchestration: "
        f"{'enabled' if DEFENSIVE_WORKFLOW_ORCHESTRATION_ENABLED else 'disabled'}\n"
        f"  Registered playbooks: {registered_playbook_count}\n"
        f"  Enabled playbooks: {enabled_playbook_count}\n"
        f"  Retained workflow runs: {retained_workflow_run_count}\n"
        "  External playbook loading: "
        f"{'enabled' if WORKFLOW_EXTERNAL_PLAYBOOK_LOADING_ENABLED else 'disabled'}\n"
        "  Dynamic step binding: "
        f"{'enabled' if WORKFLOW_DYNAMIC_STEP_BINDING_ENABLED else 'disabled'}\n"
        "  Parallel workflow execution: "
        f"{'enabled' if WORKFLOW_PARALLEL_EXECUTION_ENABLED else 'disabled'}\n"
        "  Background workflow execution: "
        f"{'enabled' if WORKFLOW_BACKGROUND_EXECUTION_ENABLED else 'disabled'}\n"
        "  Nested playbooks: "
        f"{'enabled' if WORKFLOW_NESTED_PLAYBOOKS_ENABLED else 'disabled'}\n"
        "  Workflow AI-context injection: "
        f"{'enabled' if WORKFLOW_AI_CONTEXT_INJECTION_ENABLED else 'disabled'}"
    )


def clear_conversation_history(
    conversation_history: ConversationHistory,
    *,
    retrieval_session: RetrievalSession | None = None,
) -> str:
    """Clear in-memory history and grounded source manifest for the session."""
    history_was_empty = not conversation_history.turns
    conversation_history.clear()
    if retrieval_session is not None:
        retrieval_session.clear()

    if history_was_empty:
        return CLEAR_ALREADY_EMPTY
    return CLEAR_CONFIRMATION
