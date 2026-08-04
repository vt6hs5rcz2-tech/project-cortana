"""Local slash-command framework for Project Cortana."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from src.active_memory import (
    ActiveMemoryCharLimitError,
    ActiveMemoryContext,
    ActiveMemoryCountLimitError,
    DuplicateActiveMemoryError,
    normalize_memory_id,
)
from src.config import (
    ACTIVE_MEMORY_PERSISTENCE_ENABLED,
    ALLOWED_DOCUMENT_EXTENSIONS,
    DOCUMENT_CONTEXT_INJECTION_ENABLED,
    EXPLICIT_PERSISTENT_MEMORY_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    KNOWLEDGE_VAULT_ENABLED,
    MAX_ACTIVE_MEMORIES,
    MAX_ACTIVE_MEMORY_CHARS,
    MAX_MEMORY_TEXT_LENGTH,
    MAX_STORED_DOCUMENTS,
)
from src.conversation import ConversationHistory
from src.document import (
    DocumentRecord,
    DocumentValidationError,
    validate_document_id,
)
from src.document_extractor import DocumentExtractionError, TextExtractor
from src.document_ingestion import DocumentIngestionError, ingest_local_document
from src.document_vault import (
    DocumentCountLimitError,
    DocumentStorageError,
    DocumentVault,
    DuplicateDocumentHashError,
)
from src.memory import MemoryRecord, MemoryTextTooLongError, MemoryValidationError
from src.memory_store import MemoryStorageError, MemoryStore
from src.settings import Settings

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
  /about                - Describe Project Cortana and this software milestone
  /exit                 - End the session cleanly"""

ABOUT_TEXT = (
    "Cortana: Project Cortana is an AI-powered authorized cybersecurity and "
    "defensive-operations assistant. This build is an early software milestone "
    "focused on identity, local commands, in-session conversation, explicit "
    "user-controlled persistent memory, temporary active memory context, and a "
    "local Knowledge Vault for explicit document ingestion."
)

CLEAR_CONFIRMATION = "Cortana: Conversation history cleared."
CLEAR_ALREADY_EMPTY = "Cortana: Conversation history is already empty."

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


CommandHandler = Callable[[CommandContext], CommandResult]


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
) -> CommandResult:
    """Handle a slash command locally without calling the AI service."""
    command_name = normalize_command_name(message)
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
        )
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)
    except DocumentStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    return CommandResult(outcome=CommandOutcome.CONTINUE, message=status_text)


def _handle_clear(context: CommandContext) -> CommandResult:
    """Clear only temporary conversation history."""
    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=clear_conversation_history(context.conversation_history),
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
}


def format_status(
    settings: Settings,
    conversation_history: ConversationHistory,
    memory_store: MemoryStore,
    active_memory_context: ActiveMemoryContext,
    document_vault: DocumentVault,
) -> str:
    """Build safe local session status text for /status."""
    completed_turns = conversation_history.completed_turn_count
    max_turns = conversation_history.max_completed_turns
    saved_memory_count = len(memory_store.list_memories())
    active_count = active_memory_context.active_count
    active_characters = active_memory_context.total_character_usage
    stored_document_count = document_vault.document_count()
    supported_types = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))

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
        "  Document context injection: "
        f"{'enabled' if DOCUMENT_CONTEXT_INJECTION_ENABLED else 'disabled'}"
    )


def clear_conversation_history(conversation_history: ConversationHistory) -> str:
    """Clear active in-memory history and return a user-facing confirmation."""
    if not conversation_history.turns:
        return CLEAR_ALREADY_EMPTY

    conversation_history.clear()
    return CLEAR_CONFIRMATION
