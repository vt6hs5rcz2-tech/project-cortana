"""Local slash-command framework for Project Cortana."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from src.config import (
    EXPLICIT_PERSISTENT_MEMORY_ENABLED,
    HISTORY_PERSISTENCE_ENABLED,
    MAX_MEMORY_TEXT_LENGTH,
)
from src.conversation import ConversationHistory
from src.memory import MemoryTextTooLongError, MemoryValidationError
from src.memory_store import MemoryStorageError, MemoryStore
from src.settings import Settings

COMMAND_HELP = "help"
COMMAND_STATUS = "status"
COMMAND_CLEAR = "clear"
COMMAND_ABOUT = "about"
COMMAND_EXIT = "exit"
COMMAND_REMEMBER = "remember"
COMMAND_MEMORIES = "memories"
COMMAND_FORGET = "forget"
COMMAND_FORGET_ALL = "forget-all"

FORGET_ALL_CONFIRM_TOKEN = "confirm"

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
    }
)

HELP_TEXT = """Cortana: Available commands:
  /help       - List available commands and brief descriptions
  /status     - Show safe local session information
  /clear      - Clear in-memory conversation history for this session
  /remember   - Save one explicit persistent memory
  /memories   - List saved persistent memories
  /forget     - Delete one saved memory by ID
  /forget-all - Delete all saved memories after confirmation
  /about      - Describe Project Cortana and this software milestone
  /exit       - End the session cleanly"""

ABOUT_TEXT = (
    "Cortana: Project Cortana is an AI-powered authorized cybersecurity and "
    "defensive-operations assistant. This build is an early software milestone "
    "focused on identity, local commands, in-session conversation, and "
    "explicit user-controlled persistent memory."
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
        )
    except MemoryStorageError as error:
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
    """Save one explicit persistent memory."""
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
    """Delete one saved memory by ID."""
    memory_id = extract_command_argument(context.message).strip()
    if not memory_id:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_MISSING_ID,
        )

    try:
        deleted = context.memory_store.delete_memory(memory_id)
    except MemoryStorageError as error:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=error.user_message)

    if not deleted:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_NOT_FOUND_TEMPLATE.format(memory_id=memory_id),
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=FORGET_SUCCESS_TEMPLATE.format(memory_id=memory_id),
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

    if deleted_count == 0:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=FORGET_ALL_ALREADY_EMPTY,
        )

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=FORGET_ALL_SUCCESS,
    )


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
}


def format_status(
    settings: Settings,
    conversation_history: ConversationHistory,
    memory_store: MemoryStore,
) -> str:
    """Build safe local session status text for /status."""
    completed_turns = conversation_history.completed_turn_count
    max_turns = conversation_history.max_completed_turns
    saved_memory_count = len(memory_store.list_memories())

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
        f"  Saved memories: {saved_memory_count}"
    )


def clear_conversation_history(conversation_history: ConversationHistory) -> str:
    """Clear active in-memory history and return a user-facing confirmation."""
    if not conversation_history.turns:
        return CLEAR_ALREADY_EMPTY

    conversation_history.clear()
    return CLEAR_CONFIRMATION
