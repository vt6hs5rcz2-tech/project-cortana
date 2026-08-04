"""Local slash-command framework for Project Cortana."""

from dataclasses import dataclass
from enum import Enum

from src.config import HISTORY_PERSISTENCE_ENABLED
from src.conversation import ConversationHistory
from src.settings import Settings

COMMAND_HELP = "help"
COMMAND_STATUS = "status"
COMMAND_CLEAR = "clear"
COMMAND_ABOUT = "about"
COMMAND_EXIT = "exit"

SUPPORTED_COMMANDS = frozenset(
    {
        COMMAND_HELP,
        COMMAND_STATUS,
        COMMAND_CLEAR,
        COMMAND_ABOUT,
        COMMAND_EXIT,
    }
)

HELP_TEXT = """Cortana: Available commands:
  /help   - List available commands and brief descriptions
  /status - Show safe local session information
  /clear  - Clear in-memory conversation history for this session
  /about  - Describe Project Cortana and this software milestone
  /exit   - End the session cleanly"""

ABOUT_TEXT = (
    "Cortana: Project Cortana is an AI-powered authorized cybersecurity and "
    "defensive-operations assistant. This build is an early software milestone "
    "focused on identity, local commands, and in-session conversation — "
    "without persistent long-term memory yet."
)

CLEAR_CONFIRMATION = "Cortana: Conversation history cleared."
CLEAR_ALREADY_EMPTY = "Cortana: Conversation history is already empty."

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


def handle_slash_command(
    message: str,
    *,
    settings: Settings,
    conversation_history: ConversationHistory,
) -> CommandResult:
    """Handle a slash command locally without calling the AI service."""
    command_name = normalize_command_name(message)

    if command_name == COMMAND_HELP:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=HELP_TEXT)

    if command_name == COMMAND_STATUS:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=format_status(settings, conversation_history),
        )

    if command_name == COMMAND_CLEAR:
        return CommandResult(
            outcome=CommandOutcome.CONTINUE,
            message=clear_conversation_history(conversation_history),
        )

    if command_name == COMMAND_ABOUT:
        return CommandResult(outcome=CommandOutcome.CONTINUE, message=ABOUT_TEXT)

    if command_name == COMMAND_EXIT:
        return CommandResult(outcome=CommandOutcome.EXIT)

    return CommandResult(
        outcome=CommandOutcome.CONTINUE,
        message=UNKNOWN_COMMAND_TEMPLATE.format(command=f"/{command_name}"),
    )


def format_status(
    settings: Settings,
    conversation_history: ConversationHistory,
) -> str:
    """Build safe local session status text for /status."""
    completed_turns = conversation_history.completed_turn_count
    max_turns = conversation_history.max_completed_turns

    return (
        "Cortana: Session status\n"
        "  Status: online\n"
        f"  Model: {settings.openai_model}\n"
        f"  Retained completed turns: {completed_turns}\n"
        f"  Maximum retained turns: {max_turns}\n"
        "  History persistence: "
        f"{'enabled' if HISTORY_PERSISTENCE_ENABLED else 'disabled'}"
    )


def clear_conversation_history(conversation_history: ConversationHistory) -> str:
    """Clear active in-memory history and return a user-facing confirmation."""
    if not conversation_history.turns:
        return CLEAR_ALREADY_EMPTY

    conversation_history.clear()
    return CLEAR_CONFIRMATION
