"""In-session conversation state for Project Cortana."""

from dataclasses import dataclass, field
from typing import Literal, TypedDict

EXIT_COMMANDS = frozenset({"exit", "quit", "goodbye"})

DEFAULT_MAX_COMPLETED_TURNS = 20

STARTUP_GREETING = (
    "Cortana: Hello. I am Cortana, your cybersecurity assistant. "
    "Type your message, or say exit, quit, or goodbye to end the session."
)

SHUTDOWN_MESSAGE = "Cortana: Goodbye. Stay secure."


def is_exit_command(message: str) -> bool:
    """Return True when the trimmed message is a session exit command."""
    return message.strip().lower() in EXIT_COMMANDS


@dataclass(frozen=True)
class ConversationTurn:
    """One message in an active Cortana conversation."""

    role: Literal["user", "assistant"]
    content: str


class ConversationApiMessage(TypedDict):
    """Structured message entry for the OpenAI Responses API."""

    role: Literal["user", "assistant"]
    content: str


ConversationApiInput = str | list[ConversationApiMessage]


@dataclass
class ConversationHistory:
    """In-memory conversation history for the current session."""

    max_completed_turns: int = DEFAULT_MAX_COMPLETED_TURNS
    _turns: list[ConversationTurn] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Record a user message."""
        self._turns.append(ConversationTurn(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response and trim oldest completed turns."""
        self._turns.append(ConversationTurn(role="assistant", content=content))
        self._trim_completed_turns()

    @property
    def turns(self) -> list[ConversationTurn]:
        """Return a copy of recorded conversation turns."""
        return list(self._turns)

    @property
    def completed_turn_count(self) -> int:
        """Return the number of completed user/assistant pairs in history."""
        return self._completed_turn_count()

    def clear(self) -> None:
        """Remove all in-memory conversation turns for the active session."""
        self._turns.clear()

    def _trim_completed_turns(self) -> None:
        """Remove the oldest completed user/assistant pairs when over the limit."""
        while self._completed_turn_count() > self.max_completed_turns:
            if (
                len(self._turns) >= 2
                and self._turns[0].role == "user"
                and self._turns[1].role == "assistant"
            ):
                del self._turns[0:2]
                continue
            break

    def _completed_turn_count(self) -> int:
        """Count completed user/assistant pairs in stored history."""
        completed_turns = 0
        index = 0
        while index + 1 < len(self._turns):
            if (
                self._turns[index].role == "user"
                and self._turns[index + 1].role == "assistant"
            ):
                completed_turns += 1
                index += 2
                continue
            index += 1
        return completed_turns


def _turn_to_api_message(turn: ConversationTurn) -> ConversationApiMessage:
    """Convert one stored turn into a structured API message entry."""
    return {"role": turn.role, "content": turn.content}


def build_conversation_input(
    history: ConversationHistory,
    user_message: str,
) -> ConversationApiInput:
    """Build structured API input from session history and the current message."""
    cleaned_message = user_message.strip()

    if not history.turns:
        return cleaned_message

    messages: list[ConversationApiMessage] = [
        _turn_to_api_message(turn) for turn in history.turns
    ]
    messages.append({"role": "user", "content": cleaned_message})
    return messages
