"""Pure slash-command argument parsing helpers with no domain dependencies."""


def extract_command_argument(message: str) -> str:
    """Return the raw argument text after the command token."""
    stripped = message.strip()
    parts = stripped.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1]
