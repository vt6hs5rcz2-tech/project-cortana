"""Pure path-argument parsing helpers with no domain dependencies."""


def strip_path_argument_quotes(argument: str) -> str:
    """Remove a single matching pair of surrounding quotes from a path argument."""
    stripped = argument.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped
