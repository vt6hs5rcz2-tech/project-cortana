"""Bounded user-facing message helpers.

Domain validation messages written for the operator may be shown. Internal,
provider, and storage exceptions must map to a fixed fallback so raw paths,
secrets, and implementation details never reach the console.
"""

from __future__ import annotations


def cortana_domain_message(error: BaseException, *, fallback: str) -> str:
    """Prefix a controlled domain-validation message, or return ``fallback``."""
    text = str(error).strip()
    if text.startswith("Cortana:"):
        return text
    if text:
        return f"Cortana: {text}"
    return fallback
