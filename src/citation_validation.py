"""Lightweight citation-label validation for source-grounded AI answers.

This module verifies that citation labels appearing in a model response match
the exact labels supplied with the retrieved-document context for the request.
It does not prove factual entailment of natural-language claims against chunk
text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

CITATION_LABEL_PATTERN = re.compile(r"\[DOC-\d+:C\d+\]")
UNSUPPORTED_CITATION_MARKER = "[UNSUPPORTED-CITATION]"
CITATION_WARNING = (
    "Warning: The response included citation labels that were not supplied "
    "with the retrieved evidence. Unsupported labels were marked."
)


@dataclass(frozen=True)
class CitationValidationResult:
    """Outcome of validating citation labels in an AI response.

    ``validates_entailment`` is always False: this layer checks source-label
    validity only, not full factual entailment of claims.
    """

    original_response: str
    sanitized_response: str
    found_labels: tuple[str, ...]
    valid_labels: tuple[str, ...]
    invalid_labels: tuple[str, ...]
    is_valid: bool
    validates_entailment: bool = False


def extract_citation_labels(text: str) -> tuple[str, ...]:
    """Extract citation-like labels from response text in appearance order."""
    if not isinstance(text, str):
        return ()
    return tuple(CITATION_LABEL_PATTERN.findall(text))


def validate_response_citations(
    response_text: str,
    allowed_labels: Iterable[str],
) -> CitationValidationResult:
    """Validate citation labels against the exact supplied label set.

    Invalid labels are replaced with ``UNSUPPORTED_CITATION_MARKER``. Valid
    labels are preserved. A warning is appended when any unsupported label was
    present. This does not validate factual entailment.
    """
    allowed = frozenset(allowed_labels)
    found = extract_citation_labels(response_text)
    invalid = tuple(
        label for label in dict.fromkeys(found) if label not in allowed
    )
    valid = tuple(
        label for label in dict.fromkeys(found) if label in allowed
    )

    sanitized = response_text
    for label in invalid:
        sanitized = sanitized.replace(label, UNSUPPORTED_CITATION_MARKER)

    is_valid = not invalid
    if not is_valid:
        if sanitized and not sanitized.endswith("\n"):
            sanitized = f"{sanitized}\n"
        sanitized = f"{sanitized}{CITATION_WARNING}"

    return CitationValidationResult(
        original_response=response_text,
        sanitized_response=sanitized,
        found_labels=found,
        valid_labels=valid,
        invalid_labels=invalid,
        is_valid=is_valid,
        validates_entailment=False,
    )
