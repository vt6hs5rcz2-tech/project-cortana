"""Tests for citation-label validation of grounded AI responses."""

from src.citation_validation import (
    CITATION_WARNING,
    UNSUPPORTED_CITATION_MARKER,
    extract_citation_labels,
    validate_response_citations,
)


def test_valid_citations_preserved_and_invented_detected() -> None:
    """Valid labels remain; invented labels are marked unsupported."""
    allowed = {"[DOC-1:C1]", "[DOC-1:C2]"}
    response = "Use [DOC-1:C1] and invented [DOC-9:C9] carefully."

    result = validate_response_citations(response, allowed)

    assert result.valid_labels == ("[DOC-1:C1]",)
    assert result.invalid_labels == ("[DOC-9:C9]",)
    assert "[DOC-1:C1]" in result.sanitized_response
    assert UNSUPPORTED_CITATION_MARKER in result.sanitized_response
    assert "[DOC-9:C9]" not in result.sanitized_response
    assert CITATION_WARNING in result.sanitized_response
    assert result.is_valid is False
    assert result.validates_entailment is False


def test_mixed_valid_invalid_and_no_citations() -> None:
    """Mixed and citation-free responses should behave predictably."""
    allowed = {"[DOC-1:C1]", "[DOC-2:C1]"}

    mixed = validate_response_citations(
        "See [DOC-1:C1] and [DOC-3:C1].",
        allowed,
    )
    assert mixed.valid_labels == ("[DOC-1:C1]",)
    assert mixed.invalid_labels == ("[DOC-3:C1]",)

    none = validate_response_citations("No citations here.", allowed)
    assert none.found_labels == ()
    assert none.is_valid is True
    assert none.sanitized_response == "No citations here."
    assert none.validates_entailment is False


def test_extract_citation_labels_order() -> None:
    """Extraction should preserve appearance order."""
    labels = extract_citation_labels("A [DOC-2:C3] then [DOC-1:C1] again [DOC-2:C3]")
    assert labels == ("[DOC-2:C3]", "[DOC-1:C1]", "[DOC-2:C3]")
