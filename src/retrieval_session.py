"""Session-only retrieval state for source-grounded document answers."""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field

from src.document_retrieval import SourceManifestEntry

_BOUNDARY_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_BOUNDARY_TOKEN_BYTES = 16


class RetrievalSessionError(ValueError):
    """Base error for retrieval-session operations."""


class InvalidDocumentBoundaryTokenError(RetrievalSessionError):
    """Raised when a document-context boundary token is unsafe or malformed."""


def generate_document_boundary_token() -> str:
    """Generate an unpredictable boundary token safe for text markers."""
    return secrets.token_hex(_BOUNDARY_TOKEN_BYTES)


def validate_document_boundary_token(token: str) -> str:
    """Validate and return a document-context boundary token."""
    if not _BOUNDARY_TOKEN_PATTERN.fullmatch(token):
        raise InvalidDocumentBoundaryTokenError(
            "Document boundary token must be 16-64 characters of "
            "letters, digits, underscore, or hyphen."
        )
    return token


def fingerprint_query(query: str) -> str:
    """Return a short fingerprint for a query without retaining full text."""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class RetrievalSession:
    """In-memory retrieval session state that is never persisted to disk."""

    boundary_token: str = field(default="")
    _query_fingerprint: str | None = field(default=None, init=False)
    _source_manifest: list[SourceManifestEntry] = field(
        default_factory=list,
        init=False,
    )
    _valid_citation_labels: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        """Initialize or validate the session-only document boundary token."""
        if self.boundary_token:
            self.boundary_token = validate_document_boundary_token(
                self.boundary_token
            )
        else:
            self.boundary_token = generate_document_boundary_token()

    @property
    def query_fingerprint(self) -> str | None:
        """Return the fingerprint of the latest successful grounded query."""
        return self._query_fingerprint

    @property
    def source_manifest(self) -> list[SourceManifestEntry]:
        """Return a copy of the latest source manifest."""
        return list(self._source_manifest)

    @property
    def valid_citation_labels(self) -> frozenset[str]:
        """Return the latest valid citation-label set."""
        return frozenset(self._valid_citation_labels)

    @property
    def has_source_manifest(self) -> bool:
        """Return True when a source manifest exists for this session."""
        return bool(self._source_manifest)

    def record_grounded_result(
        self,
        *,
        query: str,
        source_manifest: list[SourceManifestEntry] | tuple[SourceManifestEntry, ...],
        citation_labels: set[str] | frozenset[str],
    ) -> None:
        """Store the latest successful grounded-request source metadata."""
        self._query_fingerprint = fingerprint_query(query)
        self._source_manifest = list(source_manifest)
        self._valid_citation_labels = set(citation_labels)

    def clear(self) -> None:
        """Clear query fingerprint, source manifest, and citation labels."""
        self._query_fingerprint = None
        self._source_manifest.clear()
        self._valid_citation_labels.clear()

    def remove_document(self, document_id: str) -> int:
        """Remove or invalidate manifest entries for a deleted document."""
        target = document_id.strip()
        if not target:
            return 0

        remaining = [
            entry
            for entry in self._source_manifest
            if entry.document_id != target
        ]
        removed = len(self._source_manifest) - len(remaining)
        if removed == 0:
            return 0

        self._source_manifest = remaining
        self._valid_citation_labels = {
            entry.citation_label for entry in remaining
        }
        if not remaining:
            self._query_fingerprint = None
        return removed
