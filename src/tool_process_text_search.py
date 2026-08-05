"""Streaming literal text-search over a verified Windows safe-open handle.

Uses incremental UTF-8 decoding and bounded line assembly. Does not buffer the
full file and does not reopen by path.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass

from src.config import (
    MAX_TOOL_FILE_BYTES,
    MAX_TOOL_TEXT_SEARCH_MATCHES,
    MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS,
    MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS,
    MAX_TOOL_TEXT_SEARCH_QUERY_CHARS,
)
from src.tool_process_safe_open import (
    FileChangedDuringRead,
    FileTooLarge,
    SafeOpenError,
    SafeOpenHandle,
    assert_safe_handle_unchanged_after_read,
    read_raw_chunk_from_safe_handle,
)
from src.tool_safe_files import HASH_CHUNK_SIZE


@dataclass(frozen=True)
class TextSearchMatch:
    """One bounded literal match from a streaming text search."""

    line_number: int
    preview: str


@dataclass(frozen=True)
class TextSearchResult:
    """Immutable streaming text-search result."""

    matches: tuple[TextSearchMatch, ...]
    match_count: int
    truncated: bool
    size_bytes: int


def format_text_search_preview(
    line: str,
    *,
    max_preview_chars: int = MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS,
) -> str:
    """Format one match preview using the existing in-process semantics."""
    preview = line.strip()
    if len(preview) > max_preview_chars:
        return preview[:max_preview_chars] + "..."
    return preview


def line_contains_literal(line: str, query: str) -> bool:
    """Return whether the line contains the literal case-sensitive query."""
    return query in line


def stream_text_search_from_safe_handle(
    opened: SafeOpenHandle,
    *,
    query: str,
    max_matches: int,
    max_bytes: int = MAX_TOOL_FILE_BYTES,
    chunk_size: int = HASH_CHUNK_SIZE,
    max_pending_line_chars: int = MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS,
    max_preview_chars: int = MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS,
) -> TextSearchResult:
    """Stream a literal line-oriented search through a verified safe handle.

    Does not close the handle. Continues reading through EOF after
    ``max_matches`` so post-read change detection remains meaningful.
    """
    _validate_stream_args(
        query=query,
        max_matches=max_matches,
        max_bytes=max_bytes,
        chunk_size=chunk_size,
        max_pending_line_chars=max_pending_line_chars,
        max_preview_chars=max_preview_chars,
    )
    baseline = opened.identity
    if baseline.size_bytes > max_bytes:
        raise FileTooLarge("File exceeds the maximum permitted size.")

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    discard_mode = False
    line_number = 0
    matches: list[TextSearchMatch] = []
    collecting = True
    total = 0
    overlap_limit = max(0, len(query) - 1)

    try:
        while True:
            raw = read_raw_chunk_from_safe_handle(opened, chunk_size=chunk_size)
            if not raw:
                break
            total += len(raw)
            if total > max_bytes:
                raise FileTooLarge("File exceeds the maximum permitted size.")
            text = decoder.decode(raw, final=False)
            (
                pending,
                discard_mode,
                line_number,
                collecting,
            ) = _consume_decoded_text(
                text,
                pending=pending,
                discard_mode=discard_mode,
                line_number=line_number,
                query=query,
                matches=matches,
                collecting=collecting,
                max_matches=max_matches,
                max_pending_line_chars=max_pending_line_chars,
                max_preview_chars=max_preview_chars,
                overlap_limit=overlap_limit,
                flush=False,
            )

        # Flush decoder state exactly once at true EOF.
        flushed = decoder.decode(b"", final=True)
        (
            pending,
            discard_mode,
            line_number,
            collecting,
        ) = _consume_decoded_text(
            flushed,
            pending=pending,
            discard_mode=discard_mode,
            line_number=line_number,
            query=query,
            matches=matches,
            collecting=collecting,
            max_matches=max_matches,
            max_pending_line_chars=max_pending_line_chars,
            max_preview_chars=max_preview_chars,
            overlap_limit=overlap_limit,
            flush=True,
        )
        assert_safe_handle_unchanged_after_read(
            opened,
            total_bytes_read=total,
            max_bytes=max_bytes,
        )
    except (FileTooLarge, FileChangedDuringRead, SafeOpenError):
        raise
    except Exception as error:
        raise SafeOpenError("Secure text search failed.") from error

    truncated = len(matches) >= max_matches
    return TextSearchResult(
        matches=tuple(matches),
        match_count=len(matches),
        truncated=truncated,
        size_bytes=total,
    )


def _validate_stream_args(
    *,
    query: str,
    max_matches: int,
    max_bytes: int,
    chunk_size: int,
    max_pending_line_chars: int,
    max_preview_chars: int,
) -> None:
    if not isinstance(query, str) or not query:
        raise SafeOpenError("A non-empty query is required.")
    if len(query) > MAX_TOOL_TEXT_SEARCH_QUERY_CHARS:
        raise SafeOpenError("Query exceeds the maximum length.")
    if (
        isinstance(max_matches, bool)
        or not isinstance(max_matches, int)
        or max_matches < 1
        or max_matches > MAX_TOOL_TEXT_SEARCH_MATCHES
    ):
        raise SafeOpenError("max_matches is outside the allowed range.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise SafeOpenError("max_bytes must be a positive integer.")
    if max_bytes > MAX_TOOL_FILE_BYTES:
        raise SafeOpenError("max_bytes cannot exceed MAX_TOOL_FILE_BYTES.")
    if chunk_size != HASH_CHUNK_SIZE:
        raise SafeOpenError("chunk_size must equal the centralized HASH_CHUNK_SIZE.")
    if (
        isinstance(max_pending_line_chars, bool)
        or not isinstance(max_pending_line_chars, int)
        or max_pending_line_chars < 1
    ):
        raise SafeOpenError("max_pending_line_chars must be a positive integer.")
    if max_pending_line_chars != MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS:
        raise SafeOpenError(
            "max_pending_line_chars must equal the centralized pending-line bound."
        )
    if (
        isinstance(max_preview_chars, bool)
        or not isinstance(max_preview_chars, int)
        or max_preview_chars < 1
    ):
        raise SafeOpenError("max_preview_chars must be a positive integer.")


def _consume_decoded_text(
    text: str,
    *,
    pending: str,
    discard_mode: bool,
    line_number: int,
    query: str,
    matches: list[TextSearchMatch],
    collecting: bool,
    max_matches: int,
    max_pending_line_chars: int,
    max_preview_chars: int,
    overlap_limit: int,
    flush: bool,
) -> tuple[str, bool, int, bool]:
    pending = pending + text
    while True:
        if discard_mode:
            pending, discard_mode, line_number, collecting = _consume_discard_mode(
                pending,
                discard_mode=discard_mode,
                line_number=line_number,
                query=query,
                matches=matches,
                collecting=collecting,
                max_matches=max_matches,
                max_preview_chars=max_preview_chars,
                overlap_limit=overlap_limit,
                flush=flush,
            )
            if discard_mode:
                if flush and pending:
                    # Final unterminated oversized line already searched via overlap.
                    pending = ""
                break
            continue

        ending = _find_line_ending(pending, allow_terminal_cr=flush)
        if ending is None:
            if len(pending) > max_pending_line_chars:
                pending, discard_mode, line_number, collecting = _enter_discard_mode(
                    pending,
                    line_number=line_number,
                    query=query,
                    matches=matches,
                    collecting=collecting,
                    max_matches=max_matches,
                    max_pending_line_chars=max_pending_line_chars,
                    max_preview_chars=max_preview_chars,
                    overlap_limit=overlap_limit,
                )
                continue
            if flush and pending:
                line_number, collecting = _record_line(
                    pending,
                    line_number=line_number,
                    query=query,
                    matches=matches,
                    collecting=collecting,
                    max_matches=max_matches,
                    max_preview_chars=max_preview_chars,
                )
                pending = ""
            break

        line, pending = _split_at_ending(pending, ending)
        line_number, collecting = _record_line(
            line,
            line_number=line_number,
            query=query,
            matches=matches,
            collecting=collecting,
            max_matches=max_matches,
            max_preview_chars=max_preview_chars,
        )
    return pending, discard_mode, line_number, collecting


def _enter_discard_mode(
    pending: str,
    *,
    line_number: int,
    query: str,
    matches: list[TextSearchMatch],
    collecting: bool,
    max_matches: int,
    max_pending_line_chars: int,
    max_preview_chars: int,
    overlap_limit: int,
) -> tuple[str, bool, int, bool]:
    prefix = pending[:max_pending_line_chars]
    remainder = pending[max_pending_line_chars:]
    line_number += 1
    collecting = _maybe_record_match(
        prefix,
        line_number=line_number,
        query=query,
        matches=matches,
        collecting=collecting,
        max_matches=max_matches,
        max_preview_chars=max_preview_chars,
    )
    overlap = prefix[-overlap_limit:] if overlap_limit else ""
    return overlap + remainder, True, line_number, collecting


def _consume_discard_mode(
    pending: str,
    *,
    discard_mode: bool,
    line_number: int,
    query: str,
    matches: list[TextSearchMatch],
    collecting: bool,
    max_matches: int,
    max_preview_chars: int,
    overlap_limit: int,
    flush: bool,
) -> tuple[str, bool, int, bool]:
    del discard_mode  # Caller only invokes this while discard_mode is True.
    ending = _find_line_ending(pending, allow_terminal_cr=flush)
    if ending is None:
        collecting = _maybe_record_match(
            pending,
            line_number=line_number,
            query=query,
            matches=matches,
            collecting=collecting,
            max_matches=max_matches,
            max_preview_chars=max_preview_chars,
        )
        pending = pending[-overlap_limit:] if overlap_limit else ""
        return pending, True, line_number, collecting

    before, after = _split_at_ending(pending, ending)
    collecting = _maybe_record_match(
        before,
        line_number=line_number,
        query=query,
        matches=matches,
        collecting=collecting,
        max_matches=max_matches,
        max_preview_chars=max_preview_chars,
    )
    # Line number was already advanced when discard mode started.
    return after, False, line_number, collecting


def _record_line(
    line: str,
    *,
    line_number: int,
    query: str,
    matches: list[TextSearchMatch],
    collecting: bool,
    max_matches: int,
    max_preview_chars: int,
) -> tuple[int, bool]:
    line_number += 1
    collecting = _maybe_record_match(
        line,
        line_number=line_number,
        query=query,
        matches=matches,
        collecting=collecting,
        max_matches=max_matches,
        max_preview_chars=max_preview_chars,
    )
    return line_number, collecting


def _maybe_record_match(
    line: str,
    *,
    line_number: int,
    query: str,
    matches: list[TextSearchMatch],
    collecting: bool,
    max_matches: int,
    max_preview_chars: int,
) -> bool:
    if not collecting or not query or not line_contains_literal(line, query):
        return collecting
    if matches and matches[-1].line_number == line_number:
        return collecting
    matches.append(
        TextSearchMatch(
            line_number=line_number,
            preview=format_text_search_preview(
                line,
                max_preview_chars=max_preview_chars,
            ),
        )
    )
    if len(matches) >= max_matches:
        return False
    return collecting


def _find_line_ending(text: str, *, allow_terminal_cr: bool) -> int | None:
    """Return the index of the next line-ending character, or None.

    ``\\r\\n`` is treated as a single newline ending (the ``\\n`` index).
    A lone ``\\r`` is a line ending, matching ``str.splitlines()`` closely.
    """
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if ch == "\n":
            return index
        if ch == "\r":
            if index + 1 < length and text[index + 1] == "\n":
                # Prefer the \n index so _split_at_ending can strip \r\n.
                return index + 1
            if index == length - 1 and not allow_terminal_cr:
                return None
            return index
        index += 1
    return None


def _split_at_ending(text: str, ending_index: int) -> tuple[str, str]:
    ch = text[ending_index]
    if ch == "\n":
        line = text[:ending_index]
        if line.endswith("\r"):
            line = line[:-1]
        return line, text[ending_index + 1 :]
    # Lone \r line ending (splitlines-compatible).
    return text[:ending_index], text[ending_index + 1 :]
