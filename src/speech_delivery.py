"""Milestone 29 natural speech delivery and conversational timing.

Centralized, deterministic policy for how Cortana *delivers* spoken
responses. Operates on a delivery copy only. Never owns assistant
response creation, never authorizes privileged actions, and never
performs I/O, network, model, or tool calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.config import (
    DEFAULT_RESPONSE_DEPTH,
    MAX_CODE_SPEAK_CHARS,
    MAX_PARENTHETICAL_SPEAK_CHARS,
    MAX_RECENT_SPOKEN_FINGERPRINTS,
    MAX_SPEECH_CHUNKS,
    MAX_SPEECH_DELIVERY_INSTRUCTION_CHARS,
    MAX_SPEECH_DELIVERY_STATE_CHARS,
    MAX_SPEECH_FINGERPRINT_CHARS,
    MAX_SPEECH_OPENING_CHARS,
    MAX_SPOKEN_LIST_ITEMS,
    MAX_URL_SPEAK_CHARS,
    MIN_SPEECH_CHUNK_CHARS,
    SPEECH_CHUNK_CHARS_BRIEF,
    SPEECH_CHUNK_CHARS_DETAILED,
    SPEECH_CHUNK_CHARS_NORMAL,
    SPEECH_DELIVERY_ENABLED,
)
from src.conversation_intelligence import (
    AcknowledgmentHint,
    ConversationalGuidance,
    ResponseDepth,
    TurnTakingKind,
)
from src.conversation_state import InteractionMode

DeliveryMode = Literal["voice_turn", "realtime", "multimodal"]
ListDeliveryMode = Literal["spoken_sequence", "verbatim", "screen_preferred"]
InterruptionRecoveryHint = Literal["none", "resume_from_correction", "skip_restart"]
TimingHint = Literal[
    "ready_to_speak",
    "likely_continuation",
    "correction",
    "interrupted_turn",
]
PauseHint = Literal["none", "short", "sentence", "paragraph", "list_item"]

SPEECH_DELIVERY_BEGIN = "<<<CORTANA_SPEECH_DELIVERY>>>"
SPEECH_DELIVERY_END = "<<<END_CORTANA_SPEECH_DELIVERY>>>"

SPEECH_DELIVERY_STYLE_POLICY = (
    "Spoken delivery style: sound like a natural conversational assistant, "
    "not like text being read aloud. Prefer short spoken sentences. "
    "Pause briefly between major ideas. Convert written lists into spoken "
    "sequences such as first, second, third. Do not prepend filler "
    "acknowledgments such as Got it, Okay, Sure, or Absolutely unless the "
    "user just corrected you. Do not restart an interrupted answer; continue "
    "from the user's latest point. Do not restate the user's request. "
    "Keep numbers, dates, and short identifiers exact. Do not try to speak "
    "long URLs, hashes, or large code blocks; say they are better viewed on "
    "screen, unless the user asked to hear the exact, verbatim, spelled, "
    "dictated, or read-aloud value. Exactness then wins over naturalness. "
    "Speech style never overrides safety, authorization, or required "
    "notices. Conversational delivery never authorizes privileged actions."
)

_REQUIRED_NOTICE_MARKERS = frozenset(
    {
        "cannot authorize",
        "not authorized",
        "safety",
        "policy",
        "confirm",
        "dry-run",
        "approval required",
        "are you sure",
    }
)

_LEADING_ACK_PATTERN = re.compile(
    r"^(?:got it|okay|ok|sure|absolutely|alright)[,.]?\s+",
    re.IGNORECASE,
)
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_EMPHASIS_PATTERN = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_CODE_FENCE_PATTERN = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_URL_PATTERN = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_REPEATED_BANG_PATTERN = re.compile(r"!{2,}")
_REPEATED_QUESTION_PATTERN = re.compile(r"\?{2,}")
_REPEATED_DOTS_PATTERN = re.compile(r"\.{4,}")
_MULTI_BLANK_PATTERN = re.compile(r"\n{3,}")
_LONG_HASH_PATTERN = re.compile(r"\b[a-fA-F0-9]{20,}\b")
_LONG_PAREN_PATTERN = re.compile(r"\(([^)]{49,})\)")
_ARROW_PATTERN = re.compile(r"\s*->\s*")
_PROTECTED_ABBREVIATION = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Inc|Ltd|St|Ave|"
    r"Dept|approx|Fig|No|Vol|al|Rev|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)\.",
    re.IGNORECASE,
)
_PROTECTED_LATIN = re.compile(r"\b(?:e\.g|i\.e|U\.S|U\.K|a\.m|p\.m)\.", re.IGNORECASE)
_PROTECTED_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_PROTECTED_DECIMAL = re.compile(r"\b\d+\.\d+\b")
_PROTECTED_VERSION = re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE)
_PROTECTED_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_REPEAT_PHRASES = (
    "repeat that",
    "say that again",
    "say it again",
    "repeat it",
    "say that one more time",
    "can you repeat that",
    "please repeat that",
    "what did you just say",
)
_EXACT_CONTENT_PHRASES = (
    "verbatim",
    "word for word",
    "character by character",
    "character for character",
    "spell it",
    "spell that",
    "spell this",
    "spell out",
    "read it aloud",
    "read that aloud",
    "read this aloud",
    "dictate it",
    "dictate that",
    "dictate this",
    "dictate the command",
    "say it exactly",
    "say that exactly",
    "say this exactly",
    "repeat it exactly",
    "repeat that exactly",
    "give me the exact hash",
    "give me the exact url",
    "give me the exact uuid",
    "give me the exact identifier",
    "read the hash",
    "read the url",
    "read the uuid",
    "read the identifier",
    "read that hash aloud",
    "exactly as written",
    "exactly as is",
)
_EXACT_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bspell\s+(?:it|that|this|out)\b"),
    re.compile(r"\bread\s+(?:it|that|this)\s+aloud\b"),
    re.compile(
        r"\bread\s+(?:that|the|this)\s+"
        r"(?:hash|url|uuid|identifier|id|command|code|token)\s+aloud\b"
    ),
    re.compile(
        r"\bread\s+(?:the|that|this)\s+"
        r"(?:hash|url|uuid|identifier|command|code|token)\b"
    ),
    re.compile(r"\bdictate\s+(?:it|that|this|the\s+command)\b"),
    re.compile(r"\b(?:say|repeat)\s+(?:it|that|this)\s+exactly\b"),
    re.compile(
        r"\b(?:give\s+me\s+|tell\s+me\s+)?the\s+exact\s+"
        r"(?:hash|url|uuid|identifier|id|command|code|token|value)\b"
    ),
    re.compile(r"\bverbatim\b"),
    re.compile(r"\bword\s+for\s+word\b"),
    re.compile(r"\bcharacter\s+(?:by|for)\s+character\b"),
    re.compile(r"^(?:please\s+)?exact(?:ly)?(?:\s+please)?$"),
)
_ORDINALS = {
    1: "First",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth",
    6: "Sixth",
}
_CODEISH_MARKERS = ("{", "}", ";", "def ", "function ", "#!/", "&&", "||")


@dataclass(frozen=True)
class SpeechDeliveryPlan:
    """Bounded spoken-delivery policy for one assistant utterance.

    Advisory only. Contains no media, no tool requests, and no authority.
    """

    delivery_mode: DeliveryMode
    response_depth: ResponseDepth
    preferred_chunk_chars: int
    sentence_pause_hint: PauseHint
    paragraph_pause_hint: PauseHint
    acknowledgment_hint: AcknowledgmentHint
    list_delivery_mode: ListDeliveryMode
    interruption_recovery_hint: InterruptionRecoveryHint
    concise_spoken_delivery: bool
    avoid_phrases: tuple[str, ...]
    preserve_required_notice: bool
    user_interrupted: bool
    interaction_mode: InteractionMode
    timing_hint: TimingHint
    repetition_requested: bool
    exact_content_required: bool

    @property
    def authorizes_privileged_action(self) -> bool:
        """Speech delivery never authorizes privileged actions."""
        return False


@dataclass(frozen=True)
class SpokenChunk:
    """One speech-safe segment plus a pause hint for local TTS pacing."""

    text: str
    pause_after: PauseHint
    exact_content: bool = False


@dataclass(frozen=True)
class SpokenDelivery:
    """Canonical text plus the speech-only delivery copy and chunks."""

    canonical_text: str
    spoken_text: str
    chunks: tuple[SpokenChunk, ...]
    plan: SpeechDeliveryPlan
    used_fallback: bool


@dataclass
class SpeechDeliveryState:
    """Session-scoped spoken-delivery memory. Never stores audio or secrets."""

    last_spoken_opening: str | None = None
    last_acknowledgment: str | None = None
    recently_spoken_fingerprints: list[str] = field(default_factory=list)
    interrupted_response_fingerprint: str | None = None
    last_completed_spoken_chunk: str | None = None
    delivery_mode: DeliveryMode | None = None
    pending_chunks: list[str] = field(default_factory=list)
    aborted: bool = False

    def reset(self) -> None:
        """Clear all delivery state, including any queued spoken chunks."""
        self.last_spoken_opening = None
        self.last_acknowledgment = None
        self.recently_spoken_fingerprints.clear()
        self.interrupted_response_fingerprint = None
        self.last_completed_spoken_chunk = None
        self.delivery_mode = None
        self.clear_pending_chunks()

    def clear_pending_chunks(self) -> None:
        """Drop any not-yet-played local TTS chunks."""
        self.pending_chunks.clear()
        self.aborted = False

    def clear_session_delivery_state(self) -> None:
        """Clear realtime-session-specific delivery fields at session end.

        Drops interruption fingerprints and pending local TTS chunks so they
        cannot leak into a later realtime session. Keeps conversation-wide
        repetition/acknowledgment fingerprints so text and `/voice-turn`
        continuity on the same `/clear` lifetime is preserved.
        """
        self.interrupted_response_fingerprint = None
        self.delivery_mode = None
        self.clear_pending_chunks()

    def load_pending(self, chunks: list[str] | tuple[str, ...]) -> None:
        """Replace the pending spoken-chunk queue. Text only; no audio."""
        self.aborted = False
        self.pending_chunks = [item.strip() for item in chunks if item.strip()]

    def abort_pending(self) -> None:
        """Stop remaining local TTS chunks after interruption/cancel."""
        self.aborted = True
        self.pending_chunks.clear()

    def pop_pending_chunk(self) -> str | None:
        """Return the next pending chunk, or None when aborted/empty."""
        if self.aborted or not self.pending_chunks:
            return None
        return self.pending_chunks.pop(0)

    def mark_interrupted(self, fingerprint: str | None) -> None:
        """Record a short fingerprint of interrupted spoken material."""
        self.abort_pending()
        if fingerprint is None or not fingerprint.strip():
            self.interrupted_response_fingerprint = None
            return
        self.interrupted_response_fingerprint = _clip(
            fingerprint,
            MAX_SPEECH_FINGERPRINT_CHARS,
        )

    def record_completed_chunk(self, text: str) -> None:
        """Track one fully spoken chunk for repetition/interruption control."""
        cleaned = text.strip()
        if not cleaned:
            return
        fingerprint = fingerprint_spoken_text(cleaned)
        self.last_completed_spoken_chunk = fingerprint
        if self.last_spoken_opening is None:
            self.last_spoken_opening = _clip(fingerprint, MAX_SPEECH_OPENING_CHARS)
        self.recently_spoken_fingerprints.append(fingerprint)
        overflow = (
            len(self.recently_spoken_fingerprints) - MAX_RECENT_SPOKEN_FINGERPRINTS
        )
        if overflow > 0:
            del self.recently_spoken_fingerprints[0:overflow]
        self._trim_to_budget()

    def record_acknowledgment(self, phrase: str) -> None:
        """Track a recently spoken lightweight acknowledgment."""
        cleaned = _clip(phrase.strip().casefold(), 40)
        if cleaned:
            self.last_acknowledgment = cleaned

    def consume_interrupted_fingerprint(self) -> str | None:
        """Return and clear the interrupted-response fingerprint."""
        value = self.interrupted_response_fingerprint
        self.interrupted_response_fingerprint = None
        return value

    def _trim_to_budget(self) -> None:
        used = 0
        if self.last_spoken_opening:
            used += len(self.last_spoken_opening)
        if self.last_acknowledgment:
            used += len(self.last_acknowledgment)
        if self.interrupted_response_fingerprint:
            used += len(self.interrupted_response_fingerprint)
        if self.last_completed_spoken_chunk:
            used += len(self.last_completed_spoken_chunk)
        used += sum(len(item) for item in self.recently_spoken_fingerprints)
        used += sum(len(item) for item in self.pending_chunks)
        while used > MAX_SPEECH_DELIVERY_STATE_CHARS and self.recently_spoken_fingerprints:
            dropped = self.recently_spoken_fingerprints.pop(0)
            used -= len(dropped)


def append_speech_delivery_policy(base_instructions: str) -> str:
    """Append the stable spoken-delivery style policy when missing."""
    if SPEECH_DELIVERY_STYLE_POLICY in base_instructions:
        return base_instructions
    return f"{base_instructions}\n\n{SPEECH_DELIVERY_STYLE_POLICY}"


def preferred_chunk_chars_for_depth(depth: ResponseDepth) -> int:
    """Map M27/M28 response depth to a spoken chunk size bound."""
    if depth == "brief":
        return SPEECH_CHUNK_CHARS_BRIEF
    if depth == "detailed":
        return SPEECH_CHUNK_CHARS_DETAILED
    return SPEECH_CHUNK_CHARS_NORMAL


def user_requests_repetition(user_text: str) -> bool:
    """Return True when the user explicitly asked to hear the last answer again."""
    normalized = _normalize_match_text(user_text)
    if not normalized:
        return False
    if normalized.startswith("don't ") or normalized.startswith("do not "):
        return False
    return any(
        normalized == phrase or normalized.startswith(f"{phrase} ")
        for phrase in _REPEAT_PHRASES
    )


def user_needs_exact_content(user_text: str) -> bool:
    """Return True when the user asked for literal/technical spoken delivery.

    Matches explicit exact/verbatim/spell/dictate/read-aloud intents.
    A bare word such as "read" or "hash" is not enough.
    """
    normalized = _normalize_match_text(user_text)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _EXACT_CONTENT_PHRASES):
        return True
    return any(pattern.search(normalized) for pattern in _EXACT_CONTENT_PATTERNS)


def contains_required_notice(text: str) -> bool:
    """Return True when required safety/authorization language is present."""
    lowered = text.casefold()
    return any(marker in lowered for marker in _REQUIRED_NOTICE_MARKERS)


def fingerprint_spoken_text(text: str) -> str:
    """Return a bounded fingerprint of spoken text. Never stores audio."""
    return _clip(" ".join(text.strip().casefold().split()), MAX_SPEECH_FINGERPRINT_CHARS)


def build_speech_delivery_plan(
    *,
    delivery_mode: DeliveryMode,
    interaction_mode: InteractionMode | None = None,
    response_depth: ResponseDepth | None = None,
    acknowledgment_hint: AcknowledgmentHint | None = None,
    avoid_phrases: tuple[str, ...] = (),
    user_interrupted: bool = False,
    turn_taking: TurnTakingKind | None = None,
    user_text: str = "",
    canonical_text: str = "",
    delivery_state: SpeechDeliveryState | None = None,
    guidance: ConversationalGuidance | None = None,
) -> SpeechDeliveryPlan:
    """Build a bounded immutable spoken-delivery plan. Local and deterministic."""
    if guidance is not None:
        response_depth = response_depth or guidance.response_depth
        acknowledgment_hint = acknowledgment_hint or guidance.acknowledgment_hint
        if not avoid_phrases:
            avoid_phrases = guidance.avoid_phrases
        turn_taking = turn_taking or guidance.turn_taking
        if not user_text:
            user_text = guidance.original_user_text
    depth: ResponseDepth = response_depth or DEFAULT_RESPONSE_DEPTH  # type: ignore[assignment]
    ack: AcknowledgmentHint = acknowledgment_hint or "none"
    mode: InteractionMode = interaction_mode or _mode_from_delivery(delivery_mode)
    repetition = user_requests_repetition(user_text)
    exact = user_needs_exact_content(user_text)
    preserve_notice = contains_required_notice(canonical_text)
    concise = depth == "brief"
    list_mode: ListDeliveryMode = (
        "verbatim" if exact else "spoken_sequence"
    )
    if _looks_like_code_or_data(canonical_text) and not exact:
        list_mode = "screen_preferred"
    recovery: InterruptionRecoveryHint = "none"
    if user_interrupted:
        recovery = (
            "resume_from_correction"
            if turn_taking == "correction"
            else "skip_restart"
        )
    timing = _timing_hint(turn_taking, user_interrupted)
    avoid = list(avoid_phrases)
    if delivery_state is not None and delivery_state.last_acknowledgment:
        _append_unique(avoid, delivery_state.last_acknowledgment)
    if ack == "none":
        for phrase in ("got it", "okay", "sure", "absolutely"):
            _append_unique(avoid, phrase)
    if not repetition:
        _append_unique(avoid, "restating the user's request")
        _append_unique(avoid, "repeated introductions")
    return SpeechDeliveryPlan(
        delivery_mode=delivery_mode,
        response_depth=depth,
        preferred_chunk_chars=preferred_chunk_chars_for_depth(depth),
        sentence_pause_hint="short",
        paragraph_pause_hint="sentence",
        acknowledgment_hint=ack,
        list_delivery_mode=list_mode,
        interruption_recovery_hint=recovery,
        concise_spoken_delivery=concise,
        avoid_phrases=tuple(avoid),
        preserve_required_notice=preserve_notice,
        user_interrupted=user_interrupted,
        interaction_mode=mode,
        timing_hint=timing,
        repetition_requested=repetition,
        exact_content_required=exact,
    )


def normalize_for_speech(text: str, plan: SpeechDeliveryPlan) -> str:
    """Return a speech-friendly copy. Does not mutate canonical text."""
    spoken = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not spoken:
        return ""
    spoken = _replace_code_fences(spoken, plan)
    spoken = _replace_urls(spoken, plan)
    spoken = _HEADING_PATTERN.sub("", spoken)
    spoken = _convert_list_blocks(spoken, plan)
    if not plan.exact_content_required:
        protected, held = _protect_spans(spoken)
        protected = _EMPHASIS_PATTERN.sub(r"\2", protected)
        spoken = _restore_spans(protected, held)
    spoken = _replace_inline_code(spoken, plan)
    if not plan.exact_content_required:
        spoken = _soften_long_hashes(spoken)
        spoken = _flatten_long_parentheticals(spoken)
    spoken = _ARROW_PATTERN.sub(" to ", spoken)
    spoken = _MULTI_BLANK_PATTERN.sub("\n\n", spoken)
    spoken = spoken.replace("\n\n", " ")
    spoken = spoken.replace("\n", " ")
    spoken = _REPEATED_BANG_PATTERN.sub("!", spoken)
    spoken = _REPEATED_QUESTION_PATTERN.sub("?", spoken)
    spoken = _REPEATED_DOTS_PATTERN.sub("...", spoken)
    spoken = _strip_unwanted_acknowledgment(spoken, plan)
    spoken = re.sub(r"[ \t]+", " ", spoken).strip()
    return spoken


def chunk_spoken_text(text: str, plan: SpeechDeliveryPlan) -> tuple[SpokenChunk, ...]:
    """Split speech text on semantic boundaries. Deterministic; no model calls."""
    cleaned = text.strip()
    if not cleaned:
        return ()
    sentences = _split_sentences(cleaned)
    assembled: list[SpokenChunk] = []
    for sentence in sentences:
        pieces = _split_overlong(sentence, plan.preferred_chunk_chars)
        for index, piece in enumerate(pieces):
            if not piece.strip():
                continue
            pause: PauseHint
            if index < len(pieces) - 1:
                pause = "short"
            elif _looks_like_list_sentence(piece):
                pause = "list_item"
            else:
                pause = plan.sentence_pause_hint
            assembled.append(
                SpokenChunk(
                    text=piece.strip(),
                    pause_after=pause,
                    exact_content=plan.exact_content_required,
                )
            )
    merged = _merge_to_chunk_cap(assembled, plan)
    return tuple(chunk for chunk in merged if chunk.text)


def prepare_spoken_delivery(
    canonical_text: str,
    plan: SpeechDeliveryPlan,
    delivery_state: SpeechDeliveryState | None = None,
) -> SpokenDelivery:
    """Build a speech delivery copy and chunks from canonical assistant text."""
    if not SPEECH_DELIVERY_ENABLED:
        return _fallback_delivery(canonical_text, plan)
    spoken = normalize_for_speech(canonical_text, plan)
    if not spoken:
        return _fallback_delivery(canonical_text, plan)
    chunks = list(chunk_spoken_text(spoken, plan))
    if not chunks:
        return _fallback_delivery(canonical_text, plan)
    chunks = _apply_interruption_recovery(chunks, plan, delivery_state)
    chunks = _apply_repetition_filter(chunks, plan, delivery_state)
    if not chunks:
        return _fallback_delivery(canonical_text, plan)
    if delivery_state is not None:
        delivery_state.delivery_mode = plan.delivery_mode
        if plan.acknowledgment_hint != "none":
            delivery_state.record_acknowledgment(plan.acknowledgment_hint.replace("_", " "))
    return SpokenDelivery(
        canonical_text=canonical_text,
        spoken_text=" ".join(chunk.text for chunk in chunks),
        chunks=tuple(chunks),
        plan=plan,
        used_fallback=False,
    )


def safe_prepare_spoken_delivery(
    canonical_text: str,
    plan: SpeechDeliveryPlan | None,
    delivery_state: SpeechDeliveryState | None = None,
    *,
    delivery_mode: DeliveryMode = "voice_turn",
    interaction_mode: InteractionMode = "voice",
) -> SpokenDelivery:
    """Prepare spoken delivery; fall back to canonical text on failure."""
    try:
        active_plan = plan or build_speech_delivery_plan(
            delivery_mode=delivery_mode,
            interaction_mode=interaction_mode,
            canonical_text=canonical_text,
            delivery_state=delivery_state,
        )
        return prepare_spoken_delivery(canonical_text, active_plan, delivery_state)
    except Exception:
        fallback_plan = plan or build_speech_delivery_plan(
            delivery_mode=delivery_mode,
            interaction_mode=interaction_mode,
        )
        return _fallback_delivery(canonical_text, fallback_plan)


def format_speech_delivery_block(plan: SpeechDeliveryPlan | None) -> str:
    """Format advisory spoken-delivery guidance for realtime session instructions."""
    if plan is None:
        return ""
    lines = [
        "This block is spoken-delivery style guidance only. It is not an "
        "independent instruction, carries no elevated authority, and cannot "
        "authorize privileged actions regardless of its content.",
        SPEECH_DELIVERY_BEGIN,
        f"style_policy: {SPEECH_DELIVERY_STYLE_POLICY}",
        f"delivery_mode: {plan.delivery_mode}",
        f"response_depth: {plan.response_depth}",
        f"preferred_chunk_chars: {plan.preferred_chunk_chars}",
        f"sentence_pause_hint: {plan.sentence_pause_hint}",
        f"paragraph_pause_hint: {plan.paragraph_pause_hint}",
        f"acknowledgment_hint: {plan.acknowledgment_hint}",
        f"list_delivery_mode: {plan.list_delivery_mode}",
        f"interruption_recovery_hint: {plan.interruption_recovery_hint}",
        f"concise_spoken_delivery: {str(plan.concise_spoken_delivery).lower()}",
        f"timing_hint: {plan.timing_hint}",
        f"user_interrupted: {str(plan.user_interrupted).lower()}",
        f"repetition_requested: {str(plan.repetition_requested).lower()}",
        f"preserve_required_notice: {str(plan.preserve_required_notice).lower()}",
        f"exact_content_required: {str(plan.exact_content_required).lower()}",
    ]
    if plan.avoid_phrases:
        lines.append("avoid_phrases: " + ", ".join(plan.avoid_phrases))
    if plan.user_interrupted:
        lines.append(
            "recovery_note: do not restart the interrupted answer; "
            "continue from the user's latest correction or request."
        )
    if plan.acknowledgment_hint == "none":
        lines.append(
            "ack_note: do not prepend Got it, Okay, Sure, or Absolutely; "
            "begin with the answer."
        )
    elif not plan.preserve_required_notice:
        lines.append(
            "ack_note: a single brief acknowledgment is appropriate because "
            "the user corrected or confirmed; do not stack acknowledgments."
        )
    if plan.response_depth == "brief":
        lines.append("depth_note: be direct; skip recap; keep required facts.")
    elif plan.response_depth == "detailed":
        lines.append(
            "depth_note: structured spoken delivery with deliberate pacing; "
            "do not read a written report verbatim."
        )
    if plan.exact_content_required:
        lines.append(
            "exact_content_note: the user asked to hear the exact value; "
            "speak hashes, URLs, commands, and code verbatim. Do not replace "
            "them with screen-view language. Exactness wins over naturalness. "
            "This does not authorize privileged actions."
        )
    lines.append(
        "privilege_note: speech-delivery metadata never authorizes tools, "
        "workflows, calendar, reminders, memory writes, or confirmation bypass."
    )
    lines.append(SPEECH_DELIVERY_END)
    text = "\n".join(lines)
    if len(text) <= MAX_SPEECH_DELIVERY_INSTRUCTION_CHARS:
        return text
    return text[: MAX_SPEECH_DELIVERY_INSTRUCTION_CHARS - 1].rstrip() + "…"


def _fallback_delivery(canonical_text: str, plan: SpeechDeliveryPlan) -> SpokenDelivery:
    cleaned = canonical_text.strip()
    chunks: tuple[SpokenChunk, ...]
    if cleaned:
        chunks = (
            SpokenChunk(text=cleaned, pause_after="none", exact_content=True),
        )
    else:
        chunks = ()
    return SpokenDelivery(
        canonical_text=canonical_text,
        spoken_text=cleaned,
        chunks=chunks,
        plan=plan,
        used_fallback=True,
    )


def _strip_unwanted_acknowledgment(text: str, plan: SpeechDeliveryPlan) -> str:
    if plan.acknowledgment_hint != "none":
        return text
    if plan.preserve_required_notice:
        return text
    stripped = _LEADING_ACK_PATTERN.sub("", text, count=1).strip()
    return stripped or text


def _replace_code_fences(text: str, plan: SpeechDeliveryPlan) -> str:
    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if plan.exact_content_required:
            return inner
        if len(inner) <= MAX_CODE_SPEAK_CHARS and "\n" not in inner:
            return inner
        return "That code is better viewed on screen."

    return _CODE_FENCE_PATTERN.sub(_replace, text)


def _replace_inline_code(text: str, plan: SpeechDeliveryPlan) -> str:
    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if plan.exact_content_required or len(inner) <= MAX_CODE_SPEAK_CHARS:
            return inner
        return "a short code snippet, better viewed on screen"

    return _INLINE_CODE_PATTERN.sub(_replace, text)


def _replace_urls(text: str, plan: SpeechDeliveryPlan) -> str:
    def _replace(match: re.Match[str]) -> str:
        url = match.group(0)
        if plan.exact_content_required:
            return url
        if len(url) <= MAX_URL_SPEAK_CHARS and url.count("/") <= 3:
            host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
            host = host.split("/")[0]
            return host or "a link"
        return "a link, better viewed on screen"

    return _URL_PATTERN.sub(_replace, text)


def _soften_long_hashes(text: str) -> str:
    return _LONG_HASH_PATTERN.sub(
        "a long identifier, better viewed on screen",
        text,
    )


def _flatten_long_parentheticals(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if re.search(r"\d", inner):
            return f", {inner}, "
        if len(inner) <= MAX_PARENTHETICAL_SPEAK_CHARS:
            return f" ({inner}) "
        return " "

    return _LONG_PAREN_PATTERN.sub(_replace, text)


def _convert_list_blocks(text: str, plan: SpeechDeliveryPlan) -> str:
    lines = text.split("\n")
    output: list[str] = []
    pending: list[str] = []
    for line in lines:
        if _LIST_ITEM_PATTERN.match(line):
            pending.append(_LIST_ITEM_PATTERN.sub("", line).strip())
            continue
        if pending:
            output.append(_spoken_list(pending, plan))
            pending = []
        output.append(line)
    if pending:
        output.append(_spoken_list(pending, plan))
    return "\n".join(output)


def _spoken_list(items: list[str], plan: SpeechDeliveryPlan) -> str:
    cleaned = [item.rstrip(".;:") for item in items if item]
    if not cleaned:
        return ""
    if plan.list_delivery_mode == "verbatim" or plan.exact_content_required:
        return " ".join(cleaned)
    if plan.list_delivery_mode == "screen_preferred" or any(
        _looks_like_code_or_data(item) for item in cleaned
    ):
        return "That list is better viewed on screen."
    spoken_items: list[str] = []
    limit = min(len(cleaned), MAX_SPOKEN_LIST_ITEMS)
    for index, item in enumerate(cleaned[:limit], start=1):
        ordinal = _ORDINALS.get(index, "Next")
        spoken_items.append(f"{ordinal}, {_uncapitalize_item(item)}.")
    remainder = len(cleaned) - limit
    if remainder > 0:
        spoken_items.append("The remaining items are better viewed on screen.")
    return " ".join(spoken_items)


def _uncapitalize_item(text: str) -> str:
    if len(text) >= 2 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _split_sentences(text: str) -> list[str]:
    protected, held = _protect_spans(text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    restored = [_restore_spans(part, held).strip() for part in parts]
    return [part for part in restored if part]


def _split_overlong(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    clauses = re.split(r"(?<=[;:])\s+|(?<=—)\s+", text)
    if len(clauses) > 1:
        packed: list[str] = []
        current = ""
        for clause in clauses:
            candidate = f"{current} {clause}".strip() if current else clause
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                packed.append(current)
            if len(clause) <= limit:
                current = clause
            else:
                packed.extend(_hard_wrap(clause, limit))
                current = ""
        if current:
            packed.append(current)
        return [item for item in packed if item]
    return _hard_wrap(text, limit)


def _hard_wrap(text: str, limit: int) -> list[str]:
    remaining = text.strip()
    chunks: list[str] = []
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        window = remaining[:limit]
        space = window.rfind(" ")
        if space < max(MIN_SPEECH_CHUNK_CHARS, limit // 4):
            space = limit
        piece = remaining[:space].rstrip()
        if piece:
            chunks.append(piece)
        remaining = remaining[space:].lstrip()
    return chunks


def _merge_to_chunk_cap(
    chunks: list[SpokenChunk],
    plan: SpeechDeliveryPlan,
) -> list[SpokenChunk]:
    if plan.response_depth == "brief":
        chunks = _pack_brief_chunks(chunks, plan.preferred_chunk_chars)
    if len(chunks) <= MAX_SPEECH_CHUNKS:
        return chunks
    head = list(chunks[: MAX_SPEECH_CHUNKS - 1])
    rest = chunks[MAX_SPEECH_CHUNKS - 1 :]
    remainder = " ".join(item.text for item in rest).strip()
    head.append(
        SpokenChunk(
            text=remainder,
            pause_after="none",
            exact_content=any(item.exact_content for item in rest),
        )
    )
    return head


def _pack_brief_chunks(chunks: list[SpokenChunk], limit: int) -> list[SpokenChunk]:
    """Fewer chunks for brief delivery: pack until the brief size bound."""
    if len(chunks) <= 1:
        return chunks
    packed: list[SpokenChunk] = []
    current = chunks[0]
    for chunk in chunks[1:]:
        combined = f"{current.text} {chunk.text}".strip()
        if len(combined) <= limit:
            current = SpokenChunk(
                text=combined,
                pause_after=chunk.pause_after,
                exact_content=current.exact_content or chunk.exact_content,
            )
        else:
            packed.append(current)
            current = chunk
    packed.append(current)
    return packed


def _apply_interruption_recovery(
    chunks: list[SpokenChunk],
    plan: SpeechDeliveryPlan,
    delivery_state: SpeechDeliveryState | None,
) -> list[SpokenChunk]:
    if delivery_state is None:
        return chunks
    interrupted = None
    if plan.user_interrupted or plan.interruption_recovery_hint != "none":
        interrupted = delivery_state.consume_interrupted_fingerprint()
    if not interrupted:
        return chunks
    kept = [
        chunk
        for chunk in chunks
        if not _overlaps_fingerprint(chunk.text, interrupted)
    ]
    return kept or chunks


def _apply_repetition_filter(
    chunks: list[SpokenChunk],
    plan: SpeechDeliveryPlan,
    delivery_state: SpeechDeliveryState | None,
) -> list[SpokenChunk]:
    if plan.repetition_requested or delivery_state is None:
        return chunks
    kept: list[SpokenChunk] = []
    for index, chunk in enumerate(chunks):
        if contains_required_notice(chunk.text):
            kept.append(chunk)
            continue
        fingerprint = fingerprint_spoken_text(chunk.text)
        if fingerprint in delivery_state.recently_spoken_fingerprints:
            continue
        if (
            index == 0
            and delivery_state.last_spoken_opening
            and fingerprint.startswith(delivery_state.last_spoken_opening)
        ):
            continue
        kept.append(chunk)
    return kept or chunks


def _overlaps_fingerprint(text: str, fingerprint: str) -> bool:
    current = fingerprint_spoken_text(text)
    if not current or not fingerprint:
        return False
    return current.startswith(fingerprint) or fingerprint.startswith(current)


def _protect_spans(text: str) -> tuple[str, list[str]]:
    held: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"⟦{len(held) - 1}⟧"

    protected = _URL_PATTERN.sub(_hold, text)
    protected = _EMAIL_PATTERN.sub(_hold, protected)
    protected = _PROTECTED_UUID.sub(_hold, protected)
    protected = _PROTECTED_IPV4.sub(_hold, protected)
    protected = _PROTECTED_LATIN.sub(_hold, protected)
    protected = _PROTECTED_ABBREVIATION.sub(_hold, protected)
    protected = _PROTECTED_VERSION.sub(_hold, protected)
    protected = _PROTECTED_DECIMAL.sub(_hold, protected)
    return protected, held


def _restore_spans(text: str, held: list[str]) -> str:
    restored = text
    for index, value in enumerate(held):
        restored = restored.replace(f"⟦{index}⟧", value)
    return restored


def _looks_like_code_or_data(text: str) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    if any(marker in lowered for marker in _CODEISH_MARKERS):
        return True
    if text.count("`") >= 2:
        return True
    return False


def _looks_like_list_sentence(text: str) -> bool:
    return bool(re.match(r"^(?:First|Second|Third|Fourth|Fifth|Sixth|Next),", text))


def _timing_hint(
    turn_taking: TurnTakingKind | None,
    user_interrupted: bool,
) -> TimingHint:
    if user_interrupted:
        return "interrupted_turn"
    if turn_taking == "correction":
        return "correction"
    if turn_taking in {"continuation", "incomplete_thought"}:
        return "likely_continuation"
    return "ready_to_speak"


def _mode_from_delivery(delivery_mode: DeliveryMode) -> InteractionMode:
    if delivery_mode == "realtime":
        return "realtime"
    if delivery_mode == "multimodal":
        return "multimodal"
    return "voice"


def _normalize_match_text(text: str) -> str:
    cleaned = text.strip().casefold().replace("’", "'")
    cleaned = re.sub(r"[?!.,]+$", "", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)
