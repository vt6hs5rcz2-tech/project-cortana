"""Bounded session conversation state for Milestone 27 continuity.

Tracks only recent conversational information needed for follow-ups,
corrections, and response shaping. This is not permanent memory and never
stores raw camera or audio content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.config import (
    CONVERSATIONAL_RECENT_TURN_WINDOW,
    MAX_CONVERSATIONAL_GOAL_CHARS,
    MAX_CONVERSATIONAL_QUESTION_CHARS,
    MAX_CONVERSATIONAL_REFERENT_CHARS,
    MAX_CONVERSATIONAL_REFERENTS,
    MAX_CONVERSATIONAL_STATE_CHARS,
    MAX_CONVERSATIONAL_TOPIC_CHARS,
    MAX_RECENT_ASSISTANT_ACK_TRACK,
)

InteractionMode = Literal["text", "voice", "realtime", "multimodal"]


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class ConversationalReferent:
    """One bounded recent referent usable for follow-up resolution."""

    label: str
    description: str
    ordinal: int | None = None

    def clipped(
        self,
        *,
        label_limit: int = MAX_CONVERSATIONAL_REFERENT_CHARS,
        description_limit: int = MAX_CONVERSATIONAL_REFERENT_CHARS,
    ) -> ConversationalReferent:
        """Return a length-bounded copy of this referent."""
        return ConversationalReferent(
            label=_clip(self.label, label_limit),
            description=_clip(self.description, description_limit),
            ordinal=self.ordinal,
        )


@dataclass
class ConversationState:
    """Session-scoped conversational continuity state.

    Intentionally small and deterministic. Never authorizes tools, memory
    writes, calendar/reminder writes, workflows, or other privileged actions.
    """

    current_topic: str | None = None
    active_goal: str | None = None
    unresolved_question: str | None = None
    latest_correction: str | None = None
    waiting_for_user: bool = False
    recent_interaction_mode: InteractionMode | None = None
    visual_context_ref_id: str | None = None
    offered_options: tuple[str, ...] = ()
    recent_referents: list[ConversationalReferent] = field(default_factory=list)
    recent_ack_phrases: list[str] = field(default_factory=list)
    recent_restatement_fingerprints: list[str] = field(default_factory=list)
    turn_window: int = CONVERSATIONAL_RECENT_TURN_WINDOW
    _character_budget_used: int = 0

    def _enforce_question_waiting_invariant(self) -> None:
        """Keep unresolved_question and waiting_for_user coherent (Invariant A).

        A non-empty unresolved_question implies waiting_for_user=True.
        waiting_for_user may still be True without a stored question.
        A genuine pending question is never dropped to satisfy the pair.
        """
        if self.unresolved_question is not None and not self.unresolved_question.strip():
            self.unresolved_question = None
        if self.unresolved_question is not None:
            self.waiting_for_user = True

    def __post_init__(self) -> None:
        """Keep constructor state coherent with setter semantics.

        Invariant A: a non-empty unresolved_question implies
        waiting_for_user=True. waiting_for_user may still be True without a
        stored question (for example an incomplete thought).
        """
        self._enforce_question_waiting_invariant()

    def reset(self) -> None:
        """Clear all session conversational state."""
        self.current_topic = None
        self.active_goal = None
        self.unresolved_question = None
        self.latest_correction = None
        self.waiting_for_user = False
        self.recent_interaction_mode = None
        self.visual_context_ref_id = None
        self.offered_options = ()
        self.recent_referents.clear()
        self.recent_ack_phrases.clear()
        self.recent_restatement_fingerprints.clear()
        self._character_budget_used = 0
        self._enforce_question_waiting_invariant()

    def set_interaction_mode(self, mode: InteractionMode) -> None:
        """Record the most recent interaction mode for continuity hints."""
        self.recent_interaction_mode = mode

    def set_topic(self, topic: str | None) -> None:
        """Update the active topic, clipping to the configured bound."""
        if topic is None or not topic.strip():
            self.current_topic = None
        else:
            self.current_topic = _clip(topic, MAX_CONVERSATIONAL_TOPIC_CHARS)
        self._after_mutation()

    def set_active_goal(self, goal: str | None) -> None:
        """Update the active conversational goal/request."""
        if goal is None or not goal.strip():
            self.active_goal = None
        else:
            self.active_goal = _clip(goal, MAX_CONVERSATIONAL_GOAL_CHARS)
        self._after_mutation()

    def set_unresolved_question(self, question: str | None) -> None:
        """Track an unresolved assistant clarifying question."""
        if question is None or not question.strip():
            self.unresolved_question = None
            self.waiting_for_user = False
        else:
            self.unresolved_question = _clip(
                question,
                MAX_CONVERSATIONAL_QUESTION_CHARS,
            )
            self.waiting_for_user = True
        self._enforce_question_waiting_invariant()
        self._after_mutation()

    def set_waiting_for_user(self, waiting: bool) -> None:
        """Set waiting_for_user without violating Invariant A.

        Clearing waiting while a question is stored is refused: the question
        remains pending and waiting stays True. Callers that consumed the
        question must use set_unresolved_question(None).
        """
        if waiting:
            self.waiting_for_user = True
        elif self.unresolved_question is None:
            self.waiting_for_user = False
        else:
            self.waiting_for_user = True
        self._enforce_question_waiting_invariant()

    def set_latest_correction(self, correction: str | None) -> None:
        """Record the latest conversational correction/clarification."""
        if correction is None or not correction.strip():
            self.latest_correction = None
        else:
            self.latest_correction = _clip(
                correction,
                MAX_CONVERSATIONAL_GOAL_CHARS,
            )
        self._after_mutation()

    def set_offered_options(self, options: tuple[str, ...] | list[str]) -> None:
        """Replace offered options used by ordinal follow-ups."""
        clipped: list[str] = []
        for option in options:
            text = _clip(option, MAX_CONVERSATIONAL_REFERENT_CHARS)
            if text:
                clipped.append(text)
            if len(clipped) >= MAX_CONVERSATIONAL_REFERENTS:
                break
        self.offered_options = tuple(clipped)
        self._after_mutation()

    def add_referent(self, referent: ConversationalReferent) -> None:
        """Append one referent and trim to the configured bound."""
        clipped = referent.clipped()
        if not clipped.label and not clipped.description:
            return
        self.recent_referents.append(clipped)
        overflow = len(self.recent_referents) - MAX_CONVERSATIONAL_REFERENTS
        if overflow > 0:
            del self.recent_referents[0:overflow]
        self._after_mutation()

    def clear_referents(self) -> None:
        """Drop all recent referents."""
        self.recent_referents.clear()
        self._after_mutation()

    def clear_option_thread(self) -> None:
        """Drop offered options and option-derived ordinal referents.

        Referents without an ordinal are preserved because they are not
        bound to the current numbered option list.
        """
        self.set_offered_options(())
        self.replace_option_referents(())

    def abandon_pending_branch(self) -> None:
        """Clear the pending question and option thread; keep topic/goal."""
        self.set_unresolved_question(None)
        self.clear_option_thread()

    def replace_option_referents(self, options: tuple[str, ...] | list[str]) -> None:
        """Replace option-derived referents; keep referents without ordinals."""
        kept = [item for item in self.recent_referents if item.ordinal is None]
        self.recent_referents = kept
        for index, option in enumerate(options, start=1):
            clipped = ConversationalReferent(
                label=f"option {index}",
                description=option,
                ordinal=index,
            ).clipped()
            if clipped.label or clipped.description:
                self.recent_referents.append(clipped)
        overflow = len(self.recent_referents) - MAX_CONVERSATIONAL_REFERENTS
        if overflow > 0:
            del self.recent_referents[0:overflow]
        self._after_mutation()

    def set_visual_context_ref(self, ref_id: str | None) -> None:
        """Set or clear the optional M26 visual-context reference identifier.

        Stores only an opaque identifier string — never image/audio bytes.
        """
        if ref_id is None or not ref_id.strip():
            self.visual_context_ref_id = None
        else:
            self.visual_context_ref_id = _clip(ref_id.strip(), 128)
        self._after_mutation()

    def clear_visual_context_ref(self) -> None:
        """Clear any current visual-context reference identifier."""
        self.visual_context_ref_id = None
        self._after_mutation()

    def record_acknowledgment(self, phrase: str) -> None:
        """Track a recently used lightweight acknowledgment phrase."""
        cleaned = _clip(phrase.casefold(), 40)
        if not cleaned:
            return
        self.recent_ack_phrases.append(cleaned)
        overflow = len(self.recent_ack_phrases) - MAX_RECENT_ASSISTANT_ACK_TRACK
        if overflow > 0:
            del self.recent_ack_phrases[0:overflow]
        self._after_mutation()

    def record_restatement_fingerprint(self, fingerprint: str) -> None:
        """Track a recent request-restatement fingerprint for repetition control."""
        cleaned = _clip(fingerprint.casefold(), 120)
        if not cleaned:
            return
        self.recent_restatement_fingerprints.append(cleaned)
        overflow = (
            len(self.recent_restatement_fingerprints) - self.turn_window
        )
        if overflow > 0:
            del self.recent_restatement_fingerprints[0:overflow]
        self._after_mutation()

    def clone(self) -> ConversationState:
        """Return a detached copy of this session state."""
        copied = ConversationState(
            current_topic=self.current_topic,
            active_goal=self.active_goal,
            unresolved_question=self.unresolved_question,
            latest_correction=self.latest_correction,
            waiting_for_user=self.waiting_for_user,
            recent_interaction_mode=self.recent_interaction_mode,
            visual_context_ref_id=self.visual_context_ref_id,
            offered_options=self.offered_options,
            recent_referents=list(self.recent_referents),
            recent_ack_phrases=list(self.recent_ack_phrases),
            recent_restatement_fingerprints=list(
                self.recent_restatement_fingerprints
            ),
            turn_window=self.turn_window,
            _character_budget_used=self._character_budget_used,
        )
        return copied

    def restore(self, other: ConversationState) -> None:
        """Replace this state's fields with a previously cloned snapshot."""
        self.current_topic = other.current_topic
        self.active_goal = other.active_goal
        self.unresolved_question = other.unresolved_question
        self.latest_correction = other.latest_correction
        self.waiting_for_user = other.waiting_for_user
        self.recent_interaction_mode = other.recent_interaction_mode
        self.visual_context_ref_id = other.visual_context_ref_id
        self.offered_options = other.offered_options
        self.recent_referents = list(other.recent_referents)
        self.recent_ack_phrases = list(other.recent_ack_phrases)
        self.recent_restatement_fingerprints = list(
            other.recent_restatement_fingerprints
        )
        self.turn_window = other.turn_window
        self._character_budget_used = other._character_budget_used
        self._enforce_question_waiting_invariant()

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic serializable snapshot of current state."""
        return {
            "current_topic": self.current_topic,
            "active_goal": self.active_goal,
            "unresolved_question": self.unresolved_question,
            "latest_correction": self.latest_correction,
            "waiting_for_user": self.waiting_for_user,
            "recent_interaction_mode": self.recent_interaction_mode,
            "visual_context_ref_id": self.visual_context_ref_id,
            "offered_options": list(self.offered_options),
            "recent_referents": [
                {
                    "label": referent.label,
                    "description": referent.description,
                    "ordinal": referent.ordinal,
                }
                for referent in self.recent_referents
            ],
            "recent_ack_phrases": list(self.recent_ack_phrases),
            "recent_restatement_fingerprints": list(
                self.recent_restatement_fingerprints
            ),
            "character_budget_used": self.character_budget_used,
        }

    @property
    def character_budget_used(self) -> int:
        """Return approximate character usage of tracked conversational fields."""
        return self._character_budget_used

    @property
    def is_empty(self) -> bool:
        """Return True when no conversational continuity fields are set."""
        return (
            self.current_topic is None
            and self.active_goal is None
            and self.unresolved_question is None
            and self.latest_correction is None
            and not self.waiting_for_user
            and self.visual_context_ref_id is None
            and not self.offered_options
            and not self.recent_referents
            and not self.recent_ack_phrases
            and not self.recent_restatement_fingerprints
        )

    def _after_mutation(self) -> None:
        self._recompute_budget()
        self._trim_to_character_budget()

    def _recompute_budget(self) -> None:
        total = 0
        for value in (
            self.current_topic,
            self.active_goal,
            self.unresolved_question,
            self.latest_correction,
            self.visual_context_ref_id,
        ):
            if value is not None:
                total += len(value)
        total += sum(len(option) for option in self.offered_options)
        for referent in self.recent_referents:
            total += len(referent.label) + len(referent.description)
        total += sum(len(phrase) for phrase in self.recent_ack_phrases)
        total += sum(
            len(fingerprint)
            for fingerprint in self.recent_restatement_fingerprints
        )
        self._character_budget_used = total

    def _trim_to_character_budget(self) -> None:
        """Drop low-value historical state until within the aggregate cap.

        Preference order: restatement fingerprints, acknowledgments, oldest
        referents, older options, then correction/question/visual scalars.
        Active topic and goal are not truncated into nonsense.
        """
        self._recompute_budget()
        guard = 0
        while (
            self._character_budget_used > MAX_CONVERSATIONAL_STATE_CHARS
            and guard < 64
        ):
            guard += 1
            if self.recent_restatement_fingerprints:
                self.recent_restatement_fingerprints.pop(0)
            elif self.recent_ack_phrases:
                self.recent_ack_phrases.pop(0)
            elif self.recent_referents:
                self.recent_referents.pop(0)
            elif self.offered_options:
                self.offered_options = self.offered_options[1:]
            elif self.latest_correction is not None:
                self.latest_correction = None
            elif self.unresolved_question is not None:
                self.unresolved_question = None
                self.waiting_for_user = False
            elif self.visual_context_ref_id is not None:
                self.visual_context_ref_id = None
            else:
                break
            self._recompute_budget()
