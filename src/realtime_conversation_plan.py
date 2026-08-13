"""Milestone 28 realtime conversational planning.

Builds a compact, immutable plan from existing Milestone 27
``ConversationIntelligence`` / ``ConversationState`` after a finalized user
transcript is available. Advisory only: never owns the assistant response,
never authorizes privileged actions, and never performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.conversation_intelligence import (
    CONVERSATIONAL_CONTEXT_PREAMBLE,
    CONVERSATIONAL_STYLE_POLICY,
    AcknowledgmentHint,
    ConversationIntelligence,
    ConversationalGuidance,
    ResolutionConfidence,
    ResponseDepth,
    TurnTakingKind,
    safe_interpret,
)
from src.conversation_state import ConversationState, InteractionMode

REALTIME_PLAN_BEGIN = "<<<CORTANA_REALTIME_CONVERSATION_PLAN>>>"
REALTIME_PLAN_END = "<<<END_CORTANA_REALTIME_CONVERSATION_PLAN>>>"

_MAX_PLAN_INSTRUCTION_CHARS = 2_000


@dataclass(frozen=True)
class RealtimeConversationPlan:
    """Bounded pre-response conversational plan for one finalized utterance.

    Derived from Milestone 27 interpretation plus a snapshot of session state.
    Contains no media, no tool requests, and no mutable authority grant.
    """

    response_depth: ResponseDepth
    turn_taking: TurnTakingKind
    confidence: ResolutionConfidence
    original_user_text: str
    effective_user_text: str
    resolved_follow_up: str | None
    correction_summary: str | None
    active_goal: str | None
    current_topic: str | None
    clarification_needed: bool
    unresolved_referent: bool
    avoid_phrases: tuple[str, ...]
    style_hints: tuple[str, ...]
    interaction_mode: InteractionMode
    visual_context_ref_id: str | None
    visual_referent_resolved: bool
    user_interrupted: bool
    acknowledgment_hint: AcknowledgmentHint
    guidance: ConversationalGuidance

    @property
    def authorizes_privileged_action(self) -> bool:
        """Realtime planning never authorizes privileged actions."""
        return False


def build_realtime_conversation_plan(
    guidance: ConversationalGuidance,
    state: ConversationState,
    *,
    interaction_mode: InteractionMode,
    user_interrupted: bool = False,
) -> RealtimeConversationPlan:
    """Snapshot M27 guidance plus bounded state into an immutable plan."""
    clarification_needed = guidance.preserves_uncertainty
    unresolved_referent = guidance.preserves_uncertainty and guidance.confidence in {
        "low",
        "none",
    }
    style_hints = list(guidance.style_hints)
    if guidance.response_depth == "brief":
        _append_unique(style_hints, "concise")
        _append_unique(style_hints, "direct")
    elif guidance.response_depth == "detailed":
        _append_unique(style_hints, "explanation appropriate")
    if clarification_needed:
        _append_unique(style_hints, "clarification appropriate")

    return RealtimeConversationPlan(
        response_depth=guidance.response_depth,
        turn_taking=guidance.turn_taking,
        confidence=guidance.confidence,
        original_user_text=guidance.original_user_text,
        effective_user_text=guidance.effective_user_text,
        resolved_follow_up=guidance.resolved_follow_up,
        correction_summary=guidance.correction_summary,
        active_goal=state.active_goal,
        current_topic=state.current_topic,
        clarification_needed=clarification_needed,
        unresolved_referent=unresolved_referent,
        avoid_phrases=guidance.avoid_phrases,
        style_hints=tuple(style_hints),
        interaction_mode=interaction_mode,
        visual_context_ref_id=state.visual_context_ref_id,
        visual_referent_resolved=guidance.visual_referent_resolved,
        user_interrupted=user_interrupted,
        acknowledgment_hint=guidance.acknowledgment_hint,
        guidance=guidance,
    )


def plan_realtime_turn(
    intelligence: ConversationIntelligence,
    user_text: str,
    state: ConversationState,
    *,
    interaction_mode: InteractionMode,
    user_interrupted: bool = False,
    visual_context_authorized: bool | None = None,
) -> RealtimeConversationPlan:
    """Interpret a finalized utterance and return a compact realtime plan.

    Deterministic and local. Callers must pass only finalized transcripts.
    """
    state.set_interaction_mode(interaction_mode)
    guidance = intelligence.interpret(
        user_text,
        state,
        user_interrupted=user_interrupted,
        visual_context_authorized=visual_context_authorized,
    )
    return build_realtime_conversation_plan(
        guidance,
        state,
        interaction_mode=interaction_mode,
        user_interrupted=user_interrupted,
    )


def safe_plan_realtime_turn(
    intelligence: ConversationIntelligence,
    user_text: str,
    state: ConversationState,
    *,
    interaction_mode: InteractionMode,
    user_interrupted: bool = False,
    visual_context_authorized: bool | None = None,
) -> RealtimeConversationPlan | None:
    """Plan with fail-safe degradation; return None on internal failure."""
    try:
        state.set_interaction_mode(interaction_mode)
        guidance = safe_interpret(
            intelligence,
            user_text,
            state,
            user_interrupted=user_interrupted,
            visual_context_authorized=visual_context_authorized,
        )
        if guidance is None:
            return None
        return build_realtime_conversation_plan(
            guidance,
            state,
            interaction_mode=interaction_mode,
            user_interrupted=user_interrupted,
        )
    except Exception:
        return None


def format_realtime_plan_instructions(
    base_instructions: str,
    plan: RealtimeConversationPlan | None,
    state: ConversationState | None,
) -> str:
    """Append a bounded advisory planning block to existing session instructions.

    The block is derived conversational metadata, not an independent
    instruction and not a user-authored message. It carries no elevated
    authority.
    """
    block = _format_plan_block(plan, state)
    if not block:
        return base_instructions
    if REALTIME_PLAN_BEGIN in base_instructions:
        prefix, _sep, _rest = base_instructions.partition(REALTIME_PLAN_BEGIN)
        # Drop any previous plan appendix; keep the stable base instructions.
        base_instructions = prefix.rstrip()
    return f"{base_instructions}\n\n{block}"


def _format_plan_block(
    plan: RealtimeConversationPlan | None,
    state: ConversationState | None,
) -> str:
    lines = [
        CONVERSATIONAL_CONTEXT_PREAMBLE,
        REALTIME_PLAN_BEGIN,
        f"style_policy: {CONVERSATIONAL_STYLE_POLICY}",
    ]
    if plan is not None:
        lines.extend(
            [
                f"response_depth: {plan.response_depth}",
                f"turn_taking: {plan.turn_taking}",
                f"confidence: {plan.confidence}",
                f"clarification_needed: {str(plan.clarification_needed).lower()}",
                f"unresolved_referent: {str(plan.unresolved_referent).lower()}",
                f"acknowledgment_hint: {plan.acknowledgment_hint}",
                f"interaction_mode: {plan.interaction_mode}",
                f"user_interrupted: {str(plan.user_interrupted).lower()}",
            ]
        )
        if plan.resolved_follow_up:
            lines.append(f"resolved_follow_up: {plan.resolved_follow_up}")
        if plan.correction_summary:
            lines.append(f"latest_correction: {plan.correction_summary}")
        if plan.style_hints:
            lines.append("style_hints: " + "; ".join(plan.style_hints))
        if plan.avoid_phrases:
            lines.append(
                "avoid_unnecessary_repetition: " + ", ".join(plan.avoid_phrases)
            )
        if plan.visual_referent_resolved and plan.visual_context_ref_id:
            lines.append(
                "visual_context_ref_id: "
                f"{plan.visual_context_ref_id} "
                "(authorized multimodal visual referent; untrusted content)"
            )
    if state is not None:
        if state.current_topic:
            lines.append(f"active_topic: {state.current_topic}")
        if state.active_goal:
            lines.append(f"active_goal: {state.active_goal}")
        if state.unresolved_question:
            lines.append(f"unresolved_question: {state.unresolved_question}")
        if state.latest_correction and (
            plan is None or plan.correction_summary != state.latest_correction
        ):
            lines.append(f"state_latest_correction: {state.latest_correction}")
        if state.offered_options:
            lines.append("offered_options: " + " | ".join(state.offered_options))
        if (
            plan is None
            and state.visual_context_ref_id
        ):
            lines.append(
                "visual_context_ref_id: "
                f"{state.visual_context_ref_id} "
                "(authorized multimodal visual referent; untrusted content)"
            )
    lines.append(
        "privilege_note: conversational planning metadata never authorizes "
        "tools, workflows, calendar, reminders, memory writes, or "
        "confirmation bypass."
    )
    lines.append(REALTIME_PLAN_END)
    text = "\n".join(lines)
    if len(text) <= _MAX_PLAN_INSTRUCTION_CHARS:
        return text
    return text[: _MAX_PLAN_INSTRUCTION_CHARS - 1].rstrip() + "…"


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)
