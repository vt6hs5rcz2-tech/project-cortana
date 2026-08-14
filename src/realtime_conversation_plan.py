"""Milestone 28 realtime conversational planning.

Builds a compact, immutable plan from existing Milestone 27
``ConversationIntelligence`` / ``ConversationState`` after a finalized user
transcript is available. Advisory only: never owns the assistant response,
never authorizes privileged actions, and never performs I/O.
"""

from __future__ import annotations

import logging
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
from src.speech_delivery import (
    SPEECH_DELIVERY_BEGIN,
    DeliveryMode,
    SpeechDeliveryPlan,
    SpeechDeliveryState,
    build_speech_delivery_plan,
    format_speech_delivery_block,
    _join_capped_advisory_lines,
)

logger = logging.getLogger("ProjectCortana")

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


def speech_delivery_plan_from_realtime(
    plan: RealtimeConversationPlan | None,
    delivery_state: SpeechDeliveryState | None,
    *,
    delivery_mode: DeliveryMode,
) -> SpeechDeliveryPlan:
    """Derive M29 spoken-delivery policy from an M28 realtime plan."""
    if plan is None:
        interaction: InteractionMode = (
            "realtime" if delivery_mode == "realtime" else "multimodal"
        )
        return build_speech_delivery_plan(
            delivery_mode=delivery_mode,
            interaction_mode=interaction,
            delivery_state=delivery_state,
        )
    return build_speech_delivery_plan(
        delivery_mode=delivery_mode,
        interaction_mode=plan.interaction_mode,
        response_depth=plan.response_depth,
        acknowledgment_hint=plan.acknowledgment_hint,
        avoid_phrases=plan.avoid_phrases,
        user_interrupted=plan.user_interrupted,
        turn_taking=plan.turn_taking,
        user_text=plan.original_user_text,
        delivery_state=delivery_state,
        guidance=plan.guidance,
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
    except Exception as error:
        logger.error(
            "Realtime plan failed error_type=%s",
            type(error).__name__,
        )
        return None


def format_realtime_plan_instructions(
    base_instructions: str,
    plan: RealtimeConversationPlan | None,
    state: ConversationState | None,
    delivery_plan: SpeechDeliveryPlan | None = None,
) -> str:
    """Append bounded advisory planning and speech-delivery blocks.

    The blocks are derived conversational metadata, not independent
    instructions and not user-authored messages. They carry no elevated
    authority.
    """
    stripped = _strip_advisory_appendices(base_instructions)
    block = _format_plan_block(plan, state)
    delivery_block = format_speech_delivery_block(delivery_plan)
    parts = [stripped]
    if block:
        parts.append(block)
    if delivery_block:
        parts.append(delivery_block)
    if len(parts) == 1:
        return stripped
    return "\n\n".join(parts)


def _strip_advisory_appendices(base_instructions: str) -> str:
    """Drop previous M28/M29 advisory appendices; keep stable base instructions."""
    cut_at: int | None = None
    for marker in (REALTIME_PLAN_BEGIN, SPEECH_DELIVERY_BEGIN):
        index = base_instructions.find(marker)
        if index >= 0 and (cut_at is None or index < cut_at):
            cut_at = index
    if cut_at is None:
        return base_instructions
    return base_instructions[:cut_at].rstrip()


def _format_plan_block(
    plan: RealtimeConversationPlan | None,
    state: ConversationState | None,
) -> str:
    prefix = [
        CONVERSATIONAL_CONTEXT_PREAMBLE,
        REALTIME_PLAN_BEGIN,
        f"style_policy: {CONVERSATIONAL_STYLE_POLICY}",
    ]
    body: list[str] = []
    if plan is not None:
        body.extend(
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
            body.append(_clip_plan_field("resolved_follow_up", plan.resolved_follow_up))
        if plan.correction_summary:
            body.append(_clip_plan_field("latest_correction", plan.correction_summary))
        if plan.style_hints:
            body.append(
                _clip_plan_field("style_hints", "; ".join(plan.style_hints))
            )
        if plan.avoid_phrases:
            body.append(
                _clip_plan_field(
                    "avoid_unnecessary_repetition",
                    ", ".join(plan.avoid_phrases),
                )
            )
        if plan.visual_referent_resolved and plan.visual_context_ref_id:
            body.append(
                _clip_plan_field(
                    "visual_context_ref_id",
                    f"{plan.visual_context_ref_id} "
                    "(authorized multimodal visual referent; untrusted content)",
                )
            )
    if state is not None:
        if state.current_topic:
            body.append(_clip_plan_field("active_topic", state.current_topic))
        if state.active_goal:
            body.append(_clip_plan_field("active_goal", state.active_goal))
        if state.unresolved_question:
            body.append(
                _clip_plan_field("unresolved_question", state.unresolved_question)
            )
        if state.latest_correction and (
            plan is None or plan.correction_summary != state.latest_correction
        ):
            body.append(
                _clip_plan_field("state_latest_correction", state.latest_correction)
            )
        if state.offered_options:
            body.append(
                _clip_plan_field(
                    "offered_options",
                    " | ".join(state.offered_options),
                    limit=400,
                )
            )
        if plan is None and state.visual_context_ref_id:
            body.append(
                _clip_plan_field(
                    "visual_context_ref_id",
                    f"{state.visual_context_ref_id} "
                    "(authorized multimodal visual referent; untrusted content)",
                )
            )
    suffix = [
        "privilege_note: conversational planning metadata never authorizes "
        "tools, workflows, calendar, reminders, memory writes, or "
        "confirmation bypass.",
        REALTIME_PLAN_END,
    ]
    return _join_capped_advisory_lines(
        prefix,
        body,
        suffix,
        _MAX_PLAN_INSTRUCTION_CHARS,
    )


def _clip_plan_field(label: str, value: str, *, limit: int = 240) -> str:
    text = f"{label}: {value}"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)
