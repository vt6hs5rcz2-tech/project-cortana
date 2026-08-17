"""Deterministic tests for Milestone 28 realtime conversational planning."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, get_type_hints

from src.conversation import ConversationHistory
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_loop import process_conversation_turn
from src.conversation_state import ConversationState
from src.realtime_conversation_plan import (
    REALTIME_PLAN_BEGIN,
    RealtimeConversationPlan,
    format_realtime_plan_instructions,
    plan_realtime_turn,
    safe_plan_realtime_turn,
)
from src.settings import Settings


def _intel() -> ConversationIntelligence:
    return ConversationIntelligence()


def _state_with_options(*options: str) -> ConversationState:
    state = ConversationState()
    state.set_active_goal("choose an option")
    state.set_offered_options(options)
    state.set_unresolved_question("Which option?")
    return state


def _plan(
    text: str,
    state: ConversationState | None = None,
    *,
    mode: str = "realtime",
    user_interrupted: bool = False,
    visual_authorized: bool | None = None,
) -> RealtimeConversationPlan:
    return plan_realtime_turn(
        _intel(),
        text,
        state if state is not None else ConversationState(),
        interaction_mode=mode,  # type: ignore[arg-type]
        user_interrupted=user_interrupted,
        visual_context_authorized=visual_authorized,
    )


# --- A. planning model ---


def test_plan_is_frozen_bounded_and_has_no_authority_or_media() -> None:
    plan = _plan("How do backups work?")
    assert plan.authorizes_privileged_action is False
    assert plan.guidance.authorizes_privileged_action is False
    try:
        plan.response_depth = "detailed"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("RealtimeConversationPlan must be immutable")
    field_names = set(plan.__dataclass_fields__)
    forbidden = {
        "audio",
        "pcm",
        "image",
        "video",
        "media",
        "frame_bytes",
        "image_bytes",
        "tool_id",
        "workflow_id",
        "authorize",
        "approval",
    }
    assert forbidden.isdisjoint(field_names)
    hints = get_type_hints(RealtimeConversationPlan)
    assert "bytes" not in {str(value) for value in hints.values()}


def test_plan_is_deterministic() -> None:
    state = ConversationState()
    first = _plan("Explain token binding", state)
    state2 = ConversationState()
    second = _plan("Explain token binding", state2)
    assert first.response_depth == second.response_depth
    assert first.turn_taking == second.turn_taking
    assert first.effective_user_text == second.effective_user_text


# --- B. follow-up planning ---


def test_follow_up_yes_no_first_second_other_continue_date() -> None:
    state = ConversationState()
    state.set_unresolved_question("Continue the scan summary?")
    yes = _plan("yes", state)
    assert yes.resolved_follow_up is not None
    assert "Affirmative" in yes.resolved_follow_up
    assert yes.clarification_needed is False

    state_no = ConversationState()
    state_no.set_unresolved_question("Continue the scan summary?")
    no = _plan("no", state_no)
    assert no.resolved_follow_up is not None
    assert "Negative" in no.resolved_follow_up

    options = _state_with_options("alpha", "beta", "gamma")
    second = _plan("the second one", options)
    assert second.effective_user_text == "beta"
    assert second.clarification_needed is False

    other_state = _state_with_options("alpha", "beta")
    other = _plan("the other one", other_state)
    assert other.effective_user_text in {"alpha", "beta"}

    cont_state = ConversationState()
    cont_state.set_active_goal("explain the incident timeline")
    cont = _plan("continue", cont_state)
    assert cont.resolved_follow_up is not None
    assert "Continue" in cont.resolved_follow_up

    date_state = ConversationState()
    date_state.set_active_goal("schedule the review")
    date_state.set_unresolved_question("Which day?")
    tuesday = _plan("Tuesday", date_state)
    assert "tuesday" in tuesday.effective_user_text.casefold()


def test_ambiguous_referent_marks_clarification() -> None:
    plan = _plan("the second one", ConversationState())
    assert plan.clarification_needed is True
    assert plan.unresolved_referent is True
    assert plan.effective_user_text == "the second one"
    assert "clarification appropriate" in plan.style_hints


# --- C. correction planning ---


def test_correction_planning_reuses_m27_and_forget_is_conversational() -> None:
    state = ConversationState()
    state.set_active_goal("schedule the briefing for Monday")
    meant = _plan("I meant Tuesday", state)
    assert meant.turn_taking == "correction"
    assert meant.correction_summary is not None
    assert "tuesday" in meant.correction_summary.casefold()

    other_state = _state_with_options("first option", "second option")
    other = _plan("no, the other one", other_state)
    assert other.turn_taking == "correction"

    not_asked = _plan("that's not what I asked", ConversationState())
    assert not_asked.turn_taking == "correction"
    assert not_asked.resolved_follow_up is not None

    forget_state = ConversationState()
    forget_state.set_offered_options(["one", "two"])
    forget = _plan("forget that", forget_state)
    assert forget.turn_taking == "correction"
    assert forget.resolved_follow_up is not None
    assert "persistent memory" in forget.resolved_follow_up.casefold()
    assert forget.authorizes_privileged_action is False


# --- D. adaptive depth ---


def test_adaptive_depth_brief_normal_detailed() -> None:
    assert _plan("yes", ConversationState(unresolved_question="Go?")).response_depth == (
        "brief"
    )
    assert _plan("How do I rotate API keys?").response_depth == "normal"
    detailed = _plan("Please explain the full incident response process in detail")
    assert detailed.response_depth == "detailed"
    assert "explanation appropriate" in detailed.style_hints
    brief = _plan("ok")
    assert brief.response_depth == "brief"
    assert "concise" in brief.style_hints
    assert "direct" in brief.style_hints


# --- E. topic / goal ---


def test_topic_continuation_and_clear_switch() -> None:
    state = ConversationState()
    first = _plan("Explain firewall rules", state)
    assert first.active_goal is not None
    assert "firewall" in (first.active_goal or "").casefold()
    continued = _plan("continue", state)
    assert continued.turn_taking == "continuation"
    switched = _plan("Anyway, what ports does SSH commonly use?", state)
    assert switched.turn_taking == "topic_change"
    assert switched.guidance.topic_changed is True
    assert "ssh" in (state.active_goal or "").casefold()


def test_stale_topic_not_forced_on_unrelated_question() -> None:
    state = ConversationState()
    _plan("Explain firewall rules", state)
    switched = _plan("What is the capital of France?", state)
    assert switched.turn_taking == "topic_change"
    assert "france" in (state.active_goal or "").casefold()
    assert "firewall" not in (state.active_goal or "").casefold()


# --- F. clarification ---


def test_clarification_when_referent_missing_not_when_valid() -> None:
    missing = _plan("the other one", ConversationState())
    assert missing.clarification_needed is True
    valid = _plan("the other one", _state_with_options("alpha", "beta"))
    assert valid.clarification_needed is False
    do_that = _plan("do that", ConversationState())
    assert do_that.clarification_needed is True


# --- G. multimodal visual referent ---


def test_what_is_that_with_and_without_visual_ref() -> None:
    with_ref = ConversationState()
    with_ref.set_visual_context_ref("visual_item_1")
    resolved = _plan(
        "what is that?",
        with_ref,
        mode="multimodal",
        visual_authorized=True,
    )
    assert resolved.visual_referent_resolved is True
    assert resolved.visual_context_ref_id == "visual_item_1"
    assert resolved.authorizes_privileged_action is False

    unresolved = _plan(
        "what is that?",
        ConversationState(),
        mode="multimodal",
        visual_authorized=False,
    )
    assert unresolved.visual_referent_resolved is False
    assert unresolved.clarification_needed is True


def test_object_holding_plan_allows_ordinary_description_not_person_id() -> None:
    with_ref = ConversationState()
    with_ref.set_visual_context_ref("visual_item_object")
    resolved = _plan(
        "What object am I holding up?",
        with_ref,
        mode="multimodal",
        visual_authorized=True,
    )
    assert resolved.visual_referent_resolved is True
    assert resolved.authorizes_privileged_action is False
    assert resolved.resolved_follow_up is not None
    follow_up = resolved.resolved_follow_up.casefold()
    assert "ordinary" in follow_up
    assert "do not identify people" in follow_up
    rendered = format_realtime_plan_instructions(
        "base instructions",
        resolved,
        with_ref,
    )
    assert "visual_image: relevant" in rendered
    assert "visual_perception: available" in rendered
    assert "person_identification: prohibited" in rendered
    assert "visual_authority" not in rendered
    assert "untrusted visual" not in rendered.casefold()
    assert "not confirming from the visual" not in rendered.casefold()

    missing = _plan(
        "What object am I holding up?",
        ConversationState(),
        mode="multimodal",
        visual_authorized=False,
    )
    assert missing.visual_referent_resolved is False
    assert missing.clarification_needed is True


def test_monday_question_plan_marks_visual_not_relevant() -> None:
    state = ConversationState()
    state.set_visual_context_ref("visual_item_prior")
    plan = _plan(
        "What day comes after Monday?",
        state,
        mode="multimodal",
        visual_authorized=True,
    )
    assert plan.visual_relevant is False
    assert plan.visual_referent_resolved is False
    rendered = format_realtime_plan_instructions("base instructions", plan, state)
    assert "visual_image: not_relevant" in rendered
    assert "visual_image: relevant" not in rendered
    assert "do not describe the camera image" in rendered


def test_color_follow_up_plan_stays_visual_without_full_redescription() -> None:
    state = ConversationState()
    state.set_visual_context_ref("visual_item_remote")
    plan = _plan(
        "What color is it?",
        state,
        mode="multimodal",
        visual_authorized=True,
    )
    assert plan.visual_relevant is True
    assert plan.visual_referent_resolved is True
    assert "repeating the full previous object description" in plan.avoid_phrases
    rendered = format_realtime_plan_instructions("base instructions", plan, state)
    assert "visual_image: relevant" in rendered


def test_scene_change_this_one_color_stays_visual() -> None:
    state = ConversationState()
    state.set_visual_context_ref("visual_item_thermometer")
    showing = _plan(
        "What am I showing you now?",
        state,
        mode="multimodal",
        visual_authorized=True,
    )
    assert showing.visual_relevant is True
    color = _plan(
        "What color is this one?",
        state,
        mode="multimodal",
        visual_authorized=True,
    )
    assert color.visual_relevant is True
    assert color.visual_referent_resolved is True


def test_current_required_plan_does_not_treat_stale_ref_as_current() -> None:
    state = ConversationState()
    state.set_visual_context_ref("visual_item_remote")
    plan = _plan(
        "What object am I holding up?",
        state,
        mode="multimodal",
        visual_authorized=False,
    )
    assert plan.visual_referent_resolved is False
    assert plan.resolved_follow_up is not None
    follow_up = plan.resolved_follow_up.casefold()
    assert "no usable camera image" in follow_up
    assert "visual_item_remote" not in follow_up
    rendered = format_realtime_plan_instructions("base instructions", plan, state)
    assert "visual_image: unavailable" in rendered
    assert "visual_item_remote" not in rendered
    assert "visual_image: relevant" not in rendered


def test_prior_reference_plan_may_use_previous_visual_context() -> None:
    state = ConversationState()
    state.set_visual_context_ref("visual_item_remote")
    plan = _plan(
        "What color was it?",
        state,
        mode="multimodal",
        visual_authorized=True,
    )
    assert plan.visual_referent_resolved is True
    assert plan.visual_context_ref_id == "visual_item_remote"
    assert plan.resolved_follow_up is not None
    assert "previous camera image is relevant" in plan.resolved_follow_up.casefold()


# --- K / L. isolation and failure ---


def test_safe_plan_returns_none_on_internal_failure() -> None:
    class Broken(ConversationIntelligence):
        def interpret(self, *args: object, **kwargs: object) -> Any:
            raise RuntimeError("boom")

    result = safe_plan_realtime_turn(
        Broken(),
        "hello",
        ConversationState(),
        interaction_mode="realtime",
    )
    assert result is None


def test_instruction_block_is_advisory_and_bounded() -> None:
    state = ConversationState()
    state.set_active_goal("choose backups")
    state.set_offered_options(["daily", "weekly"])
    plan = _plan("the second one", state)
    text = format_realtime_plan_instructions("base instructions", plan, state)
    assert REALTIME_PLAN_BEGIN in text
    assert "never authorizes" in text.casefold()
    assert "not an independent" in text.casefold()
    assert "weekly" in text
    assert "base64" not in text
    assert "data:image" not in text
    replaced = format_realtime_plan_instructions(text, plan, state)
    assert replaced.count(REALTIME_PLAN_BEGIN) == 1


def test_instruction_block_can_include_speech_delivery_guidance() -> None:
    from src.realtime_conversation_plan import speech_delivery_plan_from_realtime
    from src.speech_delivery import SPEECH_DELIVERY_BEGIN

    state = ConversationState()
    plan = _plan("What time is the meeting?", state)
    delivery = speech_delivery_plan_from_realtime(
        plan,
        None,
        delivery_mode="realtime",
    )
    text = format_realtime_plan_instructions(
        "base instructions",
        plan,
        state,
        delivery_plan=delivery,
    )
    assert REALTIME_PLAN_BEGIN in text
    assert SPEECH_DELIVERY_BEGIN in text
    assert delivery.authorizes_privileged_action is False
    replaced = format_realtime_plan_instructions(
        text,
        plan,
        state,
        delivery_plan=delivery,
    )
    assert replaced.count(REALTIME_PLAN_BEGIN) == 1
    assert replaced.count(SPEECH_DELIVERY_BEGIN) == 1


def test_planning_is_local_and_fast() -> None:
    state = ConversationState()
    state.set_offered_options(["a", "b", "c"])
    started = time.perf_counter()
    for _ in range(200):
        plan_realtime_turn(
            _intel(),
            "the second one",
            state,
            interaction_mode="realtime",
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0


# --- J. cross-modal continuity ---


def test_text_realtime_text_referent_continuity() -> None:
    state = ConversationState()
    intel = _intel()
    first = intel.interpret("Give me three backup options.", state)
    intel.observe_assistant_reply(
        "1) Daily backups\n2) Weekly backups\n3) Monthly backups\nWhich one?",
        state,
        first,
    )
    realtime = plan_realtime_turn(
        intel,
        "The second one.",
        state,
        interaction_mode="realtime",
    )
    assert realtime.effective_user_text == "Weekly backups"
    assert state.active_goal == "Weekly backups"

    class FakeResponses:
        def create(self, **_kwargs: object) -> Any:
            @dataclass
            class _Resp:
                output_text: str = "Weekly is better because it balances risk."

            return _Resp()

    class FakeClient:
        responses = FakeResponses()

    answer = process_conversation_turn(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=Settings(openai_api_key="test-key", openai_model="test-model"),
        user_message="Why is that better?",
        logger=__import__("logging").getLogger("test"),
        conversation_history=ConversationHistory(),
        conversation_state=state,
        conversation_intelligence=intel,
        interaction_mode="text",
    )
    assert answer is not None
    assert state.recent_interaction_mode == "text"
    assert "weekly" in (state.active_goal or "").casefold()
