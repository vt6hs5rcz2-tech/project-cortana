"""Pre-M30 Deep Audit #4: ConversationState sibling-invariant discovery.

Discovery only. These tests encode desired invariants. Failures document
confirmed inconsistencies; they are not production regressions.
"""

from __future__ import annotations

from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState, ConversationalReferent
from src.realtime_multimodal import RealtimeMultimodalSession
from src.active_memory import ActiveMemoryContext
from src.conversation import ConversationHistory
from src.settings import Settings
from typing import Any, cast


def _intel() -> ConversationIntelligence:
    return ConversationIntelligence()


def _pending_question_state() -> ConversationState:
    state = ConversationState()
    state.set_active_goal("schedule the review")
    state.set_unresolved_question("Which day should we meet?")
    state.set_offered_options(("Monday", "Tuesday"))
    state.replace_option_referents(("Monday", "Tuesday"))
    return state


def _assert_invariant_a(state: ConversationState) -> None:
    """Non-empty unresolved_question implies waiting_for_user=True."""
    if state.unresolved_question:
        assert state.waiting_for_user is True, (
            "Invariant A broken: unresolved_question is set while "
            f"waiting_for_user is False (question={state.unresolved_question!r})"
        )


def test_deep_f1_incomplete_thought_clears_waiting_leaves_question() -> None:
    """OUTSIDE-F1 path: incomplete thought must not leave a stale question."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("I was checking the logs and", state)
    _assert_invariant_a(state)


def test_deep_f1_i_meant_correction_clears_waiting_leaves_question() -> None:
    """OUTSIDE-F1 path: 'I meant X' must clear both waiting and the question."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("I meant Tuesday", state)
    _assert_invariant_a(state)
    assert state.active_goal is not None
    assert "tuesday" in state.active_goal.casefold()


def test_deep_f1_date_word_clears_waiting_leaves_question() -> None:
    """OUTSIDE-F1 path: date-word clarification must clear the pending question."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("Tuesday", state)
    _assert_invariant_a(state)


def test_deep_f1_prefixed_date_correction_clears_waiting_leaves_question() -> None:
    """Sibling of OUTSIDE-F1: 'actually, Tuesday' uses the same date-word helper."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("actually, Tuesday", state)
    _assert_invariant_a(state)


def test_deep_yes_no_paths_preserve_invariant_a() -> None:
    """Control: yes/no follow-ups clear both fields together."""
    intel = _intel()
    for phrase in ("yes", "no"):
        state = _pending_question_state()
        intel.interpret(phrase, state)
        assert state.unresolved_question is None
        assert state.waiting_for_user is False
        _assert_invariant_a(state)


def test_deep_topic_change_clears_dependent_continuity() -> None:
    """Explicit topic-change path must drop options, referents, question, correction."""
    intel = _intel()
    state = _pending_question_state()
    state.set_latest_correction("old correction")
    intel.interpret("anyway, how do I rotate SSH keys safely?", state)
    assert state.offered_options == ()
    assert state.recent_referents == []
    assert state.unresolved_question is None
    assert state.waiting_for_user is False
    assert state.latest_correction is None
    _assert_invariant_a(state)


def test_deep_complete_request_leaves_stale_options_and_referents() -> None:
    """Sibling C: a new complete request must not keep prior offered options."""
    intel = _intel()
    state = _pending_question_state()
    state.set_latest_correction("old correction")
    intel.interpret("Please rotate the SSH keys on the jump host", state)
    assert state.offered_options == ()
    assert state.recent_referents == []
    assert state.latest_correction is None
    _assert_invariant_a(state)


def test_deep_go_back_leaves_pending_question_and_options() -> None:
    """Sibling D: 'go back' should not leave a pending clarifier against the old thread."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("go back", state)
    assert state.unresolved_question is None
    assert state.offered_options == ()
    _assert_invariant_a(state)


def test_deep_forget_that_clears_question_and_options() -> None:
    """Control: forget-that conversational discard clears thread-local continuity."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("forget that", state)
    assert state.unresolved_question is None
    assert state.waiting_for_user is False
    assert state.offered_options == ()
    _assert_invariant_a(state)


def test_deep_not_what_was_asked_waiting_without_question_is_allowed() -> None:
    """Invariant A allows waiting_for_user without a stored question."""
    intel = _intel()
    state = ConversationState()
    state.set_active_goal("summarize the scan")
    intel.interpret("that's not what I asked", state)
    assert state.waiting_for_user is True
    _assert_invariant_a(state)


def test_deep_direct_waiting_assignment_bypasses_setter() -> None:
    """Sibling I: ConversationIntelligence writes waiting_for_user directly."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("I meant Tuesday", state)
    # The setter would have cleared the question when waiting became False.
    assert state.unresolved_question is None or state.waiting_for_user is True


def test_deep_clone_heals_inconsistent_live_state() -> None:
    """Sibling H: clone() runs __post_init__ and can hide a live invariant break."""
    intel = _intel()
    state = _pending_question_state()
    intel.interpret("I meant Tuesday", state)
    live_inconsistent = bool(state.unresolved_question) and not state.waiting_for_user
    cloned = state.clone()
    if live_inconsistent:
        assert cloned.waiting_for_user is True
        assert cloned.unresolved_question == state.unresolved_question


def test_deep_restore_copies_inconsistent_pair_if_source_bypasses_clone() -> None:
    """restore() must not copy an inconsistent question/waiting pair verbatim.

    A snapshot with a pending question must restore waiting_for_user=True so
    Invariant A holds and the genuine question is not dropped.
    """
    source = ConversationState()
    source.unresolved_question = "Which day?"
    source.waiting_for_user = False
    target = ConversationState()
    target.restore(source)
    assert target.unresolved_question == "Which day?"
    assert target.waiting_for_user is True
    _assert_invariant_a(target)


def test_deep_reset_clears_all_m27_m29_fields() -> None:
    """Sibling G: reset() must clear every conversational continuity field."""
    state = ConversationState()
    state.set_topic("vpn")
    state.set_active_goal("explain vpn")
    state.set_unresolved_question("Which one?")
    state.set_latest_correction("Tuesday")
    state.set_offered_options(("A", "B"))
    state.add_referent(ConversationalReferent(label="vpn", description="rule"))
    state.set_visual_context_ref("vis_1")
    state.set_interaction_mode("realtime")
    state.record_acknowledgment("okay")
    state.record_restatement_fingerprint("explain vpn")
    state.reset()
    assert state.is_empty
    assert state.recent_interaction_mode is None
    assert state.character_budget_used == 0


def test_deep_clone_restore_round_trip_includes_all_fields() -> None:
    """Sibling F: clone/restore must not omit newer fields."""
    state = ConversationState()
    state.set_topic("vpn")
    state.set_active_goal("explain vpn")
    state.set_unresolved_question("Which one?")
    state.set_latest_correction("Tuesday")
    state.set_offered_options(("A", "B"))
    state.add_referent(ConversationalReferent(label="vpn", description="rule"))
    state.set_visual_context_ref("vis_abc")
    state.set_interaction_mode("multimodal")
    state.record_acknowledgment("okay")
    copied = state.clone()
    restored = ConversationState()
    restored.restore(copied)
    assert restored.snapshot() == state.snapshot()
    assert restored.recent_interaction_mode == "multimodal"
    assert restored.visual_context_ref_id == "vis_abc"


def test_deep_post_init_forces_waiting_when_question_set() -> None:
    """Sibling H control: constructor repairs Invariant A."""
    state = ConversationState(
        unresolved_question="Which day?",
        waiting_for_user=False,
    )
    assert state.waiting_for_user is True
    _assert_invariant_a(state)


def test_deep_setter_clears_waiting_with_question() -> None:
    """Control: the setter pair is internally consistent."""
    state = ConversationState()
    state.set_unresolved_question("Which day?")
    assert state.waiting_for_user is True
    state.set_unresolved_question(None)
    assert state.waiting_for_user is False
    _assert_invariant_a(state)


def test_deep_visual_ref_cleared_on_multimodal_cleanup() -> None:
    """Sibling E: session cleanup must drop visual_context_ref_id."""
    state = ConversationState()
    state.set_visual_context_ref("vis_live")
    session = RealtimeMultimodalSession(
        settings=Settings(
            openai_api_key="k",
            openai_model="m",
            transcription_model="t",
            tts_model="t",
            tts_voice="c",
            realtime_model="r",
            realtime_voice="c",
        ),
        client=cast(Any, object()),
        conversation_history=ConversationHistory(),
        active_memory_context=ActiveMemoryContext(),
        conversation_state=state,
        print_fn=lambda _line: None,
        transcript_wait_seconds=0.25,
    )
    session._cleanup()
    assert state.visual_context_ref_id is None
