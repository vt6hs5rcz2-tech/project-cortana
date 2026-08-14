"""Tests for Milestone 27 bounded conversation state."""

from __future__ import annotations

from src.config import (
    MAX_CONVERSATIONAL_REFERENTS,
    MAX_CONVERSATIONAL_STATE_CHARS,
    MAX_CONVERSATIONAL_TOPIC_CHARS,
)
from src.conversation_state import ConversationState, ConversationalReferent


def test_state_starts_empty() -> None:
    state = ConversationState()
    assert state.is_empty
    assert state.waiting_for_user is False
    assert state.visual_context_ref_id is None


def test_topic_and_goal_are_bounded() -> None:
    state = ConversationState()
    state.set_topic("x" * (MAX_CONVERSATIONAL_TOPIC_CHARS + 50))
    assert state.current_topic is not None
    assert len(state.current_topic) <= MAX_CONVERSATIONAL_TOPIC_CHARS
    state.set_active_goal("Investigate suspicious login")
    assert state.active_goal == "Investigate suspicious login"


def test_unresolved_question_constructor_implies_waiting() -> None:
    state = ConversationState(unresolved_question="Which option do you prefer?")
    assert state.waiting_for_user is True
    waiting_only = ConversationState(waiting_for_user=True)
    assert waiting_only.unresolved_question is None
    assert waiting_only.waiting_for_user is True


def test_unresolved_question_sets_waiting_for_user() -> None:
    state = ConversationState()
    state.set_unresolved_question("Which option do you prefer?")
    assert state.waiting_for_user is True
    assert state.unresolved_question is not None
    state.set_unresolved_question(None)
    assert state.waiting_for_user is False


def test_referents_are_bounded() -> None:
    state = ConversationState()
    for index in range(MAX_CONVERSATIONAL_REFERENTS + 5):
        state.add_referent(
            ConversationalReferent(
                label=f"option {index}",
                description=f"desc {index}",
                ordinal=index,
            )
        )
    assert len(state.recent_referents) == MAX_CONVERSATIONAL_REFERENTS
    assert state.recent_referents[0].label == "option 5"


def test_character_budget_trims_oldest_referents() -> None:
    state = ConversationState()
    bulky = "word " * 200
    for index in range(20):
        state.add_referent(
            ConversationalReferent(label=f"r{index}", description=bulky)
        )
    assert state.character_budget_used <= MAX_CONVERSATIONAL_STATE_CHARS
    assert len(state.recent_referents) <= MAX_CONVERSATIONAL_REFERENTS


def test_visual_context_ref_stores_id_only() -> None:
    state = ConversationState()
    state.set_visual_context_ref("item_abc123")
    assert state.visual_context_ref_id == "item_abc123"
    state.clear_visual_context_ref()
    assert state.visual_context_ref_id is None


def test_reset_clears_all_fields() -> None:
    state = ConversationState()
    state.set_topic("auth")
    state.set_active_goal("review logs")
    state.set_unresolved_question("Continue?")
    state.set_latest_correction("Tuesday")
    state.set_offered_options(["A", "B"])
    state.add_referent(ConversationalReferent(label="A", description="A", ordinal=1))
    state.set_visual_context_ref("vis1")
    state.record_acknowledgment("got it")
    state.record_restatement_fingerprint("review logs")
    state.reset()
    assert state.is_empty
    assert state.snapshot()["waiting_for_user"] is False


def test_snapshot_is_deterministic() -> None:
    state = ConversationState()
    state.set_topic("topic")
    state.set_offered_options(["one", "two"])
    first = state.snapshot()
    second = state.snapshot()
    assert first == second
    assert first["offered_options"] == ["one", "two"]


def test_restore_repairs_inconsistent_question_waiting_pair() -> None:
    """restore() must not copy unresolved_question with waiting=False."""
    source = ConversationState()
    source.unresolved_question = "Which day?"
    source.waiting_for_user = False
    target = ConversationState()
    target.restore(source)
    assert target.unresolved_question == "Which day?"
    assert target.waiting_for_user is True


def test_restore_valid_waiting_question() -> None:
    source = ConversationState()
    source.set_unresolved_question("Which day works?")
    source.set_active_goal("schedule the review")
    target = ConversationState()
    target.set_active_goal("unrelated")
    target.restore(source)
    assert target.unresolved_question == "Which day works?"
    assert target.waiting_for_user is True
    assert target.active_goal == "schedule the review"


def test_clone_restore_round_trip_preserves_waiting_question() -> None:
    state = ConversationState()
    state.set_topic("schedule")
    state.set_active_goal("schedule the review")
    state.set_unresolved_question("Which day works?")
    state.set_offered_options(("Monday", "Tuesday"))
    cloned = state.clone()
    restored = ConversationState()
    restored.restore(cloned)
    assert restored.snapshot() == state.snapshot()
    assert restored.waiting_for_user is True
    assert restored.unresolved_question == "Which day works?"


def test_set_waiting_for_user_refuses_impossible_pair() -> None:
    state = ConversationState()
    state.set_unresolved_question("Which option?")
    state.set_waiting_for_user(False)
    assert state.unresolved_question == "Which option?"
    assert state.waiting_for_user is True
    state.set_unresolved_question(None)
    state.set_waiting_for_user(False)
    assert state.waiting_for_user is False


def test_clear_option_thread_keeps_non_ordinal_referents() -> None:
    state = ConversationState()
    state.set_offered_options(("Alpha", "Beta"))
    state.replace_option_referents(("Alpha", "Beta"))
    state.add_referent(ConversationalReferent(label="vpn", description="rule"))
    state.clear_option_thread()
    assert state.offered_options == ()
    assert [item.label for item in state.recent_referents] == ["vpn"]


def test_state_is_not_permanent_memory() -> None:
    """ConversationState has no persist/save/load surface for memories."""
    methods = {name for name in dir(ConversationState) if not name.startswith("_")}
    forbidden = {"save", "load", "persist", "write", "commit", "to_disk"}
    assert methods.isdisjoint(forbidden)
