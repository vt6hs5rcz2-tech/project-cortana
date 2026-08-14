"""Deterministic tests for Milestone 29 natural speech delivery."""

from __future__ import annotations

from typing import get_type_hints

from src.config import MAX_SPEECH_CHUNKS, MAX_TTS_CHARS
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.speech_delivery import (
    SPEECH_DELIVERY_BEGIN,
    SpeechDeliveryPlan,
    SpeechDeliveryState,
    build_speech_delivery_plan,
    chunk_spoken_text,
    contains_required_notice,
    fingerprint_spoken_text,
    format_speech_delivery_block,
    normalize_for_speech,
    preferred_chunk_chars_for_depth,
    prepare_spoken_delivery,
    safe_prepare_spoken_delivery,
    user_needs_exact_content,
    user_requests_repetition,
)


def _plan(**kwargs: object) -> SpeechDeliveryPlan:
    defaults: dict[str, object] = {
        "delivery_mode": "voice_turn",
        "interaction_mode": "voice",
    }
    defaults.update(kwargs)
    return build_speech_delivery_plan(**defaults)  # type: ignore[arg-type]


def _spoken(canonical: str, **kwargs: object) -> str:
    plan = _plan(canonical_text=canonical, **kwargs)
    return normalize_for_speech(canonical, plan)


# --- A. Speech normalization ---


def test_normalize_markdown_heading() -> None:
    spoken = _spoken("# Backup steps\nDo this next.")
    assert "#" not in spoken
    assert "Backup steps" in spoken
    assert "Do this next." in spoken


def test_normalize_bullet_and_numbered_lists() -> None:
    numbered = _spoken(
        "1. Back up files.\n2. Update software.\n3. Restart."
    )
    assert "First," in numbered
    assert "Second," in numbered
    assert "Third," in numbered
    assert "1." not in numbered
    bullets = _spoken("- Back up files\n- Update software")
    assert "First," in bullets
    assert "Second," in bullets
    assert "-" not in bullets.split("First")[0]


def test_normalize_emphasis_blank_lines_and_punctuation() -> None:
    spoken = _spoken(
        "This is **important** and *urgent*.\n\n\nReally??  Wait!!"
    )
    assert "**" not in spoken
    assert "*" not in spoken
    assert "important" in spoken
    assert "urgent" in spoken
    assert "??" not in spoken
    assert "!!" not in spoken
    assert "Really?" in spoken
    assert "Wait!" in spoken


def test_normalize_preserves_numbers_and_dates() -> None:
    spoken = _spoken(
        "Meet on 2024-08-13 at 3:30pm. Bring 12 keys and $40."
    )
    assert "2024-08-13" in spoken
    assert "3:30pm" in spoken
    assert "12" in spoken
    assert "$40" in spoken


# --- B. Chunking ---


def test_chunking_sentence_and_list_boundaries() -> None:
    plan = _plan(response_depth="normal")
    spoken = normalize_for_speech(
        "Alpha is ready. Beta follows next. Gamma finishes.",
        plan,
    )
    chunks = chunk_spoken_text(spoken, plan)
    texts = [chunk.text for chunk in chunks]
    assert all(texts)
    assert chunk_spoken_text(spoken, plan) == chunks
    assert any("Alpha" in item for item in texts)
    listed = normalize_for_speech(
        "1. First item.\n2. Second item.\n3. Third item.",
        plan,
    )
    list_chunks = chunk_spoken_text(listed, plan)
    assert all(chunk.text for chunk in list_chunks)
    assert any(chunk.pause_after == "list_item" for chunk in list_chunks)


def test_chunking_long_sentence_fallback_and_bounds() -> None:
    plan = _plan(response_depth="detailed")
    long = (
        "This is a very long sentence without a period that keeps going "
        * 8
    ).strip()
    chunks = chunk_spoken_text(long, plan)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert max(len(chunk.text) for chunk in chunks) <= plan.preferred_chunk_chars + 20
    assert chunk_spoken_text(long, plan) == chunks


def test_chunking_does_not_split_abbreviations_or_decimals() -> None:
    plan = _plan()
    text = "Dr. Smith arrived at 3.14 seconds past 4 p.m. today."
    chunks = chunk_spoken_text(text, plan)
    joined = " ".join(chunk.text for chunk in chunks)
    assert "Dr. Smith" in joined
    assert "3.14" in joined


# --- C. Response depth ---


def test_response_depth_maps_to_delivery_behavior() -> None:
    brief = _plan(response_depth="brief", user_text="briefly, what time?")
    normal = _plan(response_depth="normal", user_text="What time is the meeting?")
    detailed = _plan(
        response_depth="detailed",
        user_text="Explain the backup process in detail",
    )
    assert brief.concise_spoken_delivery is True
    assert brief.preferred_chunk_chars > normal.preferred_chunk_chars
    assert detailed.preferred_chunk_chars < normal.preferred_chunk_chars
    assert preferred_chunk_chars_for_depth("brief") == brief.preferred_chunk_chars
    text = (
        "Alpha is one idea. Beta is another idea. Gamma is a third idea. "
        "Delta wraps it up."
    )
    brief_chunks = prepare_spoken_delivery(text, brief).chunks
    detailed_chunks = prepare_spoken_delivery(text, detailed).chunks
    assert len(brief_chunks) <= len(detailed_chunks)


# --- D. Acknowledgments ---


def test_acknowledgments_avoid_repeated_filler() -> None:
    plan = _plan(acknowledgment_hint="none", user_text="What time is the meeting?")
    spoken = normalize_for_speech("Sure, the meeting is at 3.", plan)
    assert not spoken.lower().startswith("sure")
    assert "3" in spoken
    correction = _plan(
        acknowledgment_hint="okay",
        turn_taking="correction",
        user_text="No, Tuesday.",
    )
    corrected = normalize_for_speech("Okay, Tuesday it is.", correction)
    assert corrected.lower().startswith("okay")
    notice = _plan(
        acknowledgment_hint="none",
        canonical_text="Approval required before that change.",
    )
    preserved = normalize_for_speech(
        "Sure, approval required before that change.",
        notice,
    )
    assert "approval required" in preserved.casefold()


# --- E. Interruption recovery ---


def test_interruption_recovery_skips_restarted_material() -> None:
    state = SpeechDeliveryState()
    state.mark_interrupted(fingerprint_spoken_text("There are three options. The first"))
    plan = _plan(
        user_interrupted=True,
        turn_taking="correction",
        user_text="No, the second one.",
    )
    assert plan.interruption_recovery_hint == "resume_from_correction"
    delivery = prepare_spoken_delivery(
        "There are three options. The first is daily. The second is weekly.",
        plan,
        state,
    )
    spoken = delivery.spoken_text.casefold()
    assert "weekly" in spoken
    assert state.interrupted_response_fingerprint is None
    state.load_pending(["stale chunk one", "stale chunk two"])
    state.abort_pending()
    assert state.pop_pending_chunk() is None
    assert state.pending_chunks == []


# --- F. Repetition ---


def test_repetition_skips_already_spoken_unless_requested() -> None:
    state = SpeechDeliveryState()
    first_plan = _plan(user_text="What is MFA?")
    first = prepare_spoken_delivery(
        "MFA is multi-factor authentication.",
        first_plan,
        state,
    )
    for chunk in first.chunks:
        state.record_completed_chunk(chunk.text)
    second = prepare_spoken_delivery(
        "MFA is multi-factor authentication.",
        first_plan,
        state,
    )
    # Filter would drop all matching chunks, so fail-safe keeps content rather
    # than going silent. Explicit repeat still allows a full rerun.
    assert second.spoken_text
    repeat_plan = _plan(user_text="Repeat that")
    assert repeat_plan.repetition_requested is True
    repeated = prepare_spoken_delivery(
        "MFA is multi-factor authentication.",
        repeat_plan,
        state,
    )
    assert "multi-factor authentication" in repeated.spoken_text.casefold()
    assert user_requests_repetition("say that again") is True
    assert user_requests_repetition("don't repeat that") is False


# --- G / M. Exact-content safety ---


def test_exact_content_preserves_ids_and_does_not_rewrite_code() -> None:
    spoken = _spoken("Ticket ABC12 and build 47 ship on 2024-08-13.")
    assert "ABC12" in spoken
    assert "47" in spoken
    assert "2024-08-13" in spoken
    code = _spoken("Use `sha256sum file.bin` then compare.")
    assert "sha256sum file.bin" in code
    long_code = _spoken(
        "```\ndef totally_invented_rewrite():\n    return 1\n```"
    )
    assert "def totally_invented_rewrite" not in long_code
    assert "better viewed on screen" in long_code.casefold()
    exact = _spoken(
        "The hash is abcdef0123456789abcdef0123456789.",
        user_text="give me the exact hash",
    )
    assert "abcdef0123456789abcdef0123456789" in exact


_LONG_HASH = "abcdef0123456789abcdef0123456789ff00aa11"
_LONG_URL = (
    "https://example.com/path/resource?utm_source=tracker&utm_campaign=long"
)
_LONG_COMMAND = "sha256sum --binary /var/log/auth.log | awk '{print $1}'"
_LONG_CODE = "def totally_invented_rewrite():\n    return 1"
_LONG_CODE_FENCE = f"```\n{_LONG_CODE}\n```"


def test_exact_content_phrases_preserve_hash_url_command_and_code() -> None:
    phrases = (
        "read that hash aloud",
        "read it aloud",
        "spell it",
        "spell out the UUID",
        "dictate the command",
        "say that exactly",
        "word for word",
        "character by character",
        "give me the exact URL",
    )
    for phrase in phrases:
        assert user_needs_exact_content(phrase) is True, phrase
        hash_spoken = _spoken(f"The hash is {_LONG_HASH}.", user_text=phrase)
        assert _LONG_HASH in hash_spoken, phrase
        url_spoken = _spoken(f"Open {_LONG_URL}", user_text=phrase)
        assert _LONG_URL in url_spoken, phrase
        command_spoken = _spoken(f"Run `{_LONG_COMMAND}`", user_text=phrase)
        assert "sha256sum --binary" in command_spoken, phrase
        code_spoken = _spoken(_LONG_CODE_FENCE, user_text=phrase)
        assert "def totally_invented_rewrite" in code_spoken, phrase
        assert "better viewed on screen" not in code_spoken.casefold(), phrase
        plan = _plan(user_text=phrase, canonical_text=_LONG_HASH)
        assert plan.exact_content_required is True
        assert plan.authorizes_privileged_action is False


def test_ordinary_read_explain_phrases_do_not_force_exact_content() -> None:
    negatives = (
        "read the article",
        "tell me about this",
        "explain this command",
        "what does this URL do?",
    )
    for phrase in negatives:
        assert user_needs_exact_content(phrase) is False, phrase
        plan = _plan(user_text=phrase)
        assert plan.exact_content_required is False
        url_spoken = _spoken(f"See {_LONG_URL}", user_text=phrase)
        assert "utm_campaign" not in url_spoken
        hash_spoken = _spoken(f"The hash is {_LONG_HASH}.", user_text=phrase)
        assert "better viewed on screen" in hash_spoken.casefold()
        code_spoken = _spoken(_LONG_CODE_FENCE, user_text=phrase)
        assert "def totally_invented_rewrite" not in code_spoken


def test_urls_are_not_read_character_by_character() -> None:
    spoken = _spoken(
        "See https://example.com/path?utm_source=tracker&utm_campaign=long"
    )
    assert "utm_campaign" not in spoken
    assert "link" in spoken.casefold()


# --- Plan model / failure / cleanup ---


def test_plan_is_frozen_typed_and_has_no_authority() -> None:
    plan = _plan()
    assert plan.authorizes_privileged_action is False
    try:
        plan.response_depth = "detailed"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("SpeechDeliveryPlan must be immutable")
    forbidden = {"audio", "pcm", "tool_id", "workflow_id", "authorize"}
    assert forbidden.isdisjoint(set(plan.__dataclass_fields__))
    hints = get_type_hints(SpeechDeliveryPlan)
    assert "bytes" not in {str(value) for value in hints.values()}


def test_normalization_and_chunking_failure_falls_back() -> None:
    plan = _plan()

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("boom")

    import src.speech_delivery as module

    original = module.normalize_for_speech
    module.normalize_for_speech = boom  # type: ignore[assignment]
    try:
        delivery = safe_prepare_spoken_delivery(
            "Keep this original answer.",
            plan,
        )
    finally:
        module.normalize_for_speech = original
    assert delivery.used_fallback is True
    assert delivery.spoken_text == "Keep this original answer."
    assert len(delivery.chunks) == 1


def test_queued_chunks_clear_on_abort_and_reset() -> None:
    state = SpeechDeliveryState()
    state.load_pending(["one", "two", "three"])
    assert state.pop_pending_chunk() == "one"
    state.abort_pending()
    assert state.pop_pending_chunk() is None
    state.load_pending(["again"])
    state.reset()
    assert state.pop_pending_chunk() is None
    assert state.last_spoken_opening is None


def test_required_notice_not_stripped() -> None:
    text = "I cannot authorize that tool. Approval required."
    assert contains_required_notice(text) is True
    plan = _plan(canonical_text=text, acknowledgment_hint="none")
    spoken = normalize_for_speech("Sure, I cannot authorize that tool.", plan)
    assert "cannot authorize" in spoken.casefold()


def test_delivery_instruction_block_is_advisory() -> None:
    block = format_speech_delivery_block(_plan())
    assert SPEECH_DELIVERY_BEGIN in block
    assert "never authorizes" in block.casefold()
    assert "not an independent" in block.casefold()


def test_plan_reuses_m27_guidance_fields() -> None:
    intel = ConversationIntelligence()
    state = ConversationState()
    guidance = intel.interpret("No, Tuesday.", state)
    plan = build_speech_delivery_plan(
        delivery_mode="realtime",
        interaction_mode="realtime",
        guidance=guidance,
        user_interrupted=True,
        user_text="No, Tuesday.",
    )
    assert plan.authorizes_privileged_action is False
    assert plan.timing_hint in {"interrupted_turn", "correction"}
    assert guidance.authorizes_privileged_action is False


def test_max_speech_chunks_merges_overflow_without_truncation() -> None:
    sentences = [
        f"Spoken pacing topic {index:02d} keeps unique words available."
        for index in range(1, 31)
    ]
    text = " ".join(sentences)
    plan = _plan(response_depth="detailed")
    chunks = chunk_spoken_text(text, plan)
    assert MAX_SPEECH_CHUNKS == 24
    assert len(chunks) <= MAX_SPEECH_CHUNKS
    assert len(chunks) == MAX_SPEECH_CHUNKS
    joined = " ".join(chunk.text for chunk in chunks)
    for index in range(1, 31):
        assert f"topic {index:02d}" in joined
    last = chunks[-1].text
    assert "topic 24" in last
    assert "topic 30" in last
    assert chunk_spoken_text(text, plan) == chunks
    delivery = prepare_spoken_delivery(text, plan)
    assert len(delivery.chunks) <= MAX_SPEECH_CHUNKS
    spoken_join = " ".join(chunk.text for chunk in delivery.chunks)
    for index in range(1, 31):
        assert f"topic {index:02d}" in spoken_join


def test_extreme_speech_keeps_tts_limit_without_content_loss() -> None:
    sentences = [
        f"Spoken pacing topic {index:03d} keeps unique words available for synthesis."
        for index in range(1, 201)
    ]
    text = " ".join(sentences)
    plan = _plan(response_depth="detailed")
    chunks = chunk_spoken_text(text, plan)
    assert all(len(chunk.text) <= MAX_TTS_CHARS for chunk in chunks)
    joined = " ".join(chunk.text for chunk in chunks)
    for index in range(1, 201):
        assert f"topic {index:03d}" in joined
    assert chunk_spoken_text(text, plan) == chunks
    exact_plan = _plan(
        response_depth="detailed",
        user_text="read that hash aloud exactly",
        canonical_text="a" * 5000,
    )
    exact_chunks = chunk_spoken_text("a" * 5000, exact_plan)
    assert all(len(chunk.text) <= MAX_TTS_CHARS for chunk in exact_chunks)
    assert "a" * 5000 == "".join(chunk.text for chunk in exact_chunks).replace(" ", "")


def test_normalization_preserves_technical_tokens() -> None:
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    git_hash = "a1b2c3d"
    text = (
        "Use -5 and 3.14 with v2.1.0 at 192.168.1.1 on 2026-08-13 at 11:30 PM. "
        r"Pass --force to C:\Users\name for file src/config.py. "
        f"Short hash {git_hash} and UUID {uuid}. "
        "Token git-status is the command-line token."
    )
    tokens = (
        "-5",
        "3.14",
        "v2.1.0",
        "192.168.1.1",
        "2026-08-13",
        "11:30 PM",
        "--force",
        r"C:\Users\name",
        git_hash,
        uuid,
        "src/config.py",
        "git-status",
    )
    spoken = _spoken(text)
    for token in tokens:
        assert token in spoken, token
    plan = _plan()
    chunks = chunk_spoken_text(spoken, plan)
    joined = " ".join(chunk.text for chunk in chunks)
    for token in tokens:
        assert token in joined, token
    assert chunk_spoken_text(spoken, plan) == chunks


def test_clear_session_delivery_state_drops_interrupt_keeps_conversation_memory() -> None:
    state = SpeechDeliveryState()
    state.record_completed_chunk("There are three options.")
    state.mark_interrupted("There are three options.")
    state.load_pending(["queued"])
    opening = state.last_spoken_opening
    fingerprints = list(state.recently_spoken_fingerprints)
    completed = state.last_completed_spoken_chunk
    assert state.interrupted_response_fingerprint is not None
    state.clear_session_delivery_state()
    assert state.interrupted_response_fingerprint is None
    assert state.pop_pending_chunk() is None
    assert state.pending_chunks == []
    assert state.last_spoken_opening == opening
    assert state.recently_spoken_fingerprints == fingerprints
    assert state.last_completed_spoken_chunk == completed


def test_clear_resets_speech_delivery_state() -> None:
    from src.commands import clear_conversation_history
    from src.conversation import ConversationHistory

    history = ConversationHistory()
    history.add_user_message("hi")
    history.add_assistant_message("hello")
    state = SpeechDeliveryState()
    state.record_completed_chunk("There are three options.")
    state.load_pending(["stale"])
    message = clear_conversation_history(
        history,
        speech_delivery_state=state,
    )
    assert "cleared" in message.casefold()
    assert history.turns == []
    assert state.last_spoken_opening is None
    assert state.pop_pending_chunk() is None


def test_load_pending_rejects_when_tts_chunk_cap_would_drop_tail() -> None:
    from src.config import MAX_TTS_CHARS
    from src.speech_delivery import MAX_PENDING_SPEECH_CHUNKS

    state = SpeechDeliveryState()
    chunks = ["x" * MAX_TTS_CHARS for _ in range(MAX_PENDING_SPEECH_CHUNKS + 8)]
    state.load_pending(chunks)
    assert state.pending_rejected is True
    assert state.pending_chunks == []
