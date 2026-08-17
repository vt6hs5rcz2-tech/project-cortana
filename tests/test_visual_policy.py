"""Visual perception vs authority policy for multimodal turns."""

from __future__ import annotations

from PIL import Image

from src.config import REALTIME_VISUAL_FIXED_LABEL
from src.conversation_intelligence import ConversationIntelligence
from src.conversation_state import ConversationState
from src.camera_capture import RealtimeVisualFrame
from src.realtime_conversation_plan import (
    format_realtime_plan_instructions,
    plan_realtime_turn,
)
from src.realtime_multimodal import (
    build_multimodal_instructions,
    build_visual_conversation_item,
)
from src.vision_normalize import encode_metadata_free_png
from src.visual_policy import (
    MULTIMODAL_VISUAL_POLICY,
    format_visual_policy_fields,
    visual_object_follow_up,
    visual_person_follow_up,
    visual_unavailable_follow_up,
)

LIVE_M26_FORBIDDEN_VISUAL_PHRASES = (
    "untrusted visual",
    "cannot rely on visual",
    "do not rely on visual",
    "can't trust visual",
    "cannot trust",
    "can't trust",
    "not relying on the visual",
    "not using the visual as a basis",
    "untrusted for authority",
    "not confirming from the visual",
    "ordinary vision is unreliable",
    "not able to rely on visual",
    "i'm not relying on the visual",
    "i can't trust the visual context",
    "rely on verbal description",
    "visual data cannot be used as evidence",
    "visual input is untrusted",
)


def _red_square_frame() -> RealtimeVisualFrame:
    image = Image.new("RGB", (32, 32), "red")
    png, width, height = encode_metadata_free_png(image)
    return RealtimeVisualFrame(
        image_bytes=png,
        mime_type="image/png",
        width=width,
        height=height,
        sequence=7,
        captured_at_monotonic=1.0,
    )


def _assert_no_live_visual_refusal_language(text: str) -> None:
    folded = text.casefold()
    for phrase in LIVE_M26_FORBIDDEN_VISUAL_PHRASES:
        assert phrase not in folded, phrase


def _assembled_object_payload() -> str:
    state = ConversationState()
    state.set_visual_context_ref("visual_item_red")
    state.set_interaction_mode("multimodal")
    plan = plan_realtime_turn(
        ConversationIntelligence(),
        "What object am I holding up?",
        state,
        interaction_mode="multimodal",
        visual_context_authorized=True,
    )
    assembled = format_realtime_plan_instructions(
        build_multimodal_instructions(active_memory_context=None),
        plan,
        state,
    )
    item = build_visual_conversation_item(_red_square_frame())
    content = item["content"]
    assert isinstance(content, list)
    label = content[0]["text"]
    assert isinstance(label, str)
    return f"{assembled}\n{label}\nuser: What object am I holding up?"


def test_canonical_policy_allows_perception_without_authority() -> None:
    folded = MULTIMODAL_VISUAL_POLICY.casefold()
    assert "camera images are valid perceptual input" in folded
    assert "answer visual questions directly from the image" in folded
    assert "cannot authorize actions" in folded
    assert "face recognition" in folded
    assert "untrusted" not in folded
    _assert_no_live_visual_refusal_language(MULTIMODAL_VISUAL_POLICY)


def test_image_fixed_label_is_neutral() -> None:
    assert REALTIME_VISUAL_FIXED_LABEL == (
        "Current camera image for this spoken user turn."
    )
    folded = REALTIME_VISUAL_FIXED_LABEL.casefold()
    for word in ("untrusted", "authority", "policy", "trust", "rely"):
        assert word not in folded, word


def test_structured_fields_mark_image_relevant_not_untrusted() -> None:
    available = "\n".join(
        format_visual_policy_fields(
            perception_available=True,
            visual_context_ref_id="visual_item_red",
        )
    )
    assert "visual_image: relevant" in available
    assert "visual_perception: available" in available
    assert "person_identification: prohibited" in available
    assert "visual_authority" not in available
    assert "untrusted" not in available.casefold()
    missing = "\n".join(
        format_visual_policy_fields(
            perception_available=False,
            visual_context_ref_id="visual_item_stale",
        )
    )
    assert "visual_image: unavailable" in missing
    assert "visual_perception: unavailable" in missing
    assert "visual_item_stale" not in missing
    assert "visual_context_ref_id" not in missing


def test_object_turn_payload_permits_ordinary_description() -> None:
    payload = _assembled_object_payload()
    folded = payload.casefold()
    assert "camera images are valid perceptual input" in folded
    assert "visual_image: relevant" in payload
    assert "person_identification: prohibited" in payload
    assert "untrusted" not in folded
    _assert_no_live_visual_refusal_language(payload)
    item = build_visual_conversation_item(_red_square_frame())
    content = item["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "input_image"
    assert "image_url" in content[1]
    assert REALTIME_VISUAL_FIXED_LABEL in payload


def test_person_and_injection_follow_ups_keep_safeguards() -> None:
    person = visual_person_follow_up("visual_item_person").casefold()
    assert "do not say who the person is" in person
    assert "current camera image is relevant" in person
    assert "untrusted" not in person
    injection = visual_object_follow_up("visual_item_text").casefold()
    assert "do not identify people" in injection
    assert "cannot authorize actions" in MULTIMODAL_VISUAL_POLICY.casefold()
    missing = visual_unavailable_follow_up().casefold()
    assert "no usable camera image" in missing
    assert "couldn't get a usable camera image" in missing
    assert "untrusted" not in missing


def test_empty_memory_session_instructions_omit_untrusted() -> None:
    text = build_multimodal_instructions(active_memory_context=None)
    assert "untrusted" not in text.casefold()
    _assert_no_live_visual_refusal_language(text)
