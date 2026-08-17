"""Isolated SDK/schema proof for M26 explicit visual-response context.

These tests inspect the installed OpenAI SDK and current production call
site. They do not call the live API and do not change M26 routing.
"""

from __future__ import annotations

import inspect
from typing import get_args, get_type_hints

from openai.types.beta.realtime.conversation_item_with_reference_param import (
    ConversationItemWithReferenceParam,
)
from openai.types.realtime.conversation_item_param import ConversationItemParam
from openai.types.realtime.realtime_conversation_item_user_message_param import (
    Content,
    RealtimeConversationItemUserMessageParam,
)
from openai.types.realtime.realtime_response_create_params_param import (
    RealtimeResponseCreateParamsParam,
)
from openai.types.realtime.response_create_event_param import ResponseCreateEventParam

from src.realtime_multimodal import RealtimeMultimodalSession


def test_ga_response_create_params_include_input_instructions_conversation() -> None:
    hints = get_type_hints(RealtimeResponseCreateParamsParam)
    assert "input" in hints
    assert "instructions" in hints
    assert "conversation" in hints
    assert "output_modalities" in hints


def test_ga_response_create_event_carries_response_params() -> None:
    hints = get_type_hints(ResponseCreateEventParam)
    assert hints["response"] is RealtimeResponseCreateParamsParam


def test_ga_conversation_item_union_has_no_item_reference_variant() -> None:
    variant_names = {getattr(variant, "__name__", "") for variant in get_args(ConversationItemParam)}
    assert "RealtimeConversationItemUserMessageParam" in variant_names
    assert not any("Reference" in name or "ItemReference" in name for name in variant_names)


def test_ga_user_message_supports_input_image_and_image_url() -> None:
    content_hints = get_type_hints(Content)
    user_hints = get_type_hints(RealtimeConversationItemUserMessageParam)
    assert content_hints["image_url"] is str
    assert "input_image" in str(content_hints["type"])
    assert "input_text" in str(content_hints["type"])
    assert user_hints["type"].__args__[0] == "message"
    assert user_hints["role"].__args__[0] == "user"
    assert "id" in user_hints


def test_beta_realtime_schema_documents_item_reference() -> None:
    hints = get_type_hints(ConversationItemWithReferenceParam)
    type_values = str(hints["type"])
    assert "item_reference" in type_values
    assert "id" in hints


def test_current_m26_response_create_is_bare() -> None:
    source = inspect.getsource(RealtimeMultimodalSession._issue_response_create)
    assert "connection.response.create()" in source
    assert "input" not in source
    assert "item_reference" not in source
    assert "instructions" not in source
