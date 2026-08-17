"""Command-surface tests for Milestone 23 vision slash commands."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from src.active_memory import ActiveMemoryContext
from src.ai_service import AIResponse, ResponsesApiInput, ResponsesClient
from src.commands import CommandOutcome, handle_slash_command
from src.conversation import ConversationHistory
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from src.vision_commands import VISION_ASK_USAGE, VISION_DESCRIBE_USAGE, VISION_INFO_USAGE


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="M23 vision file input requires Windows safe-open",
)


@dataclass
class FakeAIResponse:
    output_text: str


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0
        self.input: ResponsesApiInput | None = None

    def create(
        self,
        *,
        model: str,
        input: ResponsesApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        self.calls += 1
        self.input = input
        return FakeAIResponse(
            output_text=json.dumps(
                {
                    "answer": "A red rectangle is visible.",
                    "visibility": "observed",
                    "warning": None,
                }
            )
        )


class FakeClient:
    responses: ResponsesClient

    def __init__(self) -> None:
        self.responses = FakeResponses()

    @property
    def fake_responses(self) -> FakeResponses:
        assert isinstance(self.responses, FakeResponses)
        return self.responses


def _png(path: Path) -> Path:
    Image.new("RGB", (20, 10), "red").save(path, format="PNG")
    return path


def _run(
    message: str,
    *,
    tmp_path: Path,
    client: FakeClient | None = None,
    history: ConversationHistory | None = None,
) -> tuple[object, ConversationHistory, FakeClient]:
    active_client = client or FakeClient()
    conversation_history = history or ConversationHistory()
    result = handle_slash_command(
        message,
        settings=Settings(openai_api_key="test-key", openai_model="test-model"),
        conversation_history=conversation_history,
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        client=active_client,
    )
    return result, conversation_history, active_client


def test_vision_info_no_ai_call(tmp_path: Path) -> None:
    path = _png(tmp_path / "info.png")
    client = FakeClient()
    result, history, _ = _run(
        f'/vision-info "{path.resolve()}"',
        tmp_path=tmp_path,
        client=client,
    )
    assert result.outcome == CommandOutcome.CONTINUE
    assert result.message is not None
    assert "Vision image info" in result.message
    assert "info.png" in result.message
    assert "Detected format: PNG" in result.message
    assert str(path.resolve()) not in result.message
    assert client.fake_responses.calls == 0
    assert history.completed_turn_count == 0


def test_vision_describe_and_ask(tmp_path: Path) -> None:
    path = _png(tmp_path / "scene.png")
    result, history, client = _run(
        f"/vision-describe {path.resolve()}",
        tmp_path=tmp_path,
    )
    assert result.message is not None
    assert "A red rectangle is visible." in result.message
    assert "Visibility: observed" in result.message
    assert client.fake_responses.calls == 1
    assert history.turns == []

    result, history, client = _run(
        f"/vision-ask {path.resolve()} | What color is dominant?",
        tmp_path=tmp_path,
        history=history,
        client=client,
    )
    assert result.message is not None
    assert "A red rectangle is visible." in result.message
    assert client.fake_responses.calls == 2
    assert history.turns == []
    assert client.fake_responses.input is not None
    content = client.fake_responses.input[0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "What color is dominant?"


def test_vision_command_usage_errors(tmp_path: Path) -> None:
    result, _, client = _run("/vision-describe", tmp_path=tmp_path)
    assert result.message == VISION_DESCRIBE_USAGE
    assert client.fake_responses.calls == 0

    result, _, client = _run("/vision-ask only-one-field", tmp_path=tmp_path)
    assert result.message == VISION_ASK_USAGE

    result, _, client = _run("/vision-info", tmp_path=tmp_path)
    assert result.message == VISION_INFO_USAGE


def test_vision_disabled_gates_all_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _png(tmp_path / "x.png")
    monkeypatch.setattr("src.vision_commands.VISION_ANALYSIS_ENABLED", False)
    monkeypatch.setattr("src.commands.VISION_ANALYSIS_ENABLED", False)
    monkeypatch.setattr("src.vision_input.VISION_ANALYSIS_ENABLED", False)

    for message in (
        f"/vision-describe {path.resolve()}",
        f"/vision-ask {path.resolve()} | hi",
        f"/vision-info {path.resolve()}",
    ):
        result, history, client = _run(message, tmp_path=tmp_path)
        assert result.message is not None
        assert "disabled" in result.message.casefold()
        assert client.fake_responses.calls == 0
        assert history.turns == []


def test_help_and_status_include_vision(tmp_path: Path) -> None:
    help_result, _, _ = _run("/help", tmp_path=tmp_path)
    assert help_result.message is not None
    assert "/vision-describe" in help_result.message
    assert "/vision-ask" in help_result.message
    assert "/vision-info" in help_result.message

    status_result, _, _ = _run("/status verbose", tmp_path=tmp_path)
    assert status_result.message is not None
    assert "Visual analysis: enabled" in status_result.message


def test_vision_info_rejects_directory(tmp_path: Path) -> None:
    result, _, client = _run(
        f"/vision-info {tmp_path.resolve()}",
        tmp_path=tmp_path,
    )
    assert result.message is not None
    assert "could not be accessed" in result.message.casefold() or (
        "could not be found" in result.message.casefold()
        or "image" in result.message.casefold()
    )
    assert client.fake_responses.calls == 0
