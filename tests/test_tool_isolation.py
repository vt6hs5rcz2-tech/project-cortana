"""AI-isolation tests for Milestone 9 tool-control data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from src.active_memory import ActiveMemoryContext
from src.ai_service import AIResponse, OpenAIClient, ResponsesClient
from src.commands import handle_slash_command
from src.conversation import ConversationApiInput, ConversationHistory
from src.conversation_loop import handle_message
from src.document import create_document
from src.document_extractor import DefaultTextExtractor
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import build_default_tool_registry
from tests.tool_helpers import incident_repository, tool_repository

SCOPE_NOTE_MARKER = "M9_SCOPE_NOTE_MARKER_7e91aa"
JUSTIFICATION_MARKER = "M9_JUSTIFICATION_MARKER_3c22bb"
PARAM_MARKER = "M9_PARAM_QUERY_MARKER_88dd"
RESULT_MARKER = "M9_RESULT_SHOULD_NOT_LEAK"


@dataclass
class FakeAIResponse:
    output_text: str


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        input: ConversationApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "instructions": instructions,
            }
        )
        return FakeAIResponse(
            output_text=(
                '{"answer":"Isolated response [DOC-1:C1]",'
                '"support":"supported","citations":["[DOC-1:C1]"]}'
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


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _serialized_ai_payload(client: FakeClient) -> str:
    chunks: list[str] = []
    for call in client.fake_responses.calls:
        chunks.append(str(call["instructions"]))
        chunks.append(str(call["input"]))
    return "\n".join(chunks)


def _seed_tool_data(tmp_path: Path) -> None:
    repo = tool_repository(tmp_path)
    incidents = incident_repository(tmp_path)
    registry = build_default_tool_registry()
    executor = DefensiveToolExecutor(incident_repository=incidents)
    history = ConversationHistory()
    memory_store = JsonMemoryStore(tmp_path / "memories.json")
    active = ActiveMemoryContext()
    vault = JsonDocumentVault(tmp_path / "documents.json")
    client = cast(OpenAIClient, FakeClient())

    def run(message: str) -> None:
        handle_slash_command(
            message,
            settings=_settings(),
            conversation_history=history,
            memory_store=memory_store,
            active_memory_context=active,
            document_vault=vault,
            document_extractor=DefaultTextExtractor(),
            incident_repository=incidents,
            tool_registry=registry,
            tool_repository=repo,
            tool_executor=executor,
            client=client,
        )

    root = tmp_path / "root"
    root.mkdir()
    sample = root / "sample.txt"
    sample.write_text(f"line with {PARAM_MARKER}", encoding="utf-8")
    run(
        f"/scope-new Isolation | text-search,system-summary | {root} | "
        f"{SCOPE_NOTE_MARKER}"
    )
    scope_id = repo.list_scopes()[0].scope_id
    import json

    params = json.dumps({"path": str(sample), "query": PARAM_MARKER})
    run(
        f"/tool-request text-search | {scope_id} | {params} | {JUSTIFICATION_MARKER}"
    )
    request_id = repo.list_requests()[0].request_id
    run(f"/tool-dry-run {request_id}")
    run(f"/tool-approve {request_id} | approved for isolation test")
    run(f"/tool-run {request_id}")


def test_tool_markers_absent_from_ordinary_and_active_memory_ai(
    tmp_path: Path,
) -> None:
    _seed_tool_data(tmp_path)
    client = FakeClient()
    memory_store = JsonMemoryStore(tmp_path / "memories-ai.json")
    active = ActiveMemoryContext()
    memory = memory_store.add_memory("Active memory text only")
    active.activate(memory)

    handle_message(
        client=cast(OpenAIClient, client),
        settings=_settings(),
        user_message="Ordinary chat question",
        logger=__import__("logging").getLogger("ProjectCortana"),
        conversation_history=ConversationHistory(),
        active_memory_context=active,
    )

    payload = _serialized_ai_payload(client)
    for marker in (
        SCOPE_NOTE_MARKER,
        JUSTIFICATION_MARKER,
        PARAM_MARKER,
        RESULT_MARKER,
    ):
        assert marker not in payload


def test_tool_markers_absent_from_ask_docs(tmp_path: Path) -> None:
    _seed_tool_data(tmp_path)
    client = FakeClient()
    vault = JsonDocumentVault(tmp_path / "docs-ai.json")
    vault.add_document(
        create_document(
            filename="guide.txt",
            extension=".txt",
            source_size_bytes=20,
            content_hash=sha256(b"alpha beta gamma").hexdigest(),
            extracted_text="alpha beta gamma document evidence for retrieval",
        )
    )
    handle_slash_command(
        "/ask-docs What does the document say about gamma?",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories3.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        incident_repository=incident_repository(tmp_path),
        tool_registry=build_default_tool_registry(),
        tool_repository=tool_repository(tmp_path),
        tool_executor=DefensiveToolExecutor(),
        client=cast(OpenAIClient, client),
    )
    payload = _serialized_ai_payload(client)
    for marker in (SCOPE_NOTE_MARKER, JUSTIFICATION_MARKER, PARAM_MARKER):
        assert marker not in payload
