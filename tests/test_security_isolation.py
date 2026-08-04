"""Privacy and AI-isolation tests for Milestone 8 incident data."""

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
from src.evidence_store import EvidenceStore, LocalEvidenceStore
from src.incident_repository import IncidentRepository, JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.settings import Settings
from tests.security_helpers import evidence_store, incident_repository

EVENT_MARKER = "M8_EVENT_DESC_MARKER_7f3a91"
INCIDENT_MARKER = "M8_INCIDENT_SUMMARY_MARKER_2c88de"
NOTE_MARKER = "M8_NOTE_TEXT_MARKER_9aa012"
INDICATOR_MARKER = "m8-indicator-marker-unique.example"
EVIDENCE_TITLE_MARKER = "M8_EVIDENCE_TITLE_MARKER_55bb"
EVIDENCE_DESC_MARKER = "M8_EVIDENCE_DESC_MARKER_66cc"


@dataclass
class FakeAIResponse:
    """Minimal AI response used by the fake Responses API."""

    output_text: str


class FakeResponses:
    """Fake Responses API used without network access."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        input: ConversationApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        """Record the request and return a fake response."""
        self.calls.append(
            {
                "model": model,
                "input": input,
                "instructions": instructions,
            }
        )
        return FakeAIResponse(output_text="Isolated response")


class FakeClient:
    """Fake OpenAI client containing the fake Responses API."""

    responses: ResponsesClient

    def __init__(self) -> None:
        self.responses = FakeResponses()

    @property
    def fake_responses(self) -> FakeResponses:
        """Return the concrete fake Responses API for test assertions."""
        assert isinstance(self.responses, FakeResponses)
        return self.responses


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _seed_incident_data(
    tmp_path: Path,
) -> tuple[JsonIncidentRepository, LocalEvidenceStore]:
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    history = ConversationHistory()
    memory_store = JsonMemoryStore(tmp_path / "memories.json")
    active_memory_context = ActiveMemoryContext()
    document_vault = JsonDocumentVault(tmp_path / "documents.json")
    document_extractor = DefaultTextExtractor()
    client = cast(OpenAIClient, FakeClient())

    def run(message: str) -> None:
        handle_slash_command(
            message,
            settings=_settings(),
            conversation_history=history,
            memory_store=memory_store,
            active_memory_context=active_memory_context,
            document_vault=document_vault,
            document_extractor=document_extractor,
            incident_repository=repo,
            evidence_store=store,
            client=client,
        )

    run(f"/event-new low | Event title | {EVENT_MARKER}")
    run(f"/incident-new low | Incident title | {INCIDENT_MARKER}")
    incident_id = repo.list_incidents()[0].incident_id
    run(f"/indicator-add domain | {INDICATOR_MARKER} | 40")
    run(f"/incident-add-note {incident_id} | observation | {NOTE_MARKER}")
    source = tmp_path / "evidence.txt"
    source.write_text("opaque-bytes", encoding="utf-8")
    run(
        f"/evidence-register {source} | {EVIDENCE_TITLE_MARKER} | {EVIDENCE_DESC_MARKER}"
    )
    return repo, store


def _serialized_ai_payload(client: FakeClient) -> str:
    chunks: list[str] = []
    for call in client.fake_responses.calls:
        chunks.append(str(call["instructions"]))
        chunks.append(str(call["input"]))
    return "\n".join(chunks)


def _assert_markers_absent(payload: str) -> None:
    for marker in (
        EVENT_MARKER,
        INCIDENT_MARKER,
        NOTE_MARKER,
        INDICATOR_MARKER,
        EVIDENCE_TITLE_MARKER,
        EVIDENCE_DESC_MARKER,
    ):
        assert marker not in payload


def test_incident_markers_absent_from_ordinary_and_active_memory_ai(
    tmp_path: Path,
) -> None:
    """Incident foundation markers must not appear in ordinary or active-memory AI calls."""
    repo, _store = _seed_incident_data(tmp_path)
    assert repo.event_count() == 1

    client = FakeClient()
    memory_store = JsonMemoryStore(tmp_path / "memories.json")
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

    _assert_markers_absent(_serialized_ai_payload(client))


def test_incident_markers_absent_from_ask_docs(tmp_path: Path) -> None:
    """Incident foundation markers must not appear in /ask-docs AI requests."""
    _seed_incident_data(tmp_path)
    client = FakeClient()
    vault = JsonDocumentVault(tmp_path / "documents.json")
    vault.add_document(
        create_document(
            filename="guide.txt",
            extension=".txt",
            source_size_bytes=20,
            content_hash=sha256(b"alpha beta gamma").hexdigest(),
            extracted_text="alpha beta gamma document evidence for retrieval",
        )
    )

    result = handle_slash_command(
        "/ask-docs What does the document say about gamma?",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories2.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        incident_repository=cast(IncidentRepository, incident_repository(tmp_path)),
        evidence_store=cast(EvidenceStore, evidence_store(tmp_path)),
        client=cast(OpenAIClient, client),
    )

    assert result.message is not None
    _assert_markers_absent(_serialized_ai_payload(client))


def test_security_command_output_hides_evidence_paths(tmp_path: Path) -> None:
    """Command output must not expose evidence storage or source absolute paths."""
    repo = incident_repository(tmp_path)
    store = evidence_store(tmp_path)
    source = tmp_path / "secret-path-name.txt"
    source.write_text("bytes", encoding="utf-8")
    result = handle_slash_command(
        f"/evidence-register {source} | Title | Description",
        settings=_settings(),
        conversation_history=ConversationHistory(),
        memory_store=JsonMemoryStore(tmp_path / "memories.json"),
        active_memory_context=ActiveMemoryContext(),
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_extractor=DefaultTextExtractor(),
        incident_repository=repo,
        evidence_store=store,
        client=cast(OpenAIClient, FakeClient()),
    )
    assert result.message is not None
    assert str(source) not in result.message
    assert str(store.directory_path) not in result.message
