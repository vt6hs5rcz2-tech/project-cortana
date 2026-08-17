"""Command-surface tests for Milestone 22 Study Partner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from src.active_memory import ActiveMemoryContext
from src.ai_service import AIResponse, OpenAIClient, ResponsesClient
from src.commands import handle_slash_command
from src.conversation import ConversationApiInput, ConversationHistory
from src.document import create_document
from src.document_chunker import DocumentChunker
from src.document_extractor import DefaultTextExtractor
from src.document_knowledge_service import DocumentKnowledgeService
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.memory_store import JsonMemoryStore
from src.retrieval_session import RetrievalSession
from src.settings import Settings
from src.study_commands import create_default_study_service
from src.study_service import format_study_question_message, format_study_status_message


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class FakeAIResponse:
    output_text: str


class FakeResponses:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs or []
        self.calls = 0
        self.inputs: list[ConversationApiInput] = []

    def create(
        self,
        *,
        model: str,
        input: ConversationApiInput,
        instructions: str | None = None,
    ) -> AIResponse:
        self.calls += 1
        self.inputs.append(input)
        index = min(self.calls - 1, len(self.outputs) - 1)
        return FakeAIResponse(output_text=self.outputs[index])


class FakeClient:
    responses: ResponsesClient

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.responses = FakeResponses(outputs)


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def _mcq_json() -> str:
    return json.dumps(
        {
            "question_type": "mcq",
            "prompt": "What does containment require?",
            "choices": {
                "A": "isolation",
                "B": "ignore",
                "C": "delete",
                "D": "reboot",
            },
            "correct_answer": "SECRET_ANSWER_KEY_XYZ",
            "explanation": "SECRET_EXPLANATION_XYZ",
            "primary_citation": "[DOC-1:C1]",
            "citations": ["[DOC-1:C1]"],
        }
    )


def _valid_mcq_json() -> str:
    return json.dumps(
        {
            "question_type": "mcq",
            "prompt": "What does containment require?",
            "choices": {
                "A": "isolation",
                "B": "ignore",
                "C": "delete",
                "D": "reboot",
            },
            "correct_answer": "A",
            "explanation": "SECRET_EXPLANATION_XYZ",
            "primary_citation": "[DOC-1:C1]",
            "citations": ["[DOC-1:C1]"],
        }
    )


def _env(tmp_path: Path, outputs: list[str] | None = None):
    vault = JsonDocumentVault(tmp_path / "documents.json")
    text = "Containment requires isolation of the host."
    record = create_document(
        filename="guide.txt",
        extension=".txt",
        source_size_bytes=len(text.encode("utf-8")),
        content_hash=_hash(text),
        extracted_text=text,
    )
    doc_id = vault.add_document(record).id
    history = ConversationHistory()
    memory = JsonMemoryStore(tmp_path / "memories.json")
    active = ActiveMemoryContext()
    retriever = LexicalDocumentRetriever()
    retrieval_session = RetrievalSession(boundary_token="cmd_study_token_01")
    client = FakeClient(outputs or [_valid_mcq_json()])
    knowledge = DocumentKnowledgeService(
        vault=vault,
        retriever=retriever,
        retrieval_session=retrieval_session,
        settings=_settings(),
        client=cast(OpenAIClient, client),
        chunker=DocumentChunker(),
    )
    study = create_default_study_service(
        vault=vault,
        knowledge_service=knowledge,
        settings=_settings(),
        client=cast(OpenAIClient, client),
        repository_file_path=tmp_path / "study_state.json",
        chunker=DocumentChunker(),
        document_retriever=retriever,
        retrieval_session=retrieval_session,
    )
    return doc_id, history, memory, active, vault, retriever, retrieval_session, study, client


def _run(
    message: str,
    *,
    tmp_path: Path,
    history: ConversationHistory,
    memory: JsonMemoryStore,
    active: ActiveMemoryContext,
    vault: JsonDocumentVault,
    retriever: LexicalDocumentRetriever,
    retrieval_session: RetrievalSession,
    study,
    client: FakeClient,
):
    return handle_slash_command(
        message,
        settings=_settings(),
        conversation_history=history,
        memory_store=memory,
        active_memory_context=active,
        document_vault=vault,
        document_extractor=DefaultTextExtractor(),
        document_retriever=retriever,
        retrieval_session=retrieval_session,
        study_service=study,
        client=cast(OpenAIClient, client),
    )


def test_study_command_flow_and_answer_key_hidden(tmp_path: Path) -> None:
    doc_id, history, memory, active, vault, retriever, session, study, client = _env(
        tmp_path
    )
    start = _run(
        f"/study-start {doc_id}",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert start.message is not None
    assert "started" in start.message.casefold()

    question = _run(
        "/study-question mcq | -",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert question.message is not None
    assert "SECRET_EXPLANATION_XYZ" not in question.message
    assert "correct_answer" not in question.message

    status = _run(
        "/study-status",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert status.message is not None
    assert "SECRET_EXPLANATION_XYZ" not in status.message

    global_status = _run(
        "/status verbose",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert global_status.message is not None
    assert "Study Partner: enabled" in global_status.message
    assert "Active study session: yes" in global_status.message
    assert "SECRET_EXPLANATION_XYZ" not in global_status.message
    assert "What does containment require?" not in global_status.message

    answer = _run(
        "/study-answer A",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert answer.message is not None
    assert "Correct" in answer.message
    assert "SECRET_EXPLANATION_XYZ" in answer.message
    assert history.turns == []


def test_study_question_grammar_requires_two_fields(tmp_path: Path) -> None:
    doc_id, history, memory, active, vault, retriever, session, study, client = _env(
        tmp_path
    )
    _run(
        f"/study-start {doc_id}",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    bad = _run(
        "/study-question mcq",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert bad.message is not None
    assert "Usage: /study-question" in bad.message


def test_feature_flag_disables_study_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.study_commands.STUDY_PARTNER_ENABLED", False)
    monkeypatch.setattr("src.commands.STUDY_PARTNER_ENABLED", False)
    doc_id, history, memory, active, vault, retriever, session, study, client = _env(
        tmp_path
    )
    result = _run(
        f"/study-start {doc_id}",
        tmp_path=tmp_path,
        history=history,
        memory=memory,
        active=active,
        vault=vault,
        retriever=retriever,
        retrieval_session=session,
        study=study,
        client=client,
    )
    assert result.message is not None
    assert "disabled" in result.message.casefold()


def test_formatters_type_signatures_hide_keys() -> None:
    source = Path("src/study_service.py").read_text(encoding="utf-8")
    # Public formatters must accept public/status views, not full StudyQuestion.
    assert "def format_study_question_message(view: StudyQuestionPublicView)" in source
    assert "def format_study_status_message(view: StudyStatusView)" in source
    assert ".correct_answer" not in format_study_question_message.__code__.co_names
    # Ensure status formatter source does not read answer-key attributes.
    status_src = format_study_status_message.__code__.co_names
    assert "correct_answer" not in status_src
    assert "explanation" not in status_src
