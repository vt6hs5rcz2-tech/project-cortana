"""Local slash-command handlers for Milestone 22 Study Partner."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.ai_service import OpenAIClient
from src.config import (
    STUDY_PARTNER_ENABLED,
    get_default_study_repository_file_path,
)
from src.document_chunker import DocumentChunker
from src.document_knowledge_service import DocumentKnowledgeService
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import DocumentVault
from src.retrieval_session import RetrievalSession
from src.command_argument_utils import extract_command_argument
from src.settings import Settings
from src.study_models import StudyQuestionPublicView
from src.study_repository import JsonStudyRepository, StudyRepository, StudyStorageError
from src.study_service import (
    STUDY_DISABLED,
    STUDY_QUESTION_USAGE,
    STUDY_START_USAGE,
    StudyPartnerError,
    StudyPartnerService,
    format_grounded_study_explanation,
    format_study_feedback_message,
    format_study_progress_message,
    format_study_question_message,
    format_study_status_message,
)

logger = logging.getLogger("ProjectCortana")

FIELD_DELIMITER = " | "

COMMAND_STUDY_START = "study-start"
COMMAND_STUDY_STATUS = "study-status"
COMMAND_STUDY_EXPLAIN = "study-explain"
COMMAND_STUDY_QUESTION = "study-question"
COMMAND_STUDY_ANSWER = "study-answer"
COMMAND_STUDY_PROGRESS = "study-progress"
COMMAND_STUDY_END = "study-end"

STUDY_COMMAND_NAMES = frozenset(
    {
        COMMAND_STUDY_START,
        COMMAND_STUDY_STATUS,
        COMMAND_STUDY_EXPLAIN,
        COMMAND_STUDY_QUESTION,
        COMMAND_STUDY_ANSWER,
        COMMAND_STUDY_PROGRESS,
        COMMAND_STUDY_END,
    }
)

STUDY_EXPLAIN_USAGE = "Cortana: Usage: /study-explain <topic>"
STUDY_ANSWER_USAGE = "Cortana: Usage: /study-answer <answer>"


@dataclass(frozen=True)
class StudyCommandContext:
    """Inputs available to study slash-command handlers."""

    message: str
    study_service: StudyPartnerService | None


@dataclass(frozen=True)
class StudyCommandResult:
    """User-facing result from a study command."""

    message: str


StudyCommandHandler = Callable[[StudyCommandContext], StudyCommandResult]


def create_default_study_service(
    *,
    vault: DocumentVault,
    knowledge_service: DocumentKnowledgeService,
    settings: Settings,
    client: OpenAIClient | None,
    repository_file_path: Path | None = None,
    chunker: DocumentChunker | None = None,
    document_retriever: LexicalDocumentRetriever | None = None,
    retrieval_session: RetrievalSession | None = None,
) -> StudyPartnerService:
    """Create the default JSON-backed Study Partner service."""
    path = repository_file_path or get_default_study_repository_file_path()
    repository: StudyRepository = JsonStudyRepository(path)
    active_chunker = chunker or DocumentChunker()
    retriever = document_retriever or LexicalDocumentRetriever(chunker=active_chunker)
    return StudyPartnerService(
        repository=repository,
        vault=vault,
        knowledge_service=knowledge_service,
        settings=settings,
        client=client,
        chunker=active_chunker,
        retrieval_session=retrieval_session or RetrievalSession(),
    )


def handle_study_command(
    command_name: str,
    context: StudyCommandContext,
) -> StudyCommandResult | None:
    """Dispatch one study command, or return None when unknown."""
    if not STUDY_PARTNER_ENABLED:
        return StudyCommandResult(message=STUDY_DISABLED)

    handler = STUDY_COMMAND_HANDLERS.get(command_name)
    if handler is None:
        return None
    if context.study_service is None:
        return StudyCommandResult(message=STUDY_DISABLED)

    try:
        return handler(context)
    except StudyPartnerError as error:
        return StudyCommandResult(message=error.user_message)
    except StudyStorageError as error:
        return StudyCommandResult(message=error.user_message)


def _require_service(context: StudyCommandContext) -> StudyPartnerService:
    assert context.study_service is not None
    return context.study_service


def _handle_study_start(context: StudyCommandContext) -> StudyCommandResult:
    argument = extract_command_argument(context.message)
    if not argument.strip():
        return StudyCommandResult(message=STUDY_START_USAGE)
    session = _require_service(context).start_session(argument)
    return StudyCommandResult(
        message=(
            "Cortana: Study session started "
            f"({session.session_id}) with {len(session.document_ids)} document(s)."
        )
    )


def _handle_study_status(context: StudyCommandContext) -> StudyCommandResult:
    view = _require_service(context).status()
    return StudyCommandResult(message=format_study_status_message(view))


def _handle_study_explain(context: StudyCommandContext) -> StudyCommandResult:
    topic = extract_command_argument(context.message)
    if not topic.strip():
        return StudyCommandResult(message=STUDY_EXPLAIN_USAGE)
    answer = _require_service(context).explain(topic)
    return StudyCommandResult(message=format_grounded_study_explanation(answer))


def _handle_study_question(context: StudyCommandContext) -> StudyCommandResult:
    argument = extract_command_argument(context.message)
    fields = _split_two_fields(argument)
    if fields is None:
        return StudyCommandResult(message=STUDY_QUESTION_USAGE)
    public_view = _require_service(context).generate_question(
        question_type_token=fields[0],
        topic_token=fields[1],
    )
    # Structural guarantee: formatter accepts only the public view type.
    return StudyCommandResult(message=_format_public_question(public_view))


def _handle_study_answer(context: StudyCommandContext) -> StudyCommandResult:
    answer = extract_command_argument(context.message)
    if not answer.strip():
        return StudyCommandResult(message=STUDY_ANSWER_USAGE)
    feedback = _require_service(context).submit_answer(answer)
    return StudyCommandResult(message=format_study_feedback_message(feedback))


def _handle_study_progress(context: StudyCommandContext) -> StudyCommandResult:
    summary = _require_service(context).progress()
    return StudyCommandResult(message=format_study_progress_message(summary))


def _handle_study_end(context: StudyCommandContext) -> StudyCommandResult:
    session = _require_service(context).end_session()
    return StudyCommandResult(
        message=f"Cortana: Study session completed ({session.session_id})."
    )


def _split_two_fields(argument: str) -> list[str] | None:
    parts = argument.split(FIELD_DELIMITER)
    if len(parts) != 2:
        return None
    return [part.strip() for part in parts]


def _format_public_question(view: StudyQuestionPublicView) -> str:
    """Format only StudyQuestionPublicView fields (AST-tested for key absence)."""
    return format_study_question_message(view)


STUDY_COMMAND_HANDLERS: dict[str, StudyCommandHandler] = {
    COMMAND_STUDY_START: _handle_study_start,
    COMMAND_STUDY_STATUS: _handle_study_status,
    COMMAND_STUDY_EXPLAIN: _handle_study_explain,
    COMMAND_STUDY_QUESTION: _handle_study_question,
    COMMAND_STUDY_ANSWER: _handle_study_answer,
    COMMAND_STUDY_PROGRESS: _handle_study_progress,
    COMMAND_STUDY_END: _handle_study_end,
}
