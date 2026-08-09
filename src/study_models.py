"""Domain models for Milestone 22 Study Partner / Tutor.

Study Partner helps the user learn and practice authorized Knowledge Vault
material. Citation labels on generated questions are validated structurally
against the source packet allowlist; valid citations do not prove semantic
entailment of the question or answer key against the source text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from src.config import (
    MAX_STUDY_CHOICE_CHARS,
    MAX_STUDY_CORRECT_ANSWER_CHARS,
    MAX_STUDY_DOCUMENTS,
    MAX_STUDY_EXPLANATION_CHARS,
    MAX_STUDY_FEEDBACK_CHARS,
    MAX_STUDY_QUESTION_PROMPT_CHARS,
    MAX_STUDY_USER_ANSWER_CHARS,
)
from src.document import DocumentValidationError, validate_document_id
from src.document_chunk import build_chunk_id
from src.tool_common import (
    BlankToolFieldError,
    InvalidToolEnumError,
    ToolFieldTooLongError,
    ToolValidationError,
    require_non_blank_text,
    utc_timestamp,
    validate_controlled_value,
    validate_optional_uuid,
    validate_utc_timestamp,
    validate_uuid,
)

StudySessionStatus = Literal["active", "completed"]
StudyQuestionType = Literal["mcq", "short"]
StudyQuestionStatus = Literal["pending", "answered"]
StudyAttemptResult = Literal["correct", "partially_correct", "incorrect"]

STUDY_SESSION_STATUSES: frozenset[str] = frozenset({"active", "completed"})
STUDY_QUESTION_TYPES: frozenset[str] = frozenset({"mcq", "short"})
STUDY_QUESTION_STATUSES: frozenset[str] = frozenset({"pending", "answered"})
STUDY_ATTEMPT_RESULTS: frozenset[str] = frozenset(
    {"correct", "partially_correct", "incorrect"}
)
MCQ_CHOICE_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
MCQ_CHOICE_LABEL_SET: frozenset[str] = frozenset(MCQ_CHOICE_LABELS)


class StudyValidationError(ValueError):
    """Raised when a study domain record fails local validation."""


@dataclass(frozen=True)
class StudySourceRef:
    """Stable reference to one vault chunk used by a study question."""

    document_id: str
    chunk_id: str
    chunk_index: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class StudySession:
    """Persistent study session over an explicit authorized document set."""

    session_id: str
    document_ids: tuple[str, ...]
    status: StudySessionStatus
    created_at: str
    updated_at: str
    pending_question_id: str | None


@dataclass(frozen=True)
class StudyQuestion:
    """Persisted practice question including the hidden answer key."""

    question_id: str
    session_id: str
    question_type: StudyQuestionType
    prompt: str
    choices: dict[str, str] | None
    correct_answer: str = field(repr=False)
    explanation: str = field(repr=False)
    source_refs: tuple[StudySourceRef, ...]
    primary_source_ref: StudySourceRef
    created_at: str
    status: StudyQuestionStatus


@dataclass(frozen=True)
class StudyAttempt:
    """Persisted graded attempt without raw user answer text."""

    attempt_id: str
    session_id: str
    question_id: str
    primary_chunk_id: str
    result: StudyAttemptResult
    submitted_at: str


@dataclass(frozen=True)
class StudyChunkStats:
    """Deterministic per-chunk practice statistics for adaptation."""

    chunk_id: str
    attempts: int
    correct: int
    partial: int
    incorrect: int
    last_attempt_at: str


@dataclass(frozen=True)
class StudyQuestionPublicView:
    """Safe pre-answer question projection without the answer key."""

    question_id: str
    session_id: str
    question_type: StudyQuestionType
    prompt: str
    choices: dict[str, str] | None


@dataclass(frozen=True)
class StudyQuestionFeedbackView:
    """Post-grading feedback projection that may reveal the answer key."""

    question_id: str
    session_id: str
    question_type: StudyQuestionType
    prompt: str
    choices: dict[str, str] | None
    correct_answer: str
    explanation: str
    result: StudyAttemptResult
    feedback: str
    source_refs: tuple[StudySourceRef, ...]
    primary_source_ref: StudySourceRef
    source_citations_available: bool


def create_study_session(
    *,
    document_ids: tuple[str, ...] | list[str],
    session_id: str | None = None,
    created_at: str | None = None,
) -> StudySession:
    """Create a validated active study session."""
    now = created_at or utc_timestamp()
    return validate_study_session(
        StudySession(
            session_id=session_id or str(uuid4()),
            document_ids=tuple(document_ids),
            status="active",
            created_at=now,
            updated_at=now,
            pending_question_id=None,
        )
    )


def create_study_question(
    *,
    session_id: str,
    question_type: StudyQuestionType,
    prompt: str,
    choices: dict[str, str] | None,
    correct_answer: str,
    explanation: str,
    source_refs: tuple[StudySourceRef, ...] | list[StudySourceRef],
    primary_source_ref: StudySourceRef,
    question_id: str | None = None,
    created_at: str | None = None,
) -> StudyQuestion:
    """Create a validated pending study question."""
    return validate_study_question(
        StudyQuestion(
            question_id=question_id or str(uuid4()),
            session_id=session_id,
            question_type=question_type,
            prompt=prompt,
            choices=choices,
            correct_answer=correct_answer,
            explanation=explanation,
            source_refs=tuple(source_refs),
            primary_source_ref=primary_source_ref,
            created_at=created_at or utc_timestamp(),
            status="pending",
        )
    )


def create_study_attempt(
    *,
    session_id: str,
    question_id: str,
    primary_chunk_id: str,
    result: StudyAttemptResult,
    attempt_id: str | None = None,
    submitted_at: str | None = None,
) -> StudyAttempt:
    """Create a validated study attempt record."""
    return validate_study_attempt(
        StudyAttempt(
            attempt_id=attempt_id or str(uuid4()),
            session_id=session_id,
            question_id=question_id,
            primary_chunk_id=primary_chunk_id,
            result=result,
            submitted_at=submitted_at or utc_timestamp(),
        )
    )


def create_study_chunk_stats(
    *,
    chunk_id: str,
    attempts: int = 0,
    correct: int = 0,
    partial: int = 0,
    incorrect: int = 0,
    last_attempt_at: str | None = None,
) -> StudyChunkStats:
    """Create validated chunk stats, defaulting last_attempt_at when needed."""
    return validate_study_chunk_stats(
        StudyChunkStats(
            chunk_id=chunk_id,
            attempts=attempts,
            correct=correct,
            partial=partial,
            incorrect=incorrect,
            last_attempt_at=last_attempt_at or utc_timestamp(),
        )
    )


def to_public_question_view(question: StudyQuestion) -> StudyQuestionPublicView:
    """Project a question into the safe pre-answer view."""
    return StudyQuestionPublicView(
        question_id=question.question_id,
        session_id=question.session_id,
        question_type=question.question_type,
        prompt=question.prompt,
        choices=None if question.choices is None else dict(question.choices),
    )


def validate_study_source_ref(ref: StudySourceRef) -> StudySourceRef:
    """Validate one stable study source reference."""
    try:
        document_id = validate_document_id(ref.document_id)
    except DocumentValidationError as error:
        raise StudyValidationError(str(error)) from error

    if not isinstance(ref.chunk_index, int) or isinstance(ref.chunk_index, bool):
        raise StudyValidationError("Chunk index must be an integer.")
    if ref.chunk_index < 0:
        raise StudyValidationError("Chunk index cannot be negative.")
    if not isinstance(ref.start_offset, int) or isinstance(ref.start_offset, bool):
        raise StudyValidationError("Chunk start offset must be an integer.")
    if not isinstance(ref.end_offset, int) or isinstance(ref.end_offset, bool):
        raise StudyValidationError("Chunk end offset must be an integer.")
    if ref.start_offset < 0:
        raise StudyValidationError("Chunk start offset cannot be negative.")
    if ref.end_offset < ref.start_offset:
        raise StudyValidationError(
            "Chunk end offset cannot be less than start offset."
        )

    expected_chunk_id = build_chunk_id(document_id, ref.chunk_index)
    if ref.chunk_id != expected_chunk_id:
        raise StudyValidationError(
            "Chunk ID does not match document ID and chunk index."
        )
    return StudySourceRef(
        document_id=document_id,
        chunk_id=expected_chunk_id,
        chunk_index=ref.chunk_index,
        start_offset=ref.start_offset,
        end_offset=ref.end_offset,
    )


def validate_study_session(session: StudySession) -> StudySession:
    """Validate all fields of a study session."""
    try:
        session_id = validate_uuid(session.session_id, field_name="Session ID")
    except ToolValidationError as error:
        raise StudyValidationError(str(error)) from error

    if not isinstance(session.document_ids, tuple):
        raise StudyValidationError("document_ids must be a tuple.")
    if not (1 <= len(session.document_ids) <= MAX_STUDY_DOCUMENTS):
        raise StudyValidationError(
            f"A study session requires 1 to {MAX_STUDY_DOCUMENTS} document IDs."
        )

    cleaned_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in session.document_ids:
        try:
            document_id = validate_document_id(raw_id)
        except DocumentValidationError as error:
            raise StudyValidationError(str(error)) from error
        if document_id in seen:
            raise StudyValidationError(
                "Study session document IDs must be unique."
            )
        seen.add(document_id)
        cleaned_ids.append(document_id)

    try:
        status = validate_controlled_value(
            session.status,
            allowed=STUDY_SESSION_STATUSES,
            field_name="Study session status",
        )
    except (BlankToolFieldError, InvalidToolEnumError) as error:
        raise StudyValidationError(str(error)) from error

    try:
        created_at = validate_utc_timestamp(
            session.created_at, field_name="created_at"
        )
        updated_at = validate_utc_timestamp(
            session.updated_at, field_name="updated_at"
        )
        pending = validate_optional_uuid(
            session.pending_question_id, field_name="Pending question ID"
        )
    except ToolValidationError as error:
        raise StudyValidationError(str(error)) from error

    return StudySession(
        session_id=session_id,
        document_ids=tuple(cleaned_ids),
        status=status,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=updated_at,
        pending_question_id=pending,
    )


def validate_study_question(question: StudyQuestion) -> StudyQuestion:
    """Validate all fields of a study question including the answer key."""
    try:
        question_id = validate_uuid(question.question_id, field_name="Question ID")
        session_id = validate_uuid(question.session_id, field_name="Session ID")
        question_type = validate_controlled_value(
            question.question_type,
            allowed=STUDY_QUESTION_TYPES,
            field_name="Question type",
        )
        status = validate_controlled_value(
            question.status,
            allowed=STUDY_QUESTION_STATUSES,
            field_name="Question status",
        )
        prompt = require_non_blank_text(
            question.prompt,
            field_name="Question prompt",
            max_length=MAX_STUDY_QUESTION_PROMPT_CHARS,
        )
        correct_answer = require_non_blank_text(
            question.correct_answer,
            field_name="Correct answer",
            max_length=MAX_STUDY_CORRECT_ANSWER_CHARS,
        )
        explanation = require_non_blank_text(
            question.explanation,
            field_name="Explanation",
            max_length=MAX_STUDY_EXPLANATION_CHARS,
        )
        created_at = validate_utc_timestamp(
            question.created_at, field_name="created_at"
        )
    except (
        BlankToolFieldError,
        InvalidToolEnumError,
        ToolFieldTooLongError,
        ToolValidationError,
    ) as error:
        raise StudyValidationError(str(error)) from error

    if not question.source_refs:
        raise StudyValidationError("A study question requires source references.")
    source_refs = tuple(validate_study_source_ref(ref) for ref in question.source_refs)
    primary = validate_study_source_ref(question.primary_source_ref)
    if primary not in source_refs:
        raise StudyValidationError(
            "primary_source_ref must be one of source_refs."
        )

    choices: dict[str, str] | None
    if question_type == "mcq":
        choices = _validate_mcq_choices(question.choices)
        if correct_answer not in MCQ_CHOICE_LABEL_SET:
            raise StudyValidationError(
                "MCQ correct_answer must be exactly A, B, C, or D."
            )
    else:
        if question.choices is not None:
            raise StudyValidationError(
                "Short-answer questions must have null choices."
            )
        choices = None

    return StudyQuestion(
        question_id=question_id,
        session_id=session_id,
        question_type=question_type,  # type: ignore[arg-type]
        prompt=prompt,
        choices=choices,
        correct_answer=correct_answer,
        explanation=explanation,
        source_refs=source_refs,
        primary_source_ref=primary,
        created_at=created_at,
        status=status,  # type: ignore[arg-type]
    )


def validate_study_attempt(attempt: StudyAttempt) -> StudyAttempt:
    """Validate one persisted study attempt."""
    try:
        attempt_id = validate_uuid(attempt.attempt_id, field_name="Attempt ID")
        session_id = validate_uuid(attempt.session_id, field_name="Session ID")
        question_id = validate_uuid(attempt.question_id, field_name="Question ID")
        result = validate_controlled_value(
            attempt.result,
            allowed=STUDY_ATTEMPT_RESULTS,
            field_name="Attempt result",
        )
        submitted_at = validate_utc_timestamp(
            attempt.submitted_at, field_name="submitted_at"
        )
        primary_chunk_id = require_non_blank_text(
            attempt.primary_chunk_id,
            field_name="Primary chunk ID",
            max_length=200,
        )
    except (
        BlankToolFieldError,
        InvalidToolEnumError,
        ToolFieldTooLongError,
        ToolValidationError,
    ) as error:
        raise StudyValidationError(str(error)) from error

    return StudyAttempt(
        attempt_id=attempt_id,
        session_id=session_id,
        question_id=question_id,
        primary_chunk_id=primary_chunk_id,
        result=result,  # type: ignore[arg-type]
        submitted_at=submitted_at,
    )


def validate_study_chunk_stats(stats: StudyChunkStats) -> StudyChunkStats:
    """Validate per-chunk study statistics."""
    try:
        chunk_id = require_non_blank_text(
            stats.chunk_id, field_name="Chunk ID", max_length=200
        )
        last_attempt_at = validate_utc_timestamp(
            stats.last_attempt_at, field_name="last_attempt_at"
        )
    except (BlankToolFieldError, ToolFieldTooLongError, ToolValidationError) as error:
        raise StudyValidationError(str(error)) from error

    for name, value in (
        ("attempts", stats.attempts),
        ("correct", stats.correct),
        ("partial", stats.partial),
        ("incorrect", stats.incorrect),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StudyValidationError(f"{name} must be a non-negative integer.")

    if stats.correct + stats.partial + stats.incorrect > stats.attempts:
        raise StudyValidationError(
            "Chunk result counts cannot exceed attempts."
        )

    return StudyChunkStats(
        chunk_id=chunk_id,
        attempts=stats.attempts,
        correct=stats.correct,
        partial=stats.partial,
        incorrect=stats.incorrect,
        last_attempt_at=last_attempt_at,
    )


def validate_user_answer_input(answer: str) -> str:
    """Validate user answer input length without persisting it."""
    try:
        return require_non_blank_text(
            answer,
            field_name="Study answer",
            max_length=MAX_STUDY_USER_ANSWER_CHARS,
        )
    except (BlankToolFieldError, ToolFieldTooLongError, ToolValidationError) as error:
        raise StudyValidationError(str(error)) from error


def validate_feedback_text(feedback: str) -> str:
    """Validate grader feedback text length."""
    try:
        return require_non_blank_text(
            feedback,
            field_name="Feedback",
            max_length=MAX_STUDY_FEEDBACK_CHARS,
        )
    except (BlankToolFieldError, ToolFieldTooLongError, ToolValidationError) as error:
        raise StudyValidationError(str(error)) from error


def _validate_mcq_choices(choices: dict[str, str] | None) -> dict[str, str]:
    if not isinstance(choices, dict):
        raise StudyValidationError("MCQ choices must be an object.")
    if set(choices.keys()) != MCQ_CHOICE_LABEL_SET:
        raise StudyValidationError(
            "MCQ choices must contain exactly keys A, B, C, and D."
        )

    cleaned: dict[str, str] = {}
    seen_texts: set[str] = set()
    for label in MCQ_CHOICE_LABELS:
        try:
            text = require_non_blank_text(
                choices[label],
                field_name=f"Choice {label}",
                max_length=MAX_STUDY_CHOICE_CHARS,
            )
        except (BlankToolFieldError, ToolFieldTooLongError, ToolValidationError) as error:
            raise StudyValidationError(str(error)) from error
        folded = text.casefold()
        if folded in seen_texts:
            raise StudyValidationError(
                "MCQ choices must not contain duplicate text."
            )
        seen_texts.add(folded)
        cleaned[label] = text
    return cleaned
