"""Study Partner repository abstractions and atomic JSON persistence.

One application instance should access a study repository file at a time.
Atomic writes protect against partial-write corruption within a single writer.
They do not provide full concurrent multi-process coordination.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.config import (
    MAX_STORED_STUDY_SESSIONS,
    MAX_STUDY_ATTEMPTS_PER_SESSION,
    MAX_STUDY_QUESTIONS_PER_SESSION,
    STUDY_REPOSITORY_SCHEMA_VERSION,
)
from src.study_models import (
    StudyAttempt,
    StudyAttemptResult,
    StudyChunkStats,
    StudyQuestion,
    StudySession,
    StudySourceRef,
    StudyValidationError,
    create_study_chunk_stats,
    validate_study_attempt,
    validate_study_chunk_stats,
    validate_study_question,
    validate_study_session,
    validate_study_source_ref,
)
from src.tool_common import utc_timestamp, validate_uuid

logger = logging.getLogger("ProjectCortana")

STUDY_LOAD_ERROR_MESSAGE = (
    "Cortana: Study Partner data could not be loaded safely. "
    "The existing study repository file was left unchanged for inspection."
)
STUDY_SAVE_ERROR_MESSAGE = (
    "Cortana: Study Partner data could not be updated safely."
)
STUDY_CAPACITY_ERROR_MESSAGE = (
    "Cortana: Study Partner storage limit reached. "
    "End and clear older completed sessions before continuing."
)

PERSISTED_ROOT_KEYS: frozenset[str] = frozenset(
    {"version", "sessions", "questions", "attempts", "chunk_stats"}
)
PERSISTED_SESSION_KEYS: frozenset[str] = frozenset(
    {
        "session_id",
        "document_ids",
        "status",
        "created_at",
        "updated_at",
        "pending_question_id",
    }
)
PERSISTED_SOURCE_REF_KEYS: frozenset[str] = frozenset(
    {
        "document_id",
        "chunk_id",
        "chunk_index",
        "start_offset",
        "end_offset",
    }
)
PERSISTED_QUESTION_KEYS: frozenset[str] = frozenset(
    {
        "question_id",
        "session_id",
        "question_type",
        "prompt",
        "choices",
        "correct_answer",
        "explanation",
        "source_refs",
        "primary_source_ref",
        "created_at",
        "status",
    }
)
PERSISTED_ATTEMPT_KEYS: frozenset[str] = frozenset(
    {
        "attempt_id",
        "session_id",
        "question_id",
        "primary_chunk_id",
        "result",
        "submitted_at",
    }
)
PERSISTED_CHUNK_STATS_KEYS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "attempts",
        "correct",
        "partial",
        "incorrect",
        "last_attempt_at",
    }
)


class StudyStorageError(RuntimeError):
    """Raised when study repository data cannot be loaded or saved safely."""

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or STUDY_LOAD_ERROR_MESSAGE


class StudyRepository(Protocol):
    """Persistence boundary for study sessions, questions, attempts, and stats."""

    def get_active_session(self) -> StudySession | None:
        """Return the at-most-one active session, if any."""

    def get_session(self, session_id: str) -> StudySession | None:
        """Return one session by ID."""

    def list_sessions(self) -> list[StudySession]:
        """Return sessions in storage order."""

    def get_question(self, question_id: str) -> StudyQuestion | None:
        """Return one question by ID."""

    def list_questions(self, *, session_id: str | None = None) -> list[StudyQuestion]:
        """Return questions in storage order, optionally filtered."""

    def list_attempts(self, *, session_id: str | None = None) -> list[StudyAttempt]:
        """Return attempts in storage order, optionally filtered."""

    def get_chunk_stats(self, chunk_id: str) -> StudyChunkStats | None:
        """Return stats for one chunk ID."""

    def list_chunk_stats(self) -> list[StudyChunkStats]:
        """Return all chunk stats in storage order."""

    def create_session(self, session: StudySession) -> StudySession:
        """Persist a new active session, pruning oldest completed if needed."""

    def complete_session(self, session_id: str) -> StudySession:
        """Mark one active session completed."""

    def add_pending_question(
        self,
        session: StudySession,
        question: StudyQuestion,
    ) -> StudyQuestion:
        """Atomically add a pending question and update the session pointer."""

    def record_graded_answer(
        self,
        *,
        session_id: str,
        question_id: str,
        attempt: StudyAttempt,
        result: StudyAttemptResult,
    ) -> StudyAttempt:
        """Atomically record attempt, mark question answered, clear pending, update stats."""


@dataclass
class _StudyRepositoryState:
    """In-memory snapshot of the study repository."""

    sessions: dict[str, StudySession] = field(default_factory=dict)
    session_order: list[str] = field(default_factory=list)
    questions: dict[str, StudyQuestion] = field(default_factory=dict)
    question_order: list[str] = field(default_factory=list)
    attempts: dict[str, StudyAttempt] = field(default_factory=dict)
    attempt_order: list[str] = field(default_factory=list)
    chunk_stats: dict[str, StudyChunkStats] = field(default_factory=dict)
    chunk_stats_order: list[str] = field(default_factory=list)


class JsonStudyRepository:
    """JSON-backed study store with atomic writes and sticky load errors."""

    def __init__(
        self,
        file_path: Path,
        *,
        max_sessions: int = MAX_STORED_STUDY_SESSIONS,
        max_questions_per_session: int = MAX_STUDY_QUESTIONS_PER_SESSION,
        max_attempts_per_session: int = MAX_STUDY_ATTEMPTS_PER_SESSION,
    ) -> None:
        if max_sessions < 1:
            raise StudyValidationError("max_sessions must be at least 1.")
        if max_questions_per_session < 1:
            raise StudyValidationError(
                "max_questions_per_session must be at least 1."
            )
        if max_attempts_per_session < 1:
            raise StudyValidationError(
                "max_attempts_per_session must be at least 1."
            )
        self._file_path = file_path
        self._max_sessions = max_sessions
        self._max_questions_per_session = max_questions_per_session
        self._max_attempts_per_session = max_attempts_per_session
        self._state: _StudyRepositoryState | None = None
        self._load_error: StudyStorageError | None = None
        self.atomic_replace_count = 0

    @property
    def file_path(self) -> Path:
        """Return the study repository JSON file path."""
        return self._file_path

    def get_active_session(self) -> StudySession | None:
        """Return the at-most-one active session, if any."""
        active = [
            session
            for session in self.list_sessions()
            if session.status == "active"
        ]
        if len(active) > 1:
            raise StudyStorageError(
                "Multiple active study sessions found.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        return active[0] if active else None

    def get_session(self, session_id: str) -> StudySession | None:
        """Return one session by ID."""
        try:
            cleaned = validate_uuid(session_id, field_name="Session ID")
        except Exception:
            return None
        return self._ensure_loaded().sessions.get(cleaned)

    def list_sessions(self) -> list[StudySession]:
        """Return sessions in storage order."""
        state = self._ensure_loaded()
        return [
            state.sessions[session_id]
            for session_id in state.session_order
            if session_id in state.sessions
        ]

    def get_question(self, question_id: str) -> StudyQuestion | None:
        """Return one question by ID."""
        try:
            cleaned = validate_uuid(question_id, field_name="Question ID")
        except Exception:
            return None
        return self._ensure_loaded().questions.get(cleaned)

    def list_questions(self, *, session_id: str | None = None) -> list[StudyQuestion]:
        """Return questions in storage order, optionally filtered."""
        state = self._ensure_loaded()
        questions = [
            state.questions[question_id]
            for question_id in state.question_order
            if question_id in state.questions
        ]
        if session_id is None:
            return questions
        try:
            cleaned = validate_uuid(session_id, field_name="Session ID")
        except Exception:
            return []
        return [item for item in questions if item.session_id == cleaned]

    def list_attempts(self, *, session_id: str | None = None) -> list[StudyAttempt]:
        """Return attempts in storage order, optionally filtered."""
        state = self._ensure_loaded()
        attempts = [
            state.attempts[attempt_id]
            for attempt_id in state.attempt_order
            if attempt_id in state.attempts
        ]
        if session_id is None:
            return attempts
        try:
            cleaned = validate_uuid(session_id, field_name="Session ID")
        except Exception:
            return []
        return [item for item in attempts if item.session_id == cleaned]

    def get_chunk_stats(self, chunk_id: str) -> StudyChunkStats | None:
        """Return stats for one chunk ID."""
        cleaned = chunk_id.strip()
        if not cleaned:
            return None
        return self._ensure_loaded().chunk_stats.get(cleaned)

    def list_chunk_stats(self) -> list[StudyChunkStats]:
        """Return all chunk stats in storage order."""
        state = self._ensure_loaded()
        return [
            state.chunk_stats[chunk_id]
            for chunk_id in state.chunk_stats_order
            if chunk_id in state.chunk_stats
        ]

    def create_session(self, session: StudySession) -> StudySession:
        """Persist a new active session, pruning oldest completed if needed."""
        state = self._clone_state(self._ensure_loaded())
        validated = validate_study_session(session)
        if validated.status != "active":
            raise StudyStorageError(
                "New study sessions must start as active.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )
        if any(item.status == "active" for item in state.sessions.values()):
            raise StudyStorageError(
                "An active study session already exists.",
                user_message=(
                    "Cortana: A study session is already active. "
                    "Use /study-end before starting another."
                ),
            )
        if validated.session_id in state.sessions:
            raise StudyStorageError(
                "Duplicate study session ID rejected.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )

        while len(state.session_order) >= self._max_sessions:
            if not self._prune_oldest_completed(state):
                raise StudyStorageError(
                    "Study session capacity reached.",
                    user_message=STUDY_CAPACITY_ERROR_MESSAGE,
                )

        state.sessions[validated.session_id] = validated
        state.session_order.append(validated.session_id)
        self._persist(state)
        return validated

    def complete_session(self, session_id: str) -> StudySession:
        """Mark one active session completed."""
        state = self._clone_state(self._ensure_loaded())
        try:
            cleaned = validate_uuid(session_id, field_name="Session ID")
        except Exception as error:
            raise StudyStorageError(
                "Invalid study session ID.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            ) from error

        existing = state.sessions.get(cleaned)
        if existing is None:
            raise StudyStorageError(
                "Study session not found.",
                user_message="Cortana: No study session was found.",
            )
        if existing.status != "active":
            raise StudyStorageError(
                "Study session is not active.",
                user_message="Cortana: There is no active study session.",
            )

        completed = validate_study_session(
            StudySession(
                session_id=existing.session_id,
                document_ids=existing.document_ids,
                status="completed",
                created_at=existing.created_at,
                updated_at=utc_timestamp(),
                pending_question_id=existing.pending_question_id,
            )
        )
        state.sessions[completed.session_id] = completed
        self._persist(state)
        return completed

    def add_pending_question(
        self,
        session: StudySession,
        question: StudyQuestion,
    ) -> StudyQuestion:
        """Atomically add a pending question and update the session pointer."""
        state = self._clone_state(self._ensure_loaded())
        validated_session = validate_study_session(session)
        validated_question = validate_study_question(question)

        existing_session = state.sessions.get(validated_session.session_id)
        if existing_session is None or existing_session.status != "active":
            raise StudyStorageError(
                "Active study session required.",
                user_message="Cortana: There is no active study session.",
            )
        if existing_session.pending_question_id is not None:
            raise StudyStorageError(
                "Pending question already exists.",
                user_message=(
                    "Cortana: Answer the current study question before "
                    "generating another."
                ),
            )
        if validated_question.session_id != existing_session.session_id:
            raise StudyStorageError(
                "Question session mismatch.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )
        if validated_question.status != "pending":
            raise StudyStorageError(
                "New questions must be pending.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )
        if validated_question.question_id in state.questions:
            raise StudyStorageError(
                "Duplicate question ID rejected.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )

        session_question_count = sum(
            1
            for item in state.questions.values()
            if item.session_id == existing_session.session_id
        )
        if session_question_count >= self._max_questions_per_session:
            raise StudyStorageError(
                "Study question capacity reached.",
                user_message=STUDY_CAPACITY_ERROR_MESSAGE,
            )

        updated_session = validate_study_session(
            StudySession(
                session_id=existing_session.session_id,
                document_ids=existing_session.document_ids,
                status="active",
                created_at=existing_session.created_at,
                updated_at=utc_timestamp(),
                pending_question_id=validated_question.question_id,
            )
        )
        state.sessions[updated_session.session_id] = updated_session
        state.questions[validated_question.question_id] = validated_question
        state.question_order.append(validated_question.question_id)
        self._persist(state)
        return validated_question

    def record_graded_answer(
        self,
        *,
        session_id: str,
        question_id: str,
        attempt: StudyAttempt,
        result: StudyAttemptResult,
    ) -> StudyAttempt:
        """Atomically record attempt, mark question answered, clear pending, update stats."""
        state = self._clone_state(self._ensure_loaded())
        validated_attempt = validate_study_attempt(attempt)
        if validated_attempt.result != result:
            raise StudyStorageError(
                "Attempt result mismatch.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )

        try:
            cleaned_session_id = validate_uuid(session_id, field_name="Session ID")
            cleaned_question_id = validate_uuid(question_id, field_name="Question ID")
        except Exception as error:
            raise StudyStorageError(
                "Invalid study identifiers.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            ) from error

        existing_session = state.sessions.get(cleaned_session_id)
        existing_question = state.questions.get(cleaned_question_id)
        if existing_session is None or existing_session.status != "active":
            raise StudyStorageError(
                "Active study session required for grading.",
                user_message="Cortana: There is no active study session.",
            )
        if existing_question is None:
            raise StudyStorageError(
                "Study question not found.",
                user_message="Cortana: No pending study question was found.",
            )
        if existing_session.pending_question_id != existing_question.question_id:
            raise StudyStorageError(
                "Question is not the pending study question.",
                user_message="Cortana: No pending study question was found.",
            )
        if existing_question.status != "pending":
            raise StudyStorageError(
                "Question is not pending.",
                user_message="Cortana: No pending study question was found.",
            )
        if validated_attempt.session_id != existing_session.session_id:
            raise StudyStorageError(
                "Attempt session mismatch.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )
        if validated_attempt.question_id != existing_question.question_id:
            raise StudyStorageError(
                "Attempt question mismatch.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )
        if (
            validated_attempt.primary_chunk_id
            != existing_question.primary_source_ref.chunk_id
        ):
            raise StudyStorageError(
                "Attempt primary chunk mismatch.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )
        if validated_attempt.attempt_id in state.attempts:
            raise StudyStorageError(
                "Duplicate attempt ID rejected.",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            )

        session_attempt_count = sum(
            1
            for item in state.attempts.values()
            if item.session_id == existing_session.session_id
        )
        if session_attempt_count >= self._max_attempts_per_session:
            raise StudyStorageError(
                "Study attempt capacity reached.",
                user_message=STUDY_CAPACITY_ERROR_MESSAGE,
            )

        answered_question = validate_study_question(
            StudyQuestion(
                question_id=existing_question.question_id,
                session_id=existing_question.session_id,
                question_type=existing_question.question_type,
                prompt=existing_question.prompt,
                choices=existing_question.choices,
                correct_answer=existing_question.correct_answer,
                explanation=existing_question.explanation,
                source_refs=existing_question.source_refs,
                primary_source_ref=existing_question.primary_source_ref,
                created_at=existing_question.created_at,
                status="answered",
            )
        )
        updated_session = validate_study_session(
            StudySession(
                session_id=existing_session.session_id,
                document_ids=existing_session.document_ids,
                status="active",
                created_at=existing_session.created_at,
                updated_at=utc_timestamp(),
                pending_question_id=None,
            )
        )
        updated_stats = self._apply_result_to_stats(
            state,
            chunk_id=existing_question.primary_source_ref.chunk_id,
            result=result,
            submitted_at=validated_attempt.submitted_at,
        )

        state.sessions[updated_session.session_id] = updated_session
        state.questions[answered_question.question_id] = answered_question
        state.attempts[validated_attempt.attempt_id] = validated_attempt
        state.attempt_order.append(validated_attempt.attempt_id)
        state.chunk_stats[updated_stats.chunk_id] = updated_stats
        if updated_stats.chunk_id not in state.chunk_stats_order:
            state.chunk_stats_order.append(updated_stats.chunk_id)

        self._persist(state)
        return validated_attempt

    def _apply_result_to_stats(
        self,
        state: _StudyRepositoryState,
        *,
        chunk_id: str,
        result: StudyAttemptResult,
        submitted_at: str,
    ) -> StudyChunkStats:
        existing = state.chunk_stats.get(chunk_id)
        if existing is None:
            existing = create_study_chunk_stats(
                chunk_id=chunk_id,
                attempts=0,
                correct=0,
                partial=0,
                incorrect=0,
                last_attempt_at=submitted_at,
            )
        correct = existing.correct + (1 if result == "correct" else 0)
        partial = existing.partial + (1 if result == "partially_correct" else 0)
        incorrect = existing.incorrect + (1 if result == "incorrect" else 0)
        return validate_study_chunk_stats(
            StudyChunkStats(
                chunk_id=chunk_id,
                attempts=existing.attempts + 1,
                correct=correct,
                partial=partial,
                incorrect=incorrect,
                last_attempt_at=submitted_at,
            )
        )

    def _prune_oldest_completed(self, state: _StudyRepositoryState) -> bool:
        for session_id in list(state.session_order):
            session = state.sessions.get(session_id)
            if session is None or session.status != "completed":
                continue
            self._remove_session_bundle(state, session_id)
            return True
        return False

    def _remove_session_bundle(
        self,
        state: _StudyRepositoryState,
        session_id: str,
    ) -> None:
        question_ids = [
            question_id
            for question_id, question in state.questions.items()
            if question.session_id == session_id
        ]
        attempt_ids = [
            attempt_id
            for attempt_id, attempt in state.attempts.items()
            if attempt.session_id == session_id
        ]
        for question_id in question_ids:
            del state.questions[question_id]
        state.question_order = [
            item for item in state.question_order if item not in set(question_ids)
        ]
        for attempt_id in attempt_ids:
            del state.attempts[attempt_id]
        state.attempt_order = [
            item for item in state.attempt_order if item not in set(attempt_ids)
        ]
        del state.sessions[session_id]
        state.session_order = [
            item for item in state.session_order if item != session_id
        ]

    def _ensure_loaded(self) -> _StudyRepositoryState:
        if self._load_error is not None:
            raise self._load_error
        if self._state is not None:
            return self._state

        if not self._file_path.exists():
            self._state = _StudyRepositoryState()
            return self._state

        try:
            raw_text = self._file_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                raise StudyStorageError(
                    "Study repository file is empty.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise StudyStorageError(
                    "Study repository root must be an object.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            self._state = self._deserialize_state(payload)
            return self._state
        except StudyStorageError as error:
            self._load_error = StudyStorageError(
                str(error),
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
            raise self._load_error from error
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            StudyValidationError,
        ) as error:
            logger.error(
                "Study repository load failed with error_type=%s",
                type(error).__name__,
            )
            self._load_error = StudyStorageError(
                f"Study repository load failed: {type(error).__name__}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
            raise self._load_error from None

    def _deserialize_state(self, payload: dict[str, object]) -> _StudyRepositoryState:
        unexpected_root = set(payload) - PERSISTED_ROOT_KEYS
        if unexpected_root:
            raise StudyStorageError(
                f"Unexpected study repository keys: {sorted(unexpected_root)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        missing_root = PERSISTED_ROOT_KEYS - set(payload)
        if missing_root:
            raise StudyStorageError(
                f"Missing study repository keys: {sorted(missing_root)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )

        version = payload.get("version")
        if version != STUDY_REPOSITORY_SCHEMA_VERSION:
            raise StudyStorageError(
                f"Unsupported study repository schema version: {version!r}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )

        sessions_payload = payload.get("sessions")
        questions_payload = payload.get("questions")
        attempts_payload = payload.get("attempts")
        stats_payload = payload.get("chunk_stats")
        if not all(
            isinstance(item, list)
            for item in (
                sessions_payload,
                questions_payload,
                attempts_payload,
                stats_payload,
            )
        ):
            raise StudyStorageError(
                "Study repository collections must be lists.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )

        state = _StudyRepositoryState()
        assert isinstance(sessions_payload, list)
        assert isinstance(questions_payload, list)
        assert isinstance(attempts_payload, list)
        assert isinstance(stats_payload, list)

        active_count = 0
        for item in sessions_payload:
            session = self._deserialize_session(item)
            if session.session_id in state.sessions:
                raise StudyStorageError(
                    "Duplicate study session ID in repository file.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            if session.status == "active":
                active_count += 1
            state.sessions[session.session_id] = session
            state.session_order.append(session.session_id)
        if active_count > 1:
            raise StudyStorageError(
                "Multiple active study sessions in repository file.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )

        for item in questions_payload:
            question = self._deserialize_question(item)
            if question.question_id in state.questions:
                raise StudyStorageError(
                    "Duplicate study question ID in repository file.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            if question.session_id not in state.sessions:
                raise StudyStorageError(
                    "Study question references missing session.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            state.questions[question.question_id] = question
            state.question_order.append(question.question_id)

        for item in attempts_payload:
            attempt = self._deserialize_attempt(item)
            if attempt.attempt_id in state.attempts:
                raise StudyStorageError(
                    "Duplicate study attempt ID in repository file.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            if attempt.session_id not in state.sessions:
                raise StudyStorageError(
                    "Study attempt references missing session.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            if attempt.question_id not in state.questions:
                raise StudyStorageError(
                    "Study attempt references missing question.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            state.attempts[attempt.attempt_id] = attempt
            state.attempt_order.append(attempt.attempt_id)

        for item in stats_payload:
            stats = self._deserialize_chunk_stats(item)
            if stats.chunk_id in state.chunk_stats:
                raise StudyStorageError(
                    "Duplicate chunk stats ID in repository file.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            state.chunk_stats[stats.chunk_id] = stats
            state.chunk_stats_order.append(stats.chunk_id)

        for session in state.sessions.values():
            if session.pending_question_id is None:
                continue
            pending = state.questions.get(session.pending_question_id)
            if pending is None or pending.session_id != session.session_id:
                raise StudyStorageError(
                    "Session pending_question_id is inconsistent.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
            if session.status == "active" and pending.status != "pending":
                raise StudyStorageError(
                    "Active session pending question is not pending.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )

        return state

    def _deserialize_session(self, item: object) -> StudySession:
        if not isinstance(item, dict):
            raise StudyStorageError(
                "Study session must be an object.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_SESSION_KEYS
        if unexpected:
            raise StudyStorageError(
                f"Unexpected study session keys: {sorted(unexpected)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_SESSION_KEYS - set(item)
        if missing:
            raise StudyStorageError(
                f"Missing study session keys: {sorted(missing)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        document_ids_raw = item["document_ids"]
        if not isinstance(document_ids_raw, list) or not all(
            isinstance(value, str) for value in document_ids_raw
        ):
            raise StudyStorageError(
                "document_ids must be a list of strings.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        pending = item["pending_question_id"]
        if pending is not None and not isinstance(pending, str):
            raise StudyStorageError(
                "pending_question_id must be a string or null.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        for key in ("session_id", "status", "created_at", "updated_at"):
            if not isinstance(item[key], str):
                raise StudyStorageError(
                    f"{key} must be a string.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
        return validate_study_session(
            StudySession(
                session_id=str(item["session_id"]),
                document_ids=tuple(str(value) for value in document_ids_raw),
                status=str(item["status"]),  # type: ignore[arg-type]
                created_at=str(item["created_at"]),
                updated_at=str(item["updated_at"]),
                pending_question_id=pending,
            )
        )

    def _deserialize_source_ref(self, item: object) -> StudySourceRef:
        if not isinstance(item, dict):
            raise StudyStorageError(
                "Study source ref must be an object.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_SOURCE_REF_KEYS
        if unexpected:
            raise StudyStorageError(
                f"Unexpected study source ref keys: {sorted(unexpected)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_SOURCE_REF_KEYS - set(item)
        if missing:
            raise StudyStorageError(
                f"Missing study source ref keys: {sorted(missing)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        for key in ("document_id", "chunk_id"):
            if not isinstance(item[key], str):
                raise StudyStorageError(
                    f"{key} must be a string.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
        for key in ("chunk_index", "start_offset", "end_offset"):
            value = item[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise StudyStorageError(
                    f"{key} must be an integer.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
        return validate_study_source_ref(
            StudySourceRef(
                document_id=str(item["document_id"]),
                chunk_id=str(item["chunk_id"]),
                chunk_index=int(item["chunk_index"]),
                start_offset=int(item["start_offset"]),
                end_offset=int(item["end_offset"]),
            )
        )

    def _deserialize_question(self, item: object) -> StudyQuestion:
        if not isinstance(item, dict):
            raise StudyStorageError(
                "Study question must be an object.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_QUESTION_KEYS
        if unexpected:
            raise StudyStorageError(
                f"Unexpected study question keys: {sorted(unexpected)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_QUESTION_KEYS - set(item)
        if missing:
            raise StudyStorageError(
                f"Missing study question keys: {sorted(missing)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )

        choices_raw = item["choices"]
        choices: dict[str, str] | None
        if choices_raw is None:
            choices = None
        elif isinstance(choices_raw, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in choices_raw.items()
        ):
            choices = {str(key): str(value) for key, value in choices_raw.items()}
        else:
            raise StudyStorageError(
                "choices must be an object of strings or null.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )

        source_refs_raw = item["source_refs"]
        if not isinstance(source_refs_raw, list):
            raise StudyStorageError(
                "source_refs must be a list.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        source_refs = tuple(
            self._deserialize_source_ref(ref) for ref in source_refs_raw
        )
        primary = self._deserialize_source_ref(item["primary_source_ref"])

        for key in (
            "question_id",
            "session_id",
            "question_type",
            "prompt",
            "correct_answer",
            "explanation",
            "created_at",
            "status",
        ):
            if not isinstance(item[key], str):
                raise StudyStorageError(
                    f"{key} must be a string.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )

        return validate_study_question(
            StudyQuestion(
                question_id=str(item["question_id"]),
                session_id=str(item["session_id"]),
                question_type=str(item["question_type"]),  # type: ignore[arg-type]
                prompt=str(item["prompt"]),
                choices=choices,
                correct_answer=str(item["correct_answer"]),
                explanation=str(item["explanation"]),
                source_refs=source_refs,
                primary_source_ref=primary,
                created_at=str(item["created_at"]),
                status=str(item["status"]),  # type: ignore[arg-type]
            )
        )

    def _deserialize_attempt(self, item: object) -> StudyAttempt:
        if not isinstance(item, dict):
            raise StudyStorageError(
                "Study attempt must be an object.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_ATTEMPT_KEYS
        if unexpected:
            raise StudyStorageError(
                f"Unexpected study attempt keys: {sorted(unexpected)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_ATTEMPT_KEYS - set(item)
        if missing:
            raise StudyStorageError(
                f"Missing study attempt keys: {sorted(missing)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        for key in (
            "attempt_id",
            "session_id",
            "question_id",
            "primary_chunk_id",
            "result",
            "submitted_at",
        ):
            if not isinstance(item[key], str):
                raise StudyStorageError(
                    f"{key} must be a string.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
        return validate_study_attempt(
            StudyAttempt(
                attempt_id=str(item["attempt_id"]),
                session_id=str(item["session_id"]),
                question_id=str(item["question_id"]),
                primary_chunk_id=str(item["primary_chunk_id"]),
                result=str(item["result"]),  # type: ignore[arg-type]
                submitted_at=str(item["submitted_at"]),
            )
        )

    def _deserialize_chunk_stats(self, item: object) -> StudyChunkStats:
        if not isinstance(item, dict):
            raise StudyStorageError(
                "Chunk stats must be an object.",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        unexpected = set(item) - PERSISTED_CHUNK_STATS_KEYS
        if unexpected:
            raise StudyStorageError(
                f"Unexpected chunk stats keys: {sorted(unexpected)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        missing = PERSISTED_CHUNK_STATS_KEYS - set(item)
        if missing:
            raise StudyStorageError(
                f"Missing chunk stats keys: {sorted(missing)}",
                user_message=STUDY_LOAD_ERROR_MESSAGE,
            )
        for key in ("chunk_id", "last_attempt_at"):
            if not isinstance(item[key], str):
                raise StudyStorageError(
                    f"{key} must be a string.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
        for key in ("attempts", "correct", "partial", "incorrect"):
            value = item[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise StudyStorageError(
                    f"{key} must be an integer.",
                    user_message=STUDY_LOAD_ERROR_MESSAGE,
                )
        return validate_study_chunk_stats(
            StudyChunkStats(
                chunk_id=str(item["chunk_id"]),
                attempts=int(item["attempts"]),
                correct=int(item["correct"]),
                partial=int(item["partial"]),
                incorrect=int(item["incorrect"]),
                last_attempt_at=str(item["last_attempt_at"]),
            )
        )

    def _clone_state(self, state: _StudyRepositoryState) -> _StudyRepositoryState:
        return _StudyRepositoryState(
            sessions=dict(state.sessions),
            session_order=list(state.session_order),
            questions=dict(state.questions),
            question_order=list(state.question_order),
            attempts=dict(state.attempts),
            attempt_order=list(state.attempt_order),
            chunk_stats=dict(state.chunk_stats),
            chunk_stats_order=list(state.chunk_stats_order),
        )

    def _persist(self, state: _StudyRepositoryState) -> None:
        payload = {
            "version": STUDY_REPOSITORY_SCHEMA_VERSION,
            "sessions": [
                self._serialize_session(state.sessions[session_id])
                for session_id in state.session_order
                if session_id in state.sessions
            ],
            "questions": [
                self._serialize_question(state.questions[question_id])
                for question_id in state.question_order
                if question_id in state.questions
            ],
            "attempts": [
                self._serialize_attempt(state.attempts[attempt_id])
                for attempt_id in state.attempt_order
                if attempt_id in state.attempts
            ],
            "chunk_stats": [
                self._serialize_chunk_stats(state.chunk_stats[chunk_id])
                for chunk_id in state.chunk_stats_order
                if chunk_id in state.chunk_stats
            ],
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        self._atomic_write(serialized)
        self._state = state

    def _serialize_source_ref(self, ref: StudySourceRef) -> dict[str, object]:
        return {
            "document_id": ref.document_id,
            "chunk_id": ref.chunk_id,
            "chunk_index": ref.chunk_index,
            "start_offset": ref.start_offset,
            "end_offset": ref.end_offset,
        }

    def _serialize_session(self, session: StudySession) -> dict[str, object]:
        return {
            "session_id": session.session_id,
            "document_ids": list(session.document_ids),
            "status": session.status,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "pending_question_id": session.pending_question_id,
        }

    def _serialize_question(self, question: StudyQuestion) -> dict[str, object]:
        return {
            "question_id": question.question_id,
            "session_id": question.session_id,
            "question_type": question.question_type,
            "prompt": question.prompt,
            "choices": None if question.choices is None else dict(question.choices),
            "correct_answer": question.correct_answer,
            "explanation": question.explanation,
            "source_refs": [
                self._serialize_source_ref(ref) for ref in question.source_refs
            ],
            "primary_source_ref": self._serialize_source_ref(
                question.primary_source_ref
            ),
            "created_at": question.created_at,
            "status": question.status,
        }

    def _serialize_attempt(self, attempt: StudyAttempt) -> dict[str, object]:
        return {
            "attempt_id": attempt.attempt_id,
            "session_id": attempt.session_id,
            "question_id": attempt.question_id,
            "primary_chunk_id": attempt.primary_chunk_id,
            "result": attempt.result,
            "submitted_at": attempt.submitted_at,
        }

    def _serialize_chunk_stats(self, stats: StudyChunkStats) -> dict[str, object]:
        return {
            "chunk_id": stats.chunk_id,
            "attempts": stats.attempts,
            "correct": stats.correct,
            "partial": stats.partial,
            "incorrect": stats.incorrect,
            "last_attempt_at": stats.last_attempt_at,
        }

    def _atomic_write(self, content: str) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._file_path.parent,
                prefix=".study-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._file_path)
            self.atomic_replace_count += 1
            temporary_path = None
        except OSError as error:
            logger.error(
                "Study repository save failed with error_type=%s",
                type(error).__name__,
            )
            raise StudyStorageError(
                f"Study repository save failed: {type(error).__name__}",
                user_message=STUDY_SAVE_ERROR_MESSAGE,
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
