"""Tests for Milestone 22 study repository persistence and atomicity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.document_chunk import build_chunk_id
from src.study_models import (
    StudySourceRef,
    create_study_attempt,
    create_study_question,
    create_study_session,
)
from src.study_repository import JsonStudyRepository, StudyStorageError
from src.tool_common import utc_timestamp


DOC_A = "11111111-1111-1111-1111-111111111111"
DOC_B = "22222222-2222-2222-2222-222222222222"
SESSION = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
QUESTION = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _ref(document_id: str = DOC_A, index: int = 0) -> StudySourceRef:
    return StudySourceRef(
        document_id=document_id,
        chunk_id=build_chunk_id(document_id, index),
        chunk_index=index,
        start_offset=0,
        end_offset=12,
    )


def test_root_schema_has_no_active_session_id(tmp_path: Path) -> None:
    repo = JsonStudyRepository(tmp_path / "study_state.json")
    session = create_study_session(
        document_ids=(DOC_A,),
        session_id=SESSION,
        created_at=utc_timestamp(),
    )
    repo.create_session(session)
    payload = json.loads((tmp_path / "study_state.json").read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "version",
        "sessions",
        "questions",
        "attempts",
        "chunk_stats",
    }
    assert "active_session_id" not in payload
    assert repo.get_active_session() is not None


def test_only_one_active_session_allowed(tmp_path: Path) -> None:
    repo = JsonStudyRepository(tmp_path / "study_state.json")
    repo.create_session(create_study_session(document_ids=(DOC_A,), session_id=SESSION))
    with pytest.raises(StudyStorageError):
        repo.create_session(
            create_study_session(
                document_ids=(DOC_B,),
                session_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            )
        )


def test_record_graded_answer_is_one_atomic_persist(tmp_path: Path) -> None:
    repo = JsonStudyRepository(tmp_path / "study_state.json")
    session = repo.create_session(
        create_study_session(document_ids=(DOC_A,), session_id=SESSION)
    )
    ref = _ref()
    question = create_study_question(
        session_id=SESSION,
        question_type="mcq",
        prompt="Prompt",
        choices={"A": "one", "B": "two", "C": "three", "D": "four"},
        correct_answer="A",
        explanation="Because source.",
        source_refs=(ref,),
        primary_source_ref=ref,
        question_id=QUESTION,
    )
    repo.add_pending_question(session, question)
    before = repo.atomic_replace_count
    attempt = create_study_attempt(
        session_id=SESSION,
        question_id=QUESTION,
        primary_chunk_id=ref.chunk_id,
        result="correct",
    )
    repo.record_graded_answer(
        session_id=SESSION,
        question_id=QUESTION,
        attempt=attempt,
        result="correct",
    )
    assert repo.atomic_replace_count == before + 1
    updated = repo.get_active_session()
    assert updated is not None
    assert updated.pending_question_id is None
    stored_question = repo.get_question(QUESTION)
    assert stored_question is not None
    assert stored_question.status == "answered"
    stats = repo.get_chunk_stats(ref.chunk_id)
    assert stats is not None
    assert stats.correct == 1
    assert stats.attempts == 1


def test_complete_session_keeps_pending_question_historical(tmp_path: Path) -> None:
    repo = JsonStudyRepository(tmp_path / "study_state.json")
    session = repo.create_session(
        create_study_session(document_ids=(DOC_A,), session_id=SESSION)
    )
    ref = _ref()
    question = create_study_question(
        session_id=SESSION,
        question_type="short",
        prompt="Prompt",
        choices=None,
        correct_answer="answer",
        explanation="explanation",
        source_refs=(ref,),
        primary_source_ref=ref,
        question_id=QUESTION,
    )
    repo.add_pending_question(session, question)
    completed = repo.complete_session(SESSION)
    assert completed.status == "completed"
    assert completed.pending_question_id == QUESTION
    assert repo.get_active_session() is None
    with pytest.raises(StudyStorageError):
        repo.record_graded_answer(
            session_id=SESSION,
            question_id=QUESTION,
            attempt=create_study_attempt(
                session_id=SESSION,
                question_id=QUESTION,
                primary_chunk_id=ref.chunk_id,
                result="correct",
            ),
            result="correct",
        )


def test_persistence_failure_leaves_no_partial_state(tmp_path: Path) -> None:
    path = tmp_path / "study_state.json"
    repo = JsonStudyRepository(path)
    session = repo.create_session(
        create_study_session(document_ids=(DOC_A,), session_id=SESSION)
    )
    ref = _ref()
    question = create_study_question(
        session_id=SESSION,
        question_type="mcq",
        prompt="Prompt",
        choices={"A": "one", "B": "two", "C": "three", "D": "four"},
        correct_answer="A",
        explanation="Because source.",
        source_refs=(ref,),
        primary_source_ref=ref,
        question_id=QUESTION,
    )
    repo.add_pending_question(session, question)

    def boom(_content: str) -> None:
        raise StudyStorageError(
            "forced persistence failure",
            user_message="Cortana: Study Partner data could not be updated safely.",
        )

    repo._atomic_write = boom  # type: ignore[method-assign]
    with pytest.raises(StudyStorageError):
        repo.record_graded_answer(
            session_id=SESSION,
            question_id=QUESTION,
            attempt=create_study_attempt(
                session_id=SESSION,
                question_id=QUESTION,
                primary_chunk_id=ref.chunk_id,
                result="incorrect",
            ),
            result="incorrect",
        )

    reloaded = JsonStudyRepository(path)
    active = reloaded.get_active_session()
    assert active is not None
    assert active.pending_question_id == QUESTION
    assert reloaded.get_question(QUESTION) is not None
    assert reloaded.get_question(QUESTION).status == "pending"  # type: ignore[union-attr]
    assert reloaded.list_attempts(session_id=SESSION) == []
    assert reloaded.get_chunk_stats(ref.chunk_id) is None
