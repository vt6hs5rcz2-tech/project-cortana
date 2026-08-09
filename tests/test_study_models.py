"""Tests for Milestone 22 study domain models and answer-key projections."""

from __future__ import annotations

from dataclasses import fields

import pytest

from src.document_chunk import build_chunk_id
from src.study_models import (
    StudyQuestionPublicView,
    StudySourceRef,
    StudyValidationError,
    create_study_attempt,
    create_study_question,
    create_study_session,
    to_public_question_view,
    validate_study_source_ref,
)
from src.tool_common import utc_timestamp


def _ref(document_id: str = "11111111-1111-1111-1111-111111111111") -> StudySourceRef:
    return StudySourceRef(
        document_id=document_id,
        chunk_id=build_chunk_id(document_id, 0),
        chunk_index=0,
        start_offset=0,
        end_offset=10,
    )


def test_study_session_requires_unique_bounded_document_ids() -> None:
    doc_a = "11111111-1111-1111-1111-111111111111"
    doc_b = "22222222-2222-2222-2222-222222222222"
    session = create_study_session(document_ids=(doc_a, doc_b))
    assert session.status == "active"
    assert session.pending_question_id is None
    assert session.document_ids == (doc_a, doc_b)

    with pytest.raises(StudyValidationError):
        create_study_session(document_ids=(doc_a, doc_a))


def test_study_question_public_view_excludes_answer_key() -> None:
    ref = _ref()
    question = create_study_question(
        session_id="33333333-3333-3333-3333-333333333333",
        question_type="mcq",
        prompt="What is containment?",
        choices={"A": "isolate", "B": "ignore", "C": "reboot", "D": "delete"},
        correct_answer="A",
        explanation="Sources require isolation.",
        source_refs=(ref,),
        primary_source_ref=ref,
    )
    public = to_public_question_view(question)
    assert isinstance(public, StudyQuestionPublicView)
    assert public.prompt == question.prompt
    assert public.choices == question.choices
    names = {item.name for item in fields(StudyQuestionPublicView)}
    assert "correct_answer" not in names
    assert "explanation" not in names
    rendered = repr(question)
    assert "Sources require isolation." not in rendered
    assert "correct_answer" not in rendered


def test_mcq_rejects_duplicate_choice_text() -> None:
    ref = _ref()
    with pytest.raises(StudyValidationError):
        create_study_question(
            session_id="33333333-3333-3333-3333-333333333333",
            question_type="mcq",
            prompt="Prompt",
            choices={"A": "Same", "B": "same", "C": "other", "D": "else"},
            correct_answer="A",
            explanation="Because source.",
            source_refs=(ref,),
            primary_source_ref=ref,
        )


def test_short_question_requires_null_choices() -> None:
    ref = _ref()
    question = create_study_question(
        session_id="33333333-3333-3333-3333-333333333333",
        question_type="short",
        prompt="Define isolation.",
        choices=None,
        correct_answer="Containment isolation",
        explanation="Source says isolate the host.",
        source_refs=(ref,),
        primary_source_ref=ref,
    )
    assert question.choices is None

    with pytest.raises(StudyValidationError):
        create_study_question(
            session_id="33333333-3333-3333-3333-333333333333",
            question_type="short",
            prompt="Define isolation.",
            choices={"A": "x", "B": "y", "C": "z", "D": "w"},
            correct_answer="Containment isolation",
            explanation="Source says isolate the host.",
            source_refs=(ref,),
            primary_source_ref=ref,
        )


def test_primary_source_ref_must_be_in_source_refs() -> None:
    primary = _ref()
    other = StudySourceRef(
        document_id="22222222-2222-2222-2222-222222222222",
        chunk_id=build_chunk_id("22222222-2222-2222-2222-222222222222", 0),
        chunk_index=0,
        start_offset=0,
        end_offset=5,
    )
    with pytest.raises(StudyValidationError):
        create_study_question(
            session_id="33333333-3333-3333-3333-333333333333",
            question_type="short",
            prompt="Prompt",
            choices=None,
            correct_answer="answer",
            explanation="explanation",
            source_refs=(other,),
            primary_source_ref=primary,
        )


def test_attempt_does_not_store_user_answer_fields() -> None:
    attempt = create_study_attempt(
        session_id="33333333-3333-3333-3333-333333333333",
        question_id="44444444-4444-4444-4444-444444444444",
        primary_chunk_id=build_chunk_id("11111111-1111-1111-1111-111111111111", 0),
        result="correct",
        submitted_at=utc_timestamp(),
    )
    names = set(attempt.__dataclass_fields__)
    assert "submitted_answer" not in names
    assert "feedback" not in names
    assert "correct_answer" not in names


def test_source_ref_chunk_id_must_match_document_and_index() -> None:
    with pytest.raises(StudyValidationError):
        validate_study_source_ref(
            StudySourceRef(
                document_id="11111111-1111-1111-1111-111111111111",
                chunk_id="bad",
                chunk_index=0,
                start_offset=0,
                end_offset=1,
            )
        )
