"""Tests for Project Cortana system identity."""

from src.identity import CORTANA_SYSTEM_INSTRUCTIONS


def test_cortana_system_instructions_define_project_identity() -> None:
    """Identity text should name Project Cortana and its defensive role."""
    instructions = CORTANA_SYSTEM_INSTRUCTIONS.lower()

    assert "project cortana" in instructions
    assert "cybersecurity" in instructions
    assert "defensive" in instructions
    assert "human supervision" in instructions


def test_cortana_system_instructions_exclude_secrets() -> None:
    """Identity text must not embed secrets or user-specific data."""
    instructions = CORTANA_SYSTEM_INSTRUCTIONS.lower()

    assert "api_key" not in instructions
    assert "password" not in instructions
    assert "token" not in instructions
    assert ".env" not in instructions
