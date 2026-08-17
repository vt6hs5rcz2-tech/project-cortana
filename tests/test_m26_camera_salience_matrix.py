"""Isolated proof tests for the M26 camera-salience matrix spike.

No network. Does not change production routing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from src.realtime_multimodal import MULTIMODAL_CONVERSATION_INSTRUCTIONS

_SPIKE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "spike_m26_camera_salience_matrix.py"
)


def _load_spike() -> object:
    spec = importlib.util.spec_from_file_location(
        "spike_m26_camera_salience_matrix",
        _SPIKE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_salience_matrix_question_is_fixed() -> None:
    spike = _load_spike()
    assert spike.QUESTION == "What object is visible?"


def test_salience_matrix_arms_cover_required_isolations() -> None:
    spike = _load_spike()
    names = [arm.name for arm in spike.MATRIX_ARMS]
    assert names[:4] == ["A", "B", "C", "D"]
    assert spike.MATRIX_ARMS[0].question_mode == "text"
    assert spike.MATRIX_ARMS[0].image_mode == "synthetic"
    assert spike.MATRIX_ARMS[1].question_mode == "text"
    assert spike.MATRIX_ARMS[1].image_mode == "camera"
    assert spike.MATRIX_ARMS[2].question_mode == "audio"
    assert spike.MATRIX_ARMS[2].image_mode == "synthetic"
    assert spike.MATRIX_ARMS[3].question_mode == "audio"
    assert spike.MATRIX_ARMS[3].image_mode == "camera"


def test_salience_matrix_uses_production_session_instructions() -> None:
    spike = _load_spike()
    fingerprint = spike.instructions_fingerprint(MULTIMODAL_CONVERSATION_INSTRUCTIONS)
    digest = hashlib.sha256(
        MULTIMODAL_CONVERSATION_INSTRUCTIONS.encode("utf-8")
    ).hexdigest()[:16]
    assert fingerprint == (
        f"sha256={digest} len={len(MULTIMODAL_CONVERSATION_INSTRUCTIONS)}"
    )
    assert "untrusted visual" not in MULTIMODAL_CONVERSATION_INSTRUCTIONS.lower()


def test_salience_classifier_synthetic_and_refusal() -> None:
    spike = _load_spike()
    assert (
        spike.classify_transcript(
            "I see a red square in the image.",
            image_mode="synthetic",
        )
        == "IDENTIFIED"
    )
    assert (
        spike.classify_transcript(
            "I'm not relying on visuals. Please describe the object verbally.",
            image_mode="camera",
        )
        == "REFUSED"
    )
    assert (
        spike.classify_transcript(
            "You are holding a black remote control.",
            image_mode="camera",
        )
        == "IDENTIFIED"
    )
