"""Isolation tests for Milestone 26 multimodal modules."""

from __future__ import annotations

import ast
from pathlib import Path

from src.config import PROJECT_ROOT

VOICE_PCM_MODULES = (
    "src/realtime_voice.py",
    "src/realtime_voice_input.py",
    "src/voice_input.py",
    "src/voice_service.py",
    "src/voice_commands.py",
    "src/ai_service.py",
    "src/conversation.py",
    "src/realtime_multimodal.py",
    "src/realtime_multimodal_commands.py",
    "src/vision_normalize.py",
    "src/vision_input.py",
    "src/vision_service.py",
)

FORBIDDEN_IMPORT_NAMES = frozenset({"cv2", "numpy", "np"})


def _module_imports(relative_path: str) -> set[str]:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_cv2_numpy_confined_to_camera_capture() -> None:
    camera_imports = _module_imports("src/camera_capture.py")
    assert "cv2" in camera_imports or True  # may be function-local
    source = (PROJECT_ROOT / "src/camera_capture.py").read_text(encoding="utf-8")
    assert "import cv2" in source
    assert "import numpy" in source

    for relative in VOICE_PCM_MODULES:
        path = PROJECT_ROOT / relative
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_IMPORT_NAMES, relative
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in FORBIDDEN_IMPORT_NAMES, relative
        assert "import cv2" not in text, relative
        assert "import numpy" not in text, relative
        assert "from numpy" not in text, relative
        assert "from cv2" not in text, relative


def test_multimodal_modules_avoid_operational_domains() -> None:
    forbidden = frozenset(
        {
            "tool_executor",
            "workflow_executor",
            "calendar_service",
            "reminder_service",
            "security_commands",
            "incident_repository",
            "memory_store",
            "study_service",
        }
    )
    for relative in (
        "src/camera_capture.py",
        "src/realtime_multimodal.py",
        "src/realtime_multimodal_commands.py",
    ):
        tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] == "src" and len(parts) > 1:
                    assert parts[1] not in forbidden, relative
