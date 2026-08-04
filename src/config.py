"""
Configuration settings for Project Cortana.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Important directories
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Application information
APP_NAME = "Project Cortana"
APP_DATA_DIR_NAME = "ProjectCortana"
VERSION = "0.1.0"

# Session capabilities
HISTORY_PERSISTENCE_ENABLED = False
EXPLICIT_PERSISTENT_MEMORY_ENABLED = True

# Explicit persistent memory limits and storage
MAX_MEMORY_TEXT_LENGTH = 2000
MEMORY_FILENAME = "memories.json"


def get_default_memory_file_path() -> Path:
    """Return the default user-local path for explicit persistent memories."""
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / APP_DATA_DIR_NAME / MEMORY_FILENAME
        return Path.home() / "AppData" / "Local" / APP_DATA_DIR_NAME / MEMORY_FILENAME

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return Path(xdg_data_home) / APP_DATA_DIR_NAME / MEMORY_FILENAME

    return Path.home() / ".local" / "share" / APP_DATA_DIR_NAME / MEMORY_FILENAME
