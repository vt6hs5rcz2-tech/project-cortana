"""
Configuration settings for Project Cortana.
"""

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
VERSION = "0.1.0"

# Session capabilities
HISTORY_PERSISTENCE_ENABLED = False
