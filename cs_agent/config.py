"""Paths, .env loading and the fixed "today" used by business rules."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_TODAY = "2026-10-13"


def load_env() -> None:
    # Same convention as learn-claude-code: .env wins over the shell.
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    if os.getenv("ANTHROPIC_BASE_URL"):
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


def today() -> date:
    return date.fromisoformat(os.getenv("CS_TODAY", DEFAULT_TODAY))


def default_threshold() -> float:
    return float(os.getenv("HANDOFF_THRESHOLD", "0.5"))


def default_needs_human_threshold() -> float:
    return float(os.getenv("NEEDS_HUMAN_THRESHOLD", "0.5"))
