"""Centralised configuration loaded from environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "tickets.db"
KB_CORPUS_PATH = DATA_DIR / "kb_corpus.json"
BENCHMARK_PATH = PROJECT_ROOT / "eval" / "benchmark_35.json"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Copy .env.example → .env and add your key."
    )

SMALL_MODEL_ID: str = os.getenv("SMALL_MODEL_ID", "gemini-2.5-flash-lite")
LARGE_MODEL_ID: str = os.getenv("LARGE_MODEL_ID", "gemini-2.5-flash")
