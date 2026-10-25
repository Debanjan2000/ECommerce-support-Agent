"""Secure Text-to-SQL engine with read-only SQLite, EXPLAIN validation,
mutating-keyword blocking, and self-correction retries.
"""

import logging
import re
import sqlite3
from pathlib import Path

from src.config import DB_PATH
from src.llm_client import generate_text

logger = logging.getLogger(__name__)

# ── Guard: keywords that MUST NOT appear in generated SQL ─────────────
_MUTATING_RE = re.compile(
    r"\b(UPDATE|DELETE|DROP|INSERT|ALTER|CREATE|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)

# ── System prompt template ────────────────────────────────────────────
_SQL_SYSTEM_PROMPT = """\
You are a read-only SQLite SQL generator.  Given a natural-language question
about e-commerce support tickets, output **only** a valid SQLite SELECT query.

• Do NOT output any explanation, markdown fences, or comments — just raw SQL.
• Never use UPDATE, DELETE, DROP, INSERT, ALTER.
• Use exact column names and values from the schema / distinct values below.
• String comparisons are case-sensitive; use values exactly as listed.

────────────────────────────────────────────
SCHEMA
────────────────────────────────────────────
{schema}

────────────────────────────────────────────
DISTINCT CATEGORICAL VALUES
────────────────────────────────────────────
{distinct_values}
"""


class SQLEngine:
    """Text-to-SQL with read-only SQLite and optional self-correction retries.

    The database is opened in URI read-only mode (``?mode=ro``) to guarantee
    that even a malicious query cannot mutate data.
    """

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found.  "
                "Run `python -m src.prepare_data` first."
            )
        self._db_uri = f"file:{db_path.as_posix()}?mode=ro"
        self._schema = self._load_schema()
        self._distinct_values = self._load_distinct_values()

    # ==================================================================
    # Schema introspection (run once at init)
    # ==================================================================
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_schema(self) -> str:
        conn = self._connect()
        row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='support_tickets'"
        ).fetchone()
        conn.close()
        return row["sql"] if row else ""

    def _load_distinct_values(self) -> str:
        conn = self._connect()
        cols = [
            "issue_area",
            "issue_category",
            "issue_sub_category",
            "product_category",
            "product_sub_category",
            "agent_shift",
            "customer_sentiment",
            "issue_complexity",
            "agent_experience_level",
        ]
        parts: list[str] = []
        for col in cols:
            rows = conn.execute(
                f"SELECT DISTINCT {col} FROM support_tickets ORDER BY {col}"
            ).fetchall()
            vals = [r[0] for r in rows]
            parts.append(f"  {col}: {vals}")
        conn.close()
        return "\n".join(parts)

    # ==================================================================
    # Public API
    # ==================================================================
    def execute_sql_question(
        self,
        question: str,
        model_id: str,
        use_harness: bool = True,
    ) -> dict:
        """Convert *question* to SQL via LLM and execute against SQLite.

        Parameters
        ----------
        question : str
            Natural-language question about ticket data.
        model_id : str
            Gemini model to use for SQL generation.
        use_harness : bool
            ``True``  → up to 3 self-correction retries feeding SQLite
            ``OperationalError`` tracebacks back to the LLM.
            ``False`` → single attempt only (no retry).

        Returns
        -------
        dict
            ``{"sql": str, "columns": list, "rows": list[dict], "error": str|None}``
        """
        system = _SQL_SYSTEM_PROMPT.format(
            schema=self._schema,
            distinct_values=self._distinct_values,
        )
        max_attempts = 3 if use_harness else 1
        last_error: str | None = None
        prompt = question

        for attempt in range(max_attempts):
            # ── Generate SQL ──────────────────────────────────────────
            raw = generate_text(
                prompt=prompt,
                system_instruction=system,
                model_id=model_id,
                temperature=0.0,
            ).strip()

            # Strip markdown fences the model might add
            sql = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.MULTILINE)
            sql = re.sub(r"```\s*$", "", sql, flags=re.MULTILINE).strip()
            # Keep only the first statement
            sql = sql.split(";")[0].strip()

            if not sql:
                last_error = "LLM returned empty SQL."
                prompt = (
                    f"{question}\n\n"
                    "[Previous attempt returned empty SQL. "
                    "Please output a valid SELECT query.]"
                )
                continue

            # ── Block mutating keywords ───────────────────────────────
            if _MUTATING_RE.search(sql):
                return {
                    "sql": sql,
                    "columns": [],
                    "rows": [],
                    "error": "BLOCKED: query contains mutating keywords.",
                }

            # ── Validate with EXPLAIN then execute ────────────────────
            conn = self._connect()
            try:
                conn.execute(f"EXPLAIN QUERY PLAN {sql}")
                cursor = conn.execute(sql)
                columns = (
                    [d[0] for d in cursor.description]
                    if cursor.description
                    else []
                )
                rows = [dict(r) for r in cursor.fetchall()]
                conn.close()
                return {
                    "sql": sql,
                    "columns": columns,
                    "rows": rows,
                    "error": None,
                }

            except sqlite3.OperationalError as exc:
                last_error = str(exc)
                conn.close()
                if use_harness and attempt < max_attempts - 1:
                    logger.warning(
                        "SQL attempt %d failed: %s — retrying …",
                        attempt + 1,
                        last_error,
                    )
                    prompt = (
                        f"{question}\n\n"
                        f"[Your previous SQL was:\n{sql}\n"
                        f"It failed with SQLite error: {last_error}\n"
                        f"Please fix the query and try again.]"
                    )
                    continue
                return {
                    "sql": sql,
                    "columns": [],
                    "rows": [],
                    "error": last_error,
                }

        return {"sql": "", "columns": [], "rows": [], "error": last_error}
