"""Download ``rjac/e-commerce-customer-support-qa`` and prepare:

   1. ``data/tickets.db``           – structured ticket metadata (SQLite)
   2. ``data/kb_corpus.json``       – deduplicated troubleshooting knowledge chunks
   3. ``eval/benchmark_35.json``    – 35 deterministic test cases

Run with::

    python -m src.prepare_data
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import BENCHMARK_PATH, DATA_DIR, DB_PATH, KB_CORPUS_PATH

# Deterministic shift assignment (not in the original dataset)
_SHIFTS = ["morning", "afternoon", "night"]


# ======================================================================
# 1. SQLite ticket database
# ======================================================================
def _build_tickets_db(df: pd.DataFrame) -> None:
    """Create ``data/tickets.db`` with table ``support_tickets``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE support_tickets (
            ticket_id              INTEGER PRIMARY KEY,
            issue_area             TEXT NOT NULL,
            issue_category         TEXT NOT NULL,
            issue_sub_category     TEXT NOT NULL,
            product_category       TEXT NOT NULL,
            product_sub_category   TEXT NOT NULL,
            agent_shift            TEXT NOT NULL,
            customer_sentiment     TEXT NOT NULL,
            issue_complexity       TEXT NOT NULL,
            agent_experience_level TEXT NOT NULL
        );
        """
    )

    rows: list[tuple] = []
    for idx, row in df.iterrows():
        rows.append(
            (
                int(idx),
                row["issue_area"],
                row["issue_category"],
                row["issue_sub_category"],
                row["product_category"],
                row["product_sub_category"],
                _SHIFTS[int(idx) % len(_SHIFTS)],
                row["customer_sentiment"],
                row["issue_complexity"],
                row["agent_experience_level"],
            )
        )

    cur.executemany(
        "INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )

    # Indexes on columns commonly filtered / grouped
    cur.execute(
        "CREATE INDEX idx_issue_area ON support_tickets(issue_area);"
    )
    cur.execute(
        "CREATE INDEX idx_product_category ON support_tickets(product_category);"
    )
    cur.execute(
        "CREATE INDEX idx_product_sub_category ON support_tickets(product_sub_category);"
    )

    conn.commit()
    conn.close()
    print(f"  ✓ Created {DB_PATH}  ({len(rows)} rows)")


# ======================================================================
# 2. Knowledge-base corpus
# ======================================================================
def _build_kb_corpus(df: pd.DataFrame) -> list[dict]:
    """Parse the ``qa`` JSON column, deduplicate, and save to JSON."""
    seen_hashes: set[str] = set()
    docs: list[dict] = []

    for _, row in df.iterrows():
        try:
            qa_data = json.loads(row["qa"]) if isinstance(row["qa"], str) else row["qa"]
        except (json.JSONDecodeError, TypeError):
            continue

        for item in qa_data.get("knowledge", []):
            question = item.get("customer_summary_question", "").strip()
            solution = item.get("agent_summary_solution", "").strip()
            if not question or not solution:
                continue

            content_hash = hashlib.sha256(
                (question + "||" + solution).encode()
            ).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            docs.append(
                {
                    "question": question,
                    "solution": solution,
                    "product_category": row["product_category"],
                    "product_sub_category": row["product_sub_category"],
                    "issue_category": row["issue_category"],
                }
            )

    # Deterministic sort and ID assignment
    docs.sort(
        key=lambda d: (
            d["product_category"],
            d["issue_category"],
            d["question"],
        )
    )
    for i, doc in enumerate(docs):
        doc["doc_id"] = f"DOC-{i + 1:03d}"
        doc["text"] = (
            f"Product: {doc['product_category']} > {doc['product_sub_category']}\n"
            f"Issue: {doc['issue_category']}\n"
            f"Q: {doc['question']}\n"
            f"A: {doc['solution']}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KB_CORPUS_PATH.write_text(
        json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✓ Created {KB_CORPUS_PATH}  ({len(docs)} knowledge chunks)")
    return docs


# ======================================================================
# 3. Benchmark (35 deterministic test cases)
# ======================================================================
def _build_benchmark(docs: list[dict]) -> None:
    """Generate ``eval/benchmark_35.json`` from the live DB + corpus."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cases: list[dict] = []

    # ── 12 SQL-only cases ─────────────────────────────────────────────
    # SQL-01  Total count
    total = cur.execute(
        "SELECT COUNT(*) AS cnt FROM support_tickets"
    ).fetchone()["cnt"]
    cases.append(
        {
            "id": "SQL-01",
            "type": "sql",
            "question": "How many total support tickets are in the database?",
            "expected_tool": "sql_tool",
            "ground_truth_sql_result": total,
            "ground_truth_doc_ids": [],
            "ground_truth_keywords": [],
        }
    )

    # SQL-02 … SQL-05  Count by single column (most-common value)
    single_cols = [
        ("issue_area", "issue area"),
        ("product_category", "product category"),
        ("customer_sentiment", "customer sentiment"),
        ("issue_complexity", "issue complexity"),
    ]
    for col, label in single_cols:
        row = cur.execute(
            f"SELECT {col} AS val, COUNT(*) AS cnt "
            f"FROM support_tickets GROUP BY {col} ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        val, cnt = row["val"], row["cnt"]
        cases.append(
            {
                "id": f"SQL-{len(cases) + 1:02d}",
                "type": "sql",
                "question": f"How many support tickets have {label} '{val}'?",
                "expected_tool": "sql_tool",
                "ground_truth_sql_result": cnt,
                "ground_truth_doc_ids": [],
                "ground_truth_keywords": [],
            }
        )

    # SQL-06  Multi-column filter (issue_area × product_category)
    row = cur.execute(
        "SELECT issue_area, product_category, COUNT(*) AS cnt "
        "FROM support_tickets GROUP BY issue_area, product_category "
        "ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    cases.append(
        {
            "id": f"SQL-{len(cases) + 1:02d}",
            "type": "sql",
            "question": (
                f"How many tickets have issue area '{row['issue_area']}' "
                f"AND product category '{row['product_category']}'?"
            ),
            "expected_tool": "sql_tool",
            "ground_truth_sql_result": row["cnt"],
            "ground_truth_doc_ids": [],
            "ground_truth_keywords": [],
        }
    )

    # SQL-07  Multi-column filter (sentiment × complexity)
    row = cur.execute(
        "SELECT customer_sentiment, issue_complexity, COUNT(*) AS cnt "
        "FROM support_tickets GROUP BY customer_sentiment, issue_complexity "
        "ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    cases.append(
        {
            "id": f"SQL-{len(cases) + 1:02d}",
            "type": "sql",
            "question": (
                f"How many tickets have customer sentiment "
                f"'{row['customer_sentiment']}' and issue complexity "
                f"'{row['issue_complexity']}'?"
            ),
            "expected_tool": "sql_tool",
            "ground_truth_sql_result": row["cnt"],
            "ground_truth_doc_ids": [],
            "ground_truth_keywords": [],
        }
    )

    # SQL-08 … SQL-11  Group-by (top 3)
    group_cols = [
        ("issue_area", "issue area"),
        ("product_category", "product category"),
        ("customer_sentiment", "customer sentiment"),
        ("issue_complexity", "issue complexity"),
    ]
    for col, label in group_cols:
        rows = cur.execute(
            f"SELECT {col} AS val, COUNT(*) AS cnt "
            f"FROM support_tickets GROUP BY {col} ORDER BY cnt DESC LIMIT 3"
        ).fetchall()
        expected = {r["val"]: r["cnt"] for r in rows}
        cases.append(
            {
                "id": f"SQL-{len(cases) + 1:02d}",
                "type": "sql",
                "question": (
                    f"What are the top 3 {label} categories by ticket count?"
                ),
                "expected_tool": "sql_tool",
                "ground_truth_sql_result": expected,
                "ground_truth_doc_ids": [],
                "ground_truth_keywords": [],
            }
        )

    # SQL-12  product_sub_category filter
    row = cur.execute(
        "SELECT product_sub_category, COUNT(*) AS cnt "
        "FROM support_tickets GROUP BY product_sub_category "
        "ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    cases.append(
        {
            "id": f"SQL-{len(cases) + 1:02d}",
            "type": "sql",
            "question": (
                f"How many tickets involve the product sub-category "
                f"'{row['product_sub_category']}'?"
            ),
            "expected_tool": "sql_tool",
            "ground_truth_sql_result": row["cnt"],
            "ground_truth_doc_ids": [],
            "ground_truth_keywords": [],
        }
    )

    # ── 12 RAG-only cases ─────────────────────────────────────────────
    step = max(1, len(docs) // 12)
    rag_docs = [docs[i * step] for i in range(12)]
    for i, doc in enumerate(rag_docs):
        solution_words = [w for w in doc["solution"].split() if len(w) > 4][:5]
        cases.append(
            {
                "id": f"RAG-{i + 1:02d}",
                "type": "rag",
                "question": doc["question"],
                "expected_tool": "rag_tool",
                "ground_truth_sql_result": None,
                "ground_truth_doc_ids": [doc["doc_id"]],
                "ground_truth_keywords": solution_words,
            }
        )

    # ── 11 Hybrid cases ───────────────────────────────────────────────
    combos = cur.execute(
        "SELECT product_sub_category, issue_category, COUNT(*) AS cnt "
        "FROM support_tickets "
        "GROUP BY product_sub_category, issue_category "
        "ORDER BY cnt DESC"
    ).fetchall()

    hybrid_count = 0
    for combo in combos:
        if hybrid_count >= 11:
            break
        psc = combo["product_sub_category"]
        ic = combo["issue_category"]
        cnt = combo["cnt"]

        matching = [
            d
            for d in docs
            if d["product_sub_category"] == psc and d["issue_category"] == ic
        ]
        if not matching:
            continue
        doc = matching[0]
        solution_words = [w for w in doc["solution"].split() if len(w) > 4][:5]

        cases.append(
            {
                "id": f"HYB-{hybrid_count + 1:02d}",
                "type": "hybrid",
                "question": (
                    f"How many support tickets involve '{psc}' with "
                    f"'{ic}' issues, and what is the recommended "
                    f"troubleshooting procedure?"
                ),
                "expected_tool": "sql_tool+rag_tool",
                "ground_truth_sql_result": cnt,
                "ground_truth_doc_ids": [doc["doc_id"]],
                "ground_truth_keywords": solution_words,
            }
        )
        hybrid_count += 1

    conn.close()

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✓ Created {BENCHMARK_PATH}  ({len(cases)} test cases)")


# ======================================================================
# Entry point
# ======================================================================
def main() -> None:
    """Run all data-preparation steps end-to-end."""
    print("Loading dataset  rjac/e-commerce-customer-support-qa …")
    ds = load_dataset("rjac/e-commerce-customer-support-qa", split="train")
    df = ds.to_pandas()
    print(f"  Loaded {len(df)} rows\n")

    _build_tickets_db(df)
    docs = _build_kb_corpus(df)
    _build_benchmark(docs)

    print("\n✅  Data preparation complete.")


if __name__ == "__main__":
    main()
