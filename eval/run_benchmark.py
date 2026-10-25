"""Run the 35-case benchmark across 3 configuration columns.

Outputs a Markdown results table to ``eval/benchmark_results.md``.

Columns
-------
1. SMALL_MODEL_ID  (naive_baseline)
2. SMALL_MODEL_ID  (engineered_full)
3. LARGE_MODEL_ID  (engineered_full)

Usage::

    python -m eval.run_benchmark
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    BENCHMARK_PATH,
    LARGE_MODEL_ID,
    SMALL_MODEL_ID,
)
from src.orchestrator import Orchestrator  # noqa: E402
from src.rag_engine import RAGEngine  # noqa: E402
from src.sql_engine import SQLEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
)
logger = logging.getLogger("benchmark")

RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.md"


# ======================================================================
# Metric helpers
# ======================================================================
def _tool_selected_correctly(
    trace: list[dict], expected_tool: str
) -> bool:
    """Did the orchestrator invoke the correct tool(s)?"""
    used = {t["action"] for t in trace if t["action"] != "final_answer"}
    if "+" in expected_tool:
        expected = {t.strip() for t in expected_tool.split("+")}
        return expected.issubset(used)
    return expected_tool in used


def _rag_recall_at_k(
    trace: list[dict], expected_ids: list[str], k: int = 3
) -> float:
    """Proportion of expected DOC-IDs found in top-k retrieved."""
    if not expected_ids:
        return 1.0  # vacuously true
    retrieved: list[str] = []
    for t in trace:
        if t["action"] == "rag_tool" and t["observation"]:
            try:
                docs = json.loads(t["observation"])
                retrieved.extend(d.get("doc_id", "") for d in docs[:k])
            except (json.JSONDecodeError, TypeError):
                pass
    return sum(1 for d in expected_ids if d in retrieved) / len(expected_ids)


def _rag_mrr(trace: list[dict], expected_ids: list[str]) -> float:
    """Mean Reciprocal Rank of first expected DOC-ID in retrieved list."""
    if not expected_ids:
        return 1.0
    retrieved: list[str] = []
    for t in trace:
        if t["action"] == "rag_tool" and t["observation"]:
            try:
                docs = json.loads(t["observation"])
                retrieved.extend(d.get("doc_id", "") for d in docs)
            except (json.JSONDecodeError, TypeError):
                pass
    for rank, rid in enumerate(retrieved, 1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def _citation_correct(answer: str, expected_ids: list[str]) -> bool:
    """Does the final answer text mention all expected DOC-IDs?"""
    if not expected_ids:
        return True
    return all(did in answer for did in expected_ids)


def _sql_result_matches(
    trace: list[dict], expected_result
) -> bool:
    """Check if any sql_tool observation contains the expected value."""
    if expected_result is None:
        return True
    for t in trace:
        if t["action"] != "sql_tool" or not t["observation"]:
            continue
        try:
            obs = json.loads(t["observation"])
        except (json.JSONDecodeError, TypeError):
            continue
        if obs.get("error"):
            continue
        rows = obs.get("rows", [])

        # Simple scalar (count)
        if isinstance(expected_result, (int, float)):
            for row in rows:
                if expected_result in row.values():
                    return True

        # Dict (group-by top-N)
        elif isinstance(expected_result, dict):
            found: dict = {}
            for row in rows:
                vals = list(row.values())
                if len(vals) >= 2:
                    found[vals[0]] = vals[1]
            if all(found.get(k) == v for k, v in expected_result.items()):
                return True
    return False


# ======================================================================
# Run one configuration column
# ======================================================================
def _run_column(
    name: str,
    model_id: str,
    mode: str,
    cases: list[dict],
    rag: RAGEngine,
    sql: SQLEngine,
) -> dict:
    """Execute all benchmark cases under one config; return aggregate metrics."""
    logger.info("══════════════════════════════════════════════════")
    logger.info("Column: %s  (model=%s  mode=%s)", name, model_id, mode)
    logger.info("══════════════════════════════════════════════════")

    orch = Orchestrator(rag, sql, mode=mode)

    exec_ok = 0
    tool_ok = 0
    rag_recall_sum = 0.0
    rag_mrr_sum = 0.0
    citation_ok = 0
    latencies: list[float] = []
    rag_case_count = 0

    for case in cases:
        cid = case["id"]
        question = case["question"]
        expected_tool = case["expected_tool"]
        expected_doc_ids = case.get("ground_truth_doc_ids", [])
        expected_sql = case.get("ground_truth_sql_result")

        logger.info("[%s] %s: %s", name, cid, question[:80])

        try:
            result = orch.run(query=question, model_id=model_id)
            success = result["error"] is None
        except Exception as exc:
            logger.error("  EXCEPTION: %s", exc)
            result = {
                "answer": "",
                "tool_trace": [],
                "latency_s": 0.0,
                "error": str(exc),
            }
            success = False

        exec_ok += int(success)
        tool_ok += int(
            _tool_selected_correctly(result["tool_trace"], expected_tool)
        )
        latencies.append(result["latency_s"])

        # RAG metrics (only for cases with expected DOC-IDs)
        if expected_doc_ids:
            rag_case_count += 1
            rag_recall_sum += _rag_recall_at_k(
                result["tool_trace"], expected_doc_ids
            )
            rag_mrr_sum += _rag_mrr(result["tool_trace"], expected_doc_ids)
            citation_ok += int(
                _citation_correct(
                    result.get("answer", ""), expected_doc_ids
                )
            )

        logger.info(
            "  → ok=%s  lat=%.1fs  ans=%s",
            success,
            result["latency_s"],
            str(result.get("answer", ""))[:100],
        )

    n = len(cases)
    return {
        "name": name,
        "exec_success_rate": exec_ok / n if n else 0,
        "tool_selection_accuracy": tool_ok / n if n else 0,
        "rag_recall_at_3": (
            rag_recall_sum / rag_case_count if rag_case_count else 0
        ),
        "rag_mrr": rag_mrr_sum / rag_case_count if rag_case_count else 0,
        "citation_correct_rate": (
            citation_ok / rag_case_count if rag_case_count else 0
        ),
        "avg_latency_s": sum(latencies) / n if n else 0,
    }


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    if not BENCHMARK_PATH.exists():
        logger.error(
            "%s not found. Run `python -m src.prepare_data` first.",
            BENCHMARK_PATH,
        )
        sys.exit(1)

    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    logger.info(
        "Loaded %d benchmark cases from %s", len(cases), BENCHMARK_PATH
    )

    # Shared engines (loaded once, reused across all columns)
    logger.info("Initialising shared RAG + SQL engines …")
    rag = RAGEngine()
    sql = SQLEngine()

    # ── Column 1: Small model, naive ──────────────────────────────────
    col1 = _run_column(
        name=f"{SMALL_MODEL_ID} (naive)",
        model_id=SMALL_MODEL_ID,
        mode="naive_baseline",
        cases=cases,
        rag=rag,
        sql=sql,
    )

    # ── Column 2: Small model, engineered ─────────────────────────────
    col2 = _run_column(
        name=f"{SMALL_MODEL_ID} (engineered)",
        model_id=SMALL_MODEL_ID,
        mode="engineered_full",
        cases=cases,
        rag=rag,
        sql=sql,
    )

    # ── Column 3: Large model, engineered ─────────────────────────────
    col3 = _run_column(
        name=f"{LARGE_MODEL_ID} (engineered)",
        model_id=LARGE_MODEL_ID,
        mode="engineered_full",
        cases=cases,
        rag=rag,
        sql=sql,
    )

    # ── Build Markdown results table ──────────────────────────────────
    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def delta(naive: float, eng: float) -> str:
        d = (eng - naive) * 100
        return f"{'+' if d >= 0 else ''}{d:.2f} pp"

    metrics = [
        ("Execution Success Rate", "exec_success_rate"),
        ("Tool Selection Accuracy", "tool_selection_accuracy"),
        ("RAG Retrieval Recall@3", "rag_recall_at_3"),
        ("RAG Retrieval MRR", "rag_mrr"),
        ("Citation Correct Rate", "citation_correct_rate"),
        ("Average Latency (s)", "avg_latency_s"),
    ]

    lines = [
        "# Benchmark Results\n",
        (
            f"| Metric | {col1['name']} | {col2['name']} "
            f"| {col3['name']} | Net Engineering Impact |"
        ),
        "|--------|" + "|".join(["--------"] * 4) + "|",
    ]

    for label, key in metrics:
        v1, v2, v3 = col1[key], col2[key], col3[key]
        if key == "avg_latency_s":
            lines.append(
                f"| {label} | {v1:.2f}s | {v2:.2f}s "
                f"| {v3:.2f}s | {delta(v1, v2)} |"
            )
        else:
            lines.append(
                f"| {label} | {pct(v1)} | {pct(v2)} "
                f"| {pct(v3)} | {delta(v1, v2)} |"
            )

    md = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(md, encoding="utf-8")
    logger.info("Results saved to %s", RESULTS_PATH)
    print("\n" + md)


if __name__ == "__main__":
    main()
