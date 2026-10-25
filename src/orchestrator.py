"""ReAct Multi-Agent Orchestrator routing queries to SQL / RAG tools.

Supports two operating modes:

* **naive_baseline** – single-shot SQL (no retry), dense-only RAG (no BM25 /
  reranker), no JSON repair.
* **engineered_full** – SQL self-correction harness, hybrid BM25 + Dense + RRF
  + Cross-Encoder reranker, and a JSON-repair retry loop for LLM output.
"""

import json
import logging
import time

from src.config import SMALL_MODEL_ID
from src.llm_client import extract_json, generate_text
from src.rag_engine import RAGEngine
from src.sql_engine import SQLEngine

logger = logging.getLogger(__name__)

# ── ReAct system prompt ───────────────────────────────────────────────
_REACT_SYSTEM = """\
You are an expert e-commerce customer-support orchestrator.

## Available tools
1. **sql_tool** – Query the structured ``support_tickets`` SQLite table
   (counts, aggregations, filters over ticket metadata).
   Use for quantitative questions (how many, breakdown, distribution).
2. **rag_tool** – Search the troubleshooting knowledge base of
   customer-support Q&A articles.
   Use for procedural / how-to / troubleshooting questions.
3. **final_answer** – Return the finished answer to the user.

## Response format
At every step respond with EXACTLY this JSON object (no markdown fences,
no extra text):
{{
  "thought": "<your step-by-step reasoning>",
  "action":  "<sql_tool | rag_tool | final_answer>",
  "action_input": "<the natural-language query for the tool, or the final answer text>"
}}

## Rules
- Think step-by-step before choosing a tool.
- If the question needs BOTH ticket counts AND a troubleshooting procedure,
  call **sql_tool** first, then **rag_tool**, then **final_answer**.
- When using **final_answer** after a RAG lookup, you MUST include the DOC-IDs
  of the knowledge articles you referenced (e.g. "Sources: DOC-042, DOC-117").
- You have a maximum of {max_steps} steps before you MUST call final_answer.
- If a tool returns an error you may retry ONCE with a refined query.
"""


class Orchestrator:
    """ReAct loop (max 4 steps) over SQL and RAG tools."""

    def __init__(
        self,
        rag_engine: RAGEngine,
        sql_engine: SQLEngine,
        mode: str = "engineered_full",
    ) -> None:
        if mode not in ("naive_baseline", "engineered_full"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.rag = rag_engine
        self.sql = sql_engine
        self.mode = mode
        self._use_harness = mode == "engineered_full"
        self._use_hybrid_rerank = mode == "engineered_full"

    # ==================================================================
    # Public API
    # ==================================================================
    def run(
        self,
        query: str,
        model_id: str = SMALL_MODEL_ID,
        max_steps: int = 4,
    ) -> dict:
        """Execute the ReAct loop and return a structured result.

        Returns
        -------
        dict
            ``answer``      – final answer text
            ``tool_trace``  – list of {action, action_input, observation}
            ``latency_s``   – total wall-clock seconds
            ``error``       – error message or ``None``
        """
        t0 = time.time()
        system = _REACT_SYSTEM.format(max_steps=max_steps)
        history: list[str] = [f"User question: {query}"]
        tool_trace: list[dict] = []

        for step in range(max_steps):
            prompt = "\n\n".join(history)

            raw = generate_text(
                prompt=prompt,
                system_instruction=system,
                model_id=model_id,
                temperature=0.0,
            )

            # ── Parse agent JSON ──────────────────────────────────────
            parsed = self._parse_agent_json(raw, prompt, system, model_id)
            if parsed is None:
                return {
                    "answer": raw,
                    "tool_trace": tool_trace,
                    "latency_s": time.time() - t0,
                    "error": "JSON parse failure (all retries exhausted)",
                }

            action = parsed.get("action", "final_answer")
            action_input = str(parsed.get("action_input", ""))
            thought = parsed.get("thought", "")

            logger.info(
                "Step %d │ action=%-12s │ thought=%s",
                step + 1,
                action,
                thought[:80],
            )

            # ── final_answer ──────────────────────────────────────────
            if action == "final_answer":
                tool_trace.append(
                    {
                        "action": "final_answer",
                        "action_input": action_input,
                        "observation": None,
                    }
                )
                return {
                    "answer": action_input,
                    "tool_trace": tool_trace,
                    "latency_s": time.time() - t0,
                    "error": None,
                }

            # ── sql_tool ──────────────────────────────────────────────
            if action == "sql_tool":
                result = self.sql.execute_sql_question(
                    question=action_input,
                    model_id=model_id,
                    use_harness=self._use_harness,
                )
                observation = json.dumps(result, default=str)
                tool_trace.append(
                    {
                        "action": "sql_tool",
                        "action_input": action_input,
                        "observation": observation,
                    }
                )
                history.append(
                    f"Step {step + 1}:\n"
                    f"Thought: {thought}\n"
                    f"Action: sql_tool\n"
                    f"Action Input: {action_input}\n"
                    f"Observation: {observation}"
                )
                continue

            # ── rag_tool ──────────────────────────────────────────────
            if action == "rag_tool":
                results = self.rag.search(
                    query=action_input,
                    top_k=3,
                    use_hybrid_rerank=self._use_hybrid_rerank,
                )
                compact = [
                    {
                        "doc_id": r["doc_id"],
                        "question": r["question"],
                        "solution": r["solution"],
                        "score": round(r["score"], 4),
                    }
                    for r in results
                ]
                observation = json.dumps(compact, default=str)
                tool_trace.append(
                    {
                        "action": "rag_tool",
                        "action_input": action_input,
                        "observation": observation,
                    }
                )
                history.append(
                    f"Step {step + 1}:\n"
                    f"Thought: {thought}\n"
                    f"Action: rag_tool\n"
                    f"Action Input: {action_input}\n"
                    f"Observation: {observation}"
                )
                continue

            # ── Unknown action ────────────────────────────────────────
            history.append(
                f"Step {step + 1}:\n"
                f"[System: Unknown action '{action}'. "
                f"Use sql_tool, rag_tool, or final_answer.]"
            )

        # Exceeded max_steps without calling final_answer
        return {
            "answer": "Max steps exceeded without final answer.",
            "tool_trace": tool_trace,
            "latency_s": time.time() - t0,
            "error": "max_steps_exceeded",
        }

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _parse_agent_json(
        self,
        raw: str,
        prompt: str,
        system: str,
        model_id: str,
    ) -> dict | None:
        """Parse the agent's JSON response, with optional repair retry.

        In **naive_baseline** mode: one ``json.loads`` attempt, no repair.
        In **engineered_full** mode: ``extract_json`` + one LLM-level retry
        with an explicit error message asking for valid JSON.
        """
        if self.mode == "naive_baseline":
            try:
                return json.loads(raw.strip())
            except json.JSONDecodeError:
                return None

        # engineered_full  — robust extraction + retry
        try:
            return extract_json(raw)
        except ValueError:
            pass

        # One repair retry
        retry_prompt = (
            prompt
            + "\n\n[System: Your previous response was NOT valid JSON. "
            "Respond with ONLY the JSON object — no markdown, no extra text.]"
        )
        raw2 = generate_text(
            prompt=retry_prompt,
            system_instruction=system,
            model_id=model_id,
            temperature=0.0,
        )
        try:
            return extract_json(raw2)
        except ValueError:
            return None
