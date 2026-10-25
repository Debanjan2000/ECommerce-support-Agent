# E-commerce Support Orchestrator

A **ReAct Multi-Agent Orchestrator** that combines **Hybrid RAG** and **Secure Text-to-SQL** to answer e-commerce customer-support questions, powered by the **Google Gemini API** (`google-genai` SDK).

Built on the public Hugging Face dataset [`rjac/e-commerce-customer-support-qa`](https://huggingface.co/datasets/rjac/e-commerce-customer-support-qa).

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│  ReAct Orchestrator      │  max_steps = 4
│  (src/orchestrator.py)   │
│                          │
│  Think → Act → Observe   │
│         │                │
│    ┌────┴────┐           │
│    ▼         ▼           │
│ sql_tool  rag_tool       │
│    │         │           │
│    ▼         ▼           │
│ SQL Engine  RAG Engine   │
│    │         │           │
│    ▼         ▼           │
│ tickets.db  kb_corpus    │
└──────────────────────────┘
    │
    ▼
 final_answer
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Orchestrator** | `src/orchestrator.py` | ReAct loop routing to `sql_tool`, `rag_tool`, or `final_answer` |
| **RAG Engine** | `src/rag_engine.py` | BM25 + Dense (`all-MiniLM-L6-v2`) + Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) via RRF fusion |
| **SQL Engine** | `src/sql_engine.py` | Text-to-SQL with read-only SQLite, `EXPLAIN QUERY PLAN` validation, mutating-keyword blocking, self-correction retries |
| **LLM Client** | `src/llm_client.py` | Gemini SDK wrapper with exponential-backoff retry for rate limits |
| **Data Prep** | `src/prepare_data.py` | Downloads HF dataset → SQLite DB + KB corpus + 35 benchmark cases |
| **Config** | `src/config.py` | Centralised env/path config with `.env` support |

---

## Quick Start

### 1. Install dependencies

```bash
cd ecommerce_support_orchestrator
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 3. Prepare data

Downloads the dataset from Hugging Face and builds:
- `data/tickets.db` — 1,000 structured ticket rows in SQLite
- `data/kb_corpus.json` — Deduplicated troubleshooting knowledge chunks
- `eval/benchmark_35.json` — 35 deterministic test cases

```bash
python -m src.prepare_data
```

### 4. Run the benchmark

Evaluates all 35 test cases across 3 configurations and outputs a Markdown results table:

```bash
python -m eval.run_benchmark
```

Results are saved to `eval/benchmark_results.md`.

---

## Operating Modes

| Mode | SQL Retries | RAG Pipeline | JSON Repair |
|------|------------|--------------|-------------|
| `naive_baseline` | None (1 attempt) | Dense cosine only | None — raw `json.loads` |
| `engineered_full` | Up to 3 (with error feedback) | BM25 + Dense + RRF + Cross-Encoder | Markdown-fence stripping + regex extraction + 1 LLM retry |

---

## Benchmark Columns

| Column | Model | Mode |
|--------|-------|------|
| 1 | `SMALL_MODEL_ID` (default: `gemini-2.5-flash-lite`) | `naive_baseline` |
| 2 | `SMALL_MODEL_ID` | `engineered_full` |
| 3 | `LARGE_MODEL_ID` (default: `gemini-2.5-flash`) | `engineered_full` |

### Metrics

- **Execution Success Rate** — Did the orchestrator complete without errors?
- **Tool Selection Accuracy** — Did it pick the correct tool(s)?
- **RAG Retrieval Recall@3** — Were ground-truth DOC-IDs in the top-3 results?
- **RAG Retrieval MRR** — Mean Reciprocal Rank of the first correct DOC-ID
- **Citation Correct Rate** — Did the final answer cite the correct DOC-IDs?
- **Average Latency (s)** — Wall-clock time per query
- **Net Engineering Impact** — Percentage-point improvement from naive → engineered

---

## Security Guardrails (SQL Engine)

1. **Read-only URI mode** — `file:data/tickets.db?mode=ro`
2. **Mutating keyword blocking** — Rejects `UPDATE`, `DROP`, `DELETE`, `INSERT`, `ALTER`, `CREATE`, `REPLACE`, `TRUNCATE`
3. **EXPLAIN QUERY PLAN validation** — Every query is explained before execution
4. **Self-correction retries** — SQLite `OperationalError` tracebacks are fed back to the LLM for up to 3 attempts

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `SMALL_MODEL_ID` | `gemini-2.5-flash-lite` | Model for columns 1 & 2 |
| `LARGE_MODEL_ID` | `gemini-2.5-flash` | Model for column 3 |

---

## Project Structure

```
ecommerce_support_orchestrator/
├── .env.example              
├── requirements.txt          
├── README.md                 
├── src/
│   ├── __init__.py
│   ├── config.py             # Paths, env vars, model IDs
│   ├── prepare_data.py       # HF dataset → SQLite+KB+benchmark
│   ├── llm_client.py         # Gemini SDK wrapper + retry + JSON extract
│   ├── rag_engine.py         # BM25 + Dense + Cross-Encoder (CPU)
│   ├── sql_engine.py         # Text-to-SQL + read-only + EXPLAIN + retry
│   └── orchestrator.py       # ReAct loop (max_steps=4)
└── eval/
    └── run_benchmark.py      # 35-case benchmark → Markdown table
```
