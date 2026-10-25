"""Hybrid RAG Engine: BM25 (sparse) + Dense Embeddings + Cross-Encoder reranker.

All local models are forced to CPU to work on any hardware.
"""

import json
import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import KB_CORPUS_PATH

logger = logging.getLogger(__name__)


class RAGEngine:
    """In-memory hybrid retrieval over the knowledge-base corpus.

    Construction loads the corpus, builds a BM25 index, computes dense
    embeddings with ``all-MiniLM-L6-v2`` (CPU), and loads the
    ``ms-marco-MiniLM-L-6-v2`` cross-encoder reranker (CPU).
    """

    def __init__(self, corpus_path: Path | str = KB_CORPUS_PATH) -> None:
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"{corpus_path} not found.  "
                "Run `python -m src.prepare_data` first."
            )

        logger.info("Loading RAG corpus from %s …", corpus_path)
        self.docs: list[dict] = json.loads(
            corpus_path.read_text(encoding="utf-8")
        )
        self.texts: list[str] = [d["text"] for d in self.docs]

        # ── BM25 (sparse) ────────────────────────────────────────────
        logger.info("Building BM25 index over %d docs …", len(self.texts))
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # ── Dense embeddings ──────────────────────────────────────────
        logger.info("Loading sentence-transformer (CPU) …")
        self.bi_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        logger.info("Encoding %d documents …", len(self.texts))
        self.doc_embeddings: np.ndarray = self.bi_encoder.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # shape (N, dim)

        # ── Cross-encoder reranker ────────────────────────────────────
        logger.info("Loading cross-encoder reranker (CPU) …")
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        logger.info("RAG engine ready  (%d docs).", len(self.docs))

    # ==================================================================
    # Public API
    # ==================================================================
    def search(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid_rerank: bool = True,
    ) -> list[dict]:
        """Retrieve top-*k* knowledge chunks for *query*.

        Parameters
        ----------
        query : str
            Natural-language search query.
        top_k : int
            Number of results to return.
        use_hybrid_rerank : bool
            ``False`` → dense cosine similarity only.
            ``True``  → BM25(top-10) ∪ Dense(top-10) → RRF → Cross-Encoder
                        rerank → top-k.
        """
        if not use_hybrid_rerank:
            return self._dense_search(query, top_k)
        return self._hybrid_search(query, top_k)

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        """Cosine-similarity retrieval only (no BM25 / reranker)."""
        q_emb = self.bi_encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # (1, dim)
        scores = (self.doc_embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{**self.docs[i], "score": float(scores[i])} for i in top_idx]

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 + Dense → RRF → Cross-Encoder rerank."""
        k_rrf = 60  # RRF constant

        # ── BM25 top-10 ──────────────────────────────────────────────
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:10]

        # ── Dense top-10 ─────────────────────────────────────────────
        q_emb = self.bi_encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dense_scores = (self.doc_embeddings @ q_emb.T).flatten()
        dense_top = np.argsort(dense_scores)[::-1][:10]

        # ── Reciprocal Rank Fusion ───────────────────────────────────
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_top):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, idx in enumerate(dense_top):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k_rrf + rank + 1)

        # Candidates sorted by RRF score (descending)
        candidates = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)

        # ── Cross-encoder rerank ─────────────────────────────────────
        pairs = [(query, self.texts[i]) for i in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        ranked = sorted(
            zip(candidates, ce_scores), key=lambda x: x[1], reverse=True
        )

        return [
            {**self.docs[idx], "score": float(score)}
            for idx, score in ranked[:top_k]
        ]
