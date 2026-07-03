"""
hybrid_retriever.py – Hybrid Search combining FAISS (semantic) + BM25 (keyword).

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval methods
into a single ranked list.

This is the architecture used by top-performing teams in ALQAC 2024/2025.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from configs.config import (
    EMBEDDINGS_SAVE,
    FAISS_INDEX,
    BM25_INDEX,
    NUM_RESULTS,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
)
from rag_runner.indexer import (
    load_faiss_index,
    load_bm25_index,
    build_all_indexes,
    _tokenize_vietnamese,
)
from rag_runner.embedder import get_embedding_model


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines FAISS (semantic vector search) with BM25 (keyword search)
    using Reciprocal Rank Fusion (RRF).

    Usage:
        retriever = HybridRetriever.from_disk()   # load pre-built indexes
        results   = retriever.search("câu hỏi pháp luật", k=5)
    """

    def __init__(
        self,
        faiss_index: faiss.IndexFlatIP,
        bm25_index: BM25Okapi,
        df: pd.DataFrame,
    ):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.df = df
        self._model = None  # lazy-loaded embedding model

    @classmethod
    def from_disk(
        cls,
        parquet_path: Path = EMBEDDINGS_SAVE,
        faiss_path: Path = FAISS_INDEX,
        bm25_path: Path = BM25_INDEX,
    ) -> "HybridRetriever":
        """Load pre-built indexes from disk."""
        faiss_idx, bm25_idx, df = build_all_indexes(
            parquet_path, faiss_path, bm25_path, force=False
        )
        return cls(faiss_idx, bm25_idx, df)

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model

    # -----------------------------------------------------------------------
    # Individual search methods
    # -----------------------------------------------------------------------

    def _faiss_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """
        Semantic search via FAISS.

        Returns list of (doc_index, score) sorted by descending score.
        """
        model = self._get_model()
        query_emb = model.encode(
            query, normalize_embeddings=True
        ).reshape(1, -1).astype(np.float32)

        scores, indices = self.faiss_index.search(query_emb, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """
        Keyword search via BM25.

        Returns list of (doc_index, score) sorted by descending score.
        """
        tokens = _tokenize_vietnamese(query)
        scores = self.bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    # -----------------------------------------------------------------------
    # Hybrid search with Reciprocal Rank Fusion
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = NUM_RESULTS,
        alpha: float = 0.5,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> list[dict]:
        """
        Hybrid search combining FAISS + BM25 via Reciprocal Rank Fusion.

        Args:
            query: the user's question in Vietnamese.
            k: number of final results to return.
            alpha: weight for semantic search (1-alpha = weight for BM25).
                   0.5 = balanced, 0.7 = prefer semantic, 0.3 = prefer keyword.
            rrf_k: RRF smoothing constant (standard = 60).
            candidate_multiplier: fetch k * this from each retriever before fusion.

        Returns:
            List of dicts with keys: rank, aid, law_id, text, score, faiss_score, bm25_score
        """
        n_candidates = k * candidate_multiplier

        # Retrieve from both sources
        faiss_results = self._faiss_search(query, n_candidates)
        bm25_results = self._bm25_search(query, n_candidates)

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        faiss_score_map: dict[int, float] = {}
        bm25_score_map: dict[int, float] = {}

        for rank, (idx, score) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (rrf_k + rank)
            faiss_score_map[idx] = score

        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (rrf_k + rank)
            bm25_score_map[idx] = score

        # Sort by fused RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Build result dicts
        results = []
        for rank, (idx, rrf_score) in enumerate(ranked, start=1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "aid": row.get("aid", "?"),
                "law_id": row.get("law_id", "?"),
                "text": row.get("text", ""),
                "score": round(rrf_score, 6),
                "faiss_score": round(faiss_score_map.get(idx, 0.0), 4),
                "bm25_score": round(bm25_score_map.get(idx, 0.0), 4),
            })
        return results
