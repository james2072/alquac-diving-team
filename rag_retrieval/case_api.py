"""
case_api.py – Cache-only case evidence retriever.

Retrieves evidence chunks from local disk cache (case_evidence_cache.json)
without making external API calls or requiring legacy all_cases.json.
Applies BM25 ranking and structural keyword bonuses to select top relevant chunks.
"""
from __future__ import annotations

import json
from typing import Any
from rank_bm25 import BM25Okapi

from configs.config import (
    CACHE_FILE,
    CITATION_KEYWORD_BOOST,
    DEFAULT_MAX_CHUNKS,
    VERDICT_KEYWORD_BOOST,
    VERDICT_KEYWORDS,
)


class CaseAPIClient:
    """Cache-only case evidence retriever (no API calls)."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.all_cases = self._load_all_cases()
        print(f"[CaseAPI] Loaded {len(self.all_cases)} cases from cache (no API calls)")

    def _load_all_cases(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load case evidence chunks from local disk cache (case_evidence_cache.json).
        """
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)

                all_cases: dict[str, list[dict[str, Any]]] = {}
                for case_id, queries in cache.items():
                    chunks: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    if isinstance(queries, dict):
                        for _, result in queries.items():
                            if isinstance(result, dict):
                                chunk_id = result.get("chunk_id")
                                if chunk_id and chunk_id not in seen:
                                    chunks.append(result)
                                    seen.add(chunk_id)
                    all_cases[str(case_id).strip()] = chunks
                
                print(f"[CaseAPI] Successfully loaded from {CACHE_FILE}")
                return all_cases
            except (json.JSONDecodeError, OSError) as e:
                print(f"[CaseAPI] Failed to load {CACHE_FILE}: {e}")
        else:
            print(f"[CaseAPI] WARNING: Cache file not found at {CACHE_FILE}!")

        return {}

    def get_adaptive_evidence(
        self,
        case: dict[str, Any],
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Rank and retrieve evidence chunks for a case using BM25 and structural bonuses.
        
        Returns a list of evidence chunks sorted by descending relevance score.
        """
        cid = str(case.get("case_id", "")).strip()
        query = str(case.get("case_query", ""))
        fact = str(case.get("case_fact", ""))
        combined_query = f"{query} {fact[:2500]}"

        chunks = self.all_cases.get(cid, [])
        if not chunks:
            print(f"  [Cache] No chunks found for case {cid}")
            return []

        # Build BM25 index on candidate chunks
        chunk_texts = [
            str(c.get("text", "")).lower().split() for c in chunks
        ]
        bm25 = BM25Okapi(chunk_texts)

        # Score chunks against combined query and fact tokens
        query_tokens = combined_query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Apply structural keyword bonuses
        for i, chunk in enumerate(chunks):
            text = str(chunk.get("text", "")).lower()
            if any(kw in text for kw in VERDICT_KEYWORDS):
                scores[i] += VERDICT_KEYWORD_BOOST
            if "điều" in text and any(c.isdigit() for c in text):
                scores[i] += CITATION_KEYWORD_BOOST

        # Sort by boosted score and select top-k
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        result = [c for _, c in scored_chunks[:max_chunks]]
        print(f"  [Cache] Selected {len(result)}/{len(chunks)} chunks (BM25 ranked, no API)")
        return result