"""
evidence_api_client.py – Client for retrieving case evidence from the competition API or disk cache.

Handles multi-query generation, rate-limit backoff (429/403), and local disk caching
to ensure zero latency penalty on repeated queries.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any
import requests

from configs.config import (
    API_URL,
    CACHE_FILE,
    MAX_CALLS_PER_CASE,
    MAX_QUERY_LENGTH,
    MAX_RETRIES,
    MIN_SCORE,
    REQUEST_TIMEOUT,
    RETRY_DELAY_429,
    RETRY_DELAY_ERROR,
    RETRY_DELAY_NORMAL,
    SUB_QUERY_LENGTH,
)
from rag_retrieval.llm_client import chat
from rag_retrieval.utils import extract_json_from_text, strip_think_tags


def _load_cache() -> dict[str, Any]:
    """Load cached case evidence from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist case evidence cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_cached_case_queries(case_id: str) -> list[str]:
    """Retrieve cached investigative queries generated for a case from disk cache."""
    cid = str(case_id).strip()
    cache = _load_cache()
    if cid in cache and isinstance(cache[cid], dict):
        queries = cache[cid].get("queries", [])
        if isinstance(queries, list):
            return [str(q) for q in queries if q]
    return []



def build_diverse_queries(
    case_query: str, use_llm: bool = True, case_fact: str = "", case: dict[str, Any] | None = None
) -> list[str]:
    """
    Generate exactly 2 queries (Dual-Anchor Retrieval) for BM25-based API retrieval.
    - Query 1: case_query (captures fact/context section of the case document)
    - Query 2: first ~300 chars of court_reasoning (query-by-example – matches reasoning
               section chunks via shared BM25 vocabulary, no LLM needed, deterministic)
    Guarantees c_i = 2 <= 2*n_i -> E_i = 1.0 (no API efficiency penalty).
    """
    q_clean = case_query.strip()
    queries = [q_clean]

    # Query-by-example: use the court reasoning text itself as the BM25 anchor.
    # The reasoning section shares exact legal vocabulary with the evidence chunks
    # that contain the court's analysis – far more precise than a generic template.
    court_reasoning = str(case.get("court_reasoning", "")).strip() if case else ""
    if court_reasoning:
        reasoning_anchor = court_reasoning[:MAX_QUERY_LENGTH]
        if reasoning_anchor and reasoning_anchor not in queries:
            queries.append(reasoning_anchor)

    # Return exactly 2 queries -> 2 API calls per case (E_i = 1.0)
    return queries[:2]


def _resolve_api_token(api_key: str | None = None) -> str | None:
    """Resolve API token from arguments or environment variables."""
    token = (
        api_key
        or os.getenv("ALQAC_TOKEN")
        or os.getenv("ALQAC_API_KEY")
        or os.getenv("X_API_KEY")
    )
    return token.strip() if token else None


def _fetch_from_api_with_retries(
    url: str, headers: dict[str, str], payload: dict[str, str], case_id: str, query_idx: int
) -> list[dict[str, Any]]:
    """Execute API POST request with exponential backoff and rate-limit handling."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            
            if resp.status_code == 429:
                backoff = RETRY_DELAY_429 * (2 ** attempt)
                print(f"  [API 429] Rate limit exceeded for case {case_id}. Waiting {backoff:.1f}s before retry...")
                time.sleep(backoff)
                continue
            
            if resp.status_code == 403:
                print(f"  [API 403] Invalid or expired X-API-Key token for case {case_id}.")
                return []
                
            if resp.status_code >= 500:
                backoff = RETRY_DELAY_ERROR * (2 ** attempt)
                print(f"  [API {resp.status_code}] Server error for case {case_id}. Exponential backoff {backoff:.1f}s...")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])

        except requests.RequestException as e:
            backoff = RETRY_DELAY_ERROR * (2 ** attempt)
            print(f"  [WARN] Retrieval API error for {case_id} (query {query_idx + 1}, attempt {attempt + 1}/{MAX_RETRIES}): {e}. Waiting {backoff:.1f}s...")
            time.sleep(backoff)
            
    return []


def _filter_and_deduplicate_chunks(
    results: list[Any], seen_chunk_ids: set[str]
) -> list[dict[str, Any]]:
    """Filter API results by minimum score and deduplicate by chunk ID."""
    valid_chunks: list[dict[str, Any]] = []
    for res in results:
        if isinstance(res, dict):
            chunk_id = res.get("chunk_id")
            score = float(res.get("score", 0.0) or 0.0)
            if chunk_id and chunk_id not in seen_chunk_ids and score >= min(MIN_SCORE, 3.0):
                valid_chunks.append(res)
                seen_chunk_ids.add(chunk_id)
    return valid_chunks


def get_case_evidence(
    case_id: str,
    query: str,
    api_key: str | None = None,
    multi_query: bool = True,
    case_fact: str = "",
    force_refresh: bool = False,
    case: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve case evidence chunks (containing chunk_id, text, score) using disk cache or API.
    
    Collects all relevant chunks without truncation or premature filtering.
    """
    cid = str(case_id).strip()
    cache = _load_cache()

    # Step 1: Check local disk cache first (zero latency, zero penalty)
    if not force_refresh and cid in cache:
        cached = cache[cid]
        if isinstance(cached, dict) and "results" in cached:
            return cached["results"]
        if isinstance(cached, list) and cached and isinstance(cached[0], dict) and "text" in cached[0]:
            return cached

    # Step 2: Resolve API token
    token = _resolve_api_token(api_key)
    if not token:
        cached = cache.get(cid, [])
        return cached.get("results", []) if isinstance(cached, dict) else cached

    # Step 3: Query API with concise queries
    headers = {
        "X-API-Key": token,
        "Content-Type": "application/json",
    }
    queries = build_diverse_queries(query, case_fact=case_fact, case=case) if multi_query else [query[:MAX_QUERY_LENGTH]]
    
    evidence_segments: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for i, q in enumerate(queries):
        payload = {"query": q, "case_id": cid}
        raw_results = _fetch_from_api_with_retries(API_URL, headers, payload, cid, i)
        
        new_chunks = _filter_and_deduplicate_chunks(raw_results, seen_chunk_ids)
        evidence_segments.extend(new_chunks)

        if i < len(queries) - 1:
            time.sleep(RETRY_DELAY_NORMAL)

    # Step 4: Persist retrieved segments and generated queries to cache
    cache[cid] = (
        {"multi_query": True, "queries": queries, "results": evidence_segments}
        if multi_query
        else {"multi_query": False, "queries": queries, "results": evidence_segments}
    )
    _save_cache(cache)
    return evidence_segments
