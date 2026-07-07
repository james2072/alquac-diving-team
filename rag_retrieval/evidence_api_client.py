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


def build_diverse_queries(case_query: str, case_fact: str = "") -> list[str]:
    """
    Generate concise, diverse queries covering case facts and rulings.
    
    Returns up to 4 queries to avoid exceeding efficiency penalty thresholds (c_i <= 2 * n_i).
    """
    q_clean = case_query[:MAX_QUERY_LENGTH].strip()
    sub_q = q_clean[:SUB_QUERY_LENGTH]
    queries = [
        q_clean,                                                           # 1. Original query
        f"lời khai chứng cứ nguyên đơn bị đơn tranh chấp {sub_q}",           # 2. Evidence and disputes
        f"nhận định của Hội đồng xét xử và quyết định của Tòa án {sub_q}",  # 3. Rulings and decisions
    ]
    
    if case_fact:
        # Extract core legal dispute arguments (plaintiff claim, defendant counterclaim, court reasoning)
        fact_lines = [line.strip() for line in case_fact.split("\n") if len(line.strip()) > 30]
        
        # Try to find specific plaintiff/defendant/court statements
        plaintiff_line = ""
        defendant_line = ""
        court_line = ""
        
        for line in fact_lines:
            low = line.lower()
            if not plaintiff_line and any(kw in low for kw in ["nguyên đơn trình bày", "yêu cầu khởi kiện", "nguyên đơn đòi"]):
                plaintiff_line = line[:MAX_QUERY_LENGTH].strip()
            elif not defendant_line and any(kw in low for kw in ["bị đơn trình bày", "ý kiến của bị đơn", "phản tố", "bị đơn cho rằng"]):
                defendant_line = line[:MAX_QUERY_LENGTH].strip()
            elif not court_line and any(kw in low for kw in ["tòa án nhận định", "hội đồng xét xử nhận định", "kết quả giám định", "căn cứ"]):
                court_line = line[:MAX_QUERY_LENGTH].strip()
                
        # Append the most informative extracted line that isn't already included
        best_snippet = plaintiff_line or court_line or defendant_line
        if not best_snippet:
            for line in fact_lines:
                if any(kw in line.lower() for kw in ["yêu cầu", "bồi thường", "tranh chấp", "khởi kiện", "hợp đồng", "lời khai"]):
                    best_snippet = line[:MAX_QUERY_LENGTH].strip()
                    break
            else:
                if fact_lines:
                    best_snippet = fact_lines[0][:MAX_QUERY_LENGTH].strip()
                    
        if best_snippet and best_snippet not in queries:
            queries.append(best_snippet)

    return queries


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
            if chunk_id and chunk_id not in seen_chunk_ids and score >= MIN_SCORE:
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
    queries = build_diverse_queries(query, case_fact=case_fact) if multi_query else [query[:MAX_QUERY_LENGTH]]
    
    evidence_segments: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for i, q in enumerate(queries):
        payload = {"query": q, "case_id": cid}
        raw_results = _fetch_from_api_with_retries(API_URL, headers, payload, cid, i)
        
        new_chunks = _filter_and_deduplicate_chunks(raw_results, seen_chunk_ids)
        evidence_segments.extend(new_chunks)

        if i < len(queries) - 1:
            time.sleep(RETRY_DELAY_NORMAL)

    # Step 4: Persist retrieved segments to cache
    cache[cid] = (
        {"multi_query": True, "results": evidence_segments}
        if multi_query
        else evidence_segments
    )
    _save_cache(cache)
    return evidence_segments
