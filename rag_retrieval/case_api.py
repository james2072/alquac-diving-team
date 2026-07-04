from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi


CACHE_FILE = Path("data/cache/case_evidence_cache.json")
ALL_CASES_FILE = Path("data/all_cases.json")


class CaseAPIClient:
    """Cache-only case evidence retriever (no API calls)."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        
        # Load all chunks from cache
        self.all_cases = self._load_all_cases()
        print(f"[CaseAPI] Loaded {len(self.all_cases)} cases from cache (NO API calls)")
    
    def _load_all_cases(self) -> dict:
        """
        Load cases from cache. Try multiple formats:
        1. data/cache/case_evidence_cache.json (query-keyed)
        2. data/all_cases.json (list of cases with results)
        """
        # Format 1: cache file {case_id: {query: {chunk_id, text, score}}}
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                
                # Convert to {case_id: [chunks]}
                all_cases = {}
                for case_id, queries in cache.items():
                    chunks = []
                    seen = set()
                    if isinstance(queries, dict):
                        for query, result in queries.items():
                            if isinstance(result, dict) and result.get("chunk_id") not in seen:
                                chunks.append(result)
                                seen.add(result.get("chunk_id"))
                    all_cases[case_id] = chunks
                print(f"[CaseAPI] Loaded from {CACHE_FILE}")
                return all_cases
            except Exception as e:
                print(f"[CaseAPI] Failed to load {CACHE_FILE}: {e}")
        
        # Format 2: all_cases.json [{case_id, results: [...]}]
        if ALL_CASES_FILE.exists():
            try:
                with open(ALL_CASES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                all_cases = {}
                if isinstance(data, list):
                    for item in data:
                        cid = str(item.get("case_id", "")).strip()
                        results = item.get("results", [])
                        if cid:
                            all_cases[cid] = results
                elif isinstance(data, dict):
                    for cid, item in data.items():
                        if isinstance(item, dict):
                            all_cases[str(cid).strip()] = item.get("results", [])
                        elif isinstance(item, list):
                            all_cases[str(cid).strip()] = item
                
                print(f"[CaseAPI] Loaded from {ALL_CASES_FILE}")
                return all_cases
            except Exception as e:
                print(f"[CaseAPI] Failed to load {ALL_CASES_FILE}: {e}")
        
        print("[CaseAPI] WARNING: No cache file found!")
        return {}
    
    def get_adaptive_evidence(
        self, 
        case: dict, 
        max_chunks: int = 8,
        **kwargs,
    ) -> list[dict]:
        """
        Get evidence chunks for a case using BM25 ranking
        Returns list of evidence chunks sorted by relevance
        """
        cid = str(case.get("case_id", "")).strip()
        query = case.get("case_query", "")
        
        # Get all chunks for this case
        chunks = self.all_cases.get(cid, [])
        
        if not chunks:
            print(f"  [Cache] No chunks for case {cid}")
            return []
        
        # Build BM25 index on chunks
        chunk_texts = [
            c.get("text", "").lower().split() 
            for c in chunks
        ]
        bm25 = BM25Okapi(chunk_texts)
        
        # Score chunks against query
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Add structural keyword bonuses
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").lower()
            # Boost chunks containing verdict-related keywords
            if any(kw in text for kw in ["quyết định", "chấp nhận", "không chấp nhận", "bác", "về án phí"]):
                scores[i] += 5.0
            # Boost chunks with legal citations
            if "điều" in text and any(c.isdigit() for c in text):
                scores[i] += 2.0
        
        # Sort by score and return top-k
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        result = [c for _, c in scored_chunks[:max_chunks]]
        print(f"  [Cache] Selected {len(result)}/{len(chunks)} chunks (BM25 ranked, no API)")
        return result