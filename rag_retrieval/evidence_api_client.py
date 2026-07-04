import json
import os
import re
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

from configs.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

API_URL = "https://alqac-api.ngrok.pro/retrieve"
CACHE_FILE = PROJECT_ROOT / "data" / "cache" / "case_evidence_cache.json"
MAX_CALLS_PER_CASE = 6
MIN_SCORE = 2.0


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_diverse_queries(case_query: str) -> list[str]:
    """Sinh các câu truy vấn súc tích, bao quát toàn bộ diễn biến và phán quyết vụ án (tối đa 3 calls để không bị phạt E_i)."""
    q_clean = case_query[:250].strip()
    return [
        q_clean,                                                        # Nguyên văn truy vấn
        f"lời khai chứng cứ nguyên đơn bị đơn tranh chấp {q_clean[:120]}",  # Diễn biến & chứng cứ
        f"nhận định của Hội đồng xét xử và quyết định của Tòa án {q_clean[:120]}",  # Phán quyết & nhận định
    ]


def get_case_evidence(case_id: str, query: str, api_key: str | None = None, multi_query: bool = True) -> list[dict | str]:
    """
    Retrieve case evidence results (containing chunk_id, text, score) for a given case using disk cache.
    Thu thập toàn bộ các chunk có liên quan từ API mà không phức tạp hóa hay cắt xén kết quả.
    """
    cid = str(case_id).strip()
    cache = _load_cache()

    # 1. Check local disk cache first (0 penalty, 0 delay)
    if cid in cache:
        cached = cache[cid]
        if isinstance(cached, dict) and "results" in cached:
            return cached["results"]
        if isinstance(cached, list) and cached and isinstance(cached[0], dict) and "text" in cached[0]:
            return cached

    # 2. Get API Token
    token = api_key or os.getenv("ALQAC_TOKEN") or os.getenv("ALQAC_API_KEY") or os.getenv("X_API_KEY")
    if not token:
        cached = cache.get(cid, [])
        return cached.get("results", []) if isinstance(cached, dict) else cached

    # 3. Call API với tối đa 3 câu truy vấn cô đọng (đảm bảo c_i <= 2*n_i, không bị phạt hiệu năng)
    headers = {
        "X-API-Key": token.strip(),
        "Content-Type": "application/json",
    }

    queries = build_diverse_queries(query) if multi_query else [query[:250]]
    evidence_segments = []
    seen_chunk_ids = set()

    for i, q in enumerate(queries):
        payload = {"query": q, "case_id": cid}
        max_retries = 3

        for attempt in range(max_retries):
            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                if resp.status_code == 429:
                    print(f"  [API 429] Rate limit exceeded cho {cid}. Đang chờ 5.5s để retry...")
                    time.sleep(5.5)
                    continue
                elif resp.status_code == 403:
                    print(f"  [API 403] Token X-API-Key không hợp lệ hoặc hết hạn cho {cid}.")
                    cached = cache.get(cid, [])
                    return cached.get("results", []) if isinstance(cached, dict) else cached

                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                
                # QUAN TRỌNG: Lấy TOÀN BỘ các chunk hợp lệ do API trả về (thay vì chỉ lấy mỗi results[0] như code cũ)
                for res in results:
                    if isinstance(res, dict):
                        chunk_id = res.get("chunk_id")
                        score = res.get("score", 0.0) or 0.0
                        if chunk_id and chunk_id not in seen_chunk_ids and score >= MIN_SCORE:
                            evidence_segments.append(res)
                            seen_chunk_ids.add(chunk_id)

                if i < len(queries) - 1:
                    time.sleep(4.5)
                break

            except Exception as e:
                print(f"  [WARN] Lỗi gọi API Retrieval cho {cid} query {i+1} (lần {attempt+1}/{max_retries}): {e}")
                time.sleep(3)

    # Cache lại toàn bộ object results vào disk
    cache[cid] = evidence_segments if not multi_query else {"multi_query": True, "results": evidence_segments}
    _save_cache(cache)
    return evidence_segments
