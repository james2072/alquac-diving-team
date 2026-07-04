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
MIN_SCORE = 4.0


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
    """Sinh danh sách câu truy vấn đa chiều (vai nguyên đơn, bị đơn, chứng cứ, HĐXX, quyết định)."""
    names = re.findall(
        r'(?:Ông|Bà|Anh|Chị|Cụ)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){0,3})',
        case_query,
    )
    dispute_match = re.search(r'tranh chấp\s+([^\.]+)', case_query, re.IGNORECASE)
    dispute = dispute_match.group(1).strip() if dispute_match else case_query[:80]

    plaintiff = names[0] if len(names) > 0 else "nguyên đơn"
    defendant = names[1] if len(names) > 1 else "bị đơn"

    queries = [
        case_query[:200],                                              # nguyên văn truy vấn
        f"yêu cầu khởi kiện của {plaintiff} về {dispute}"[:200],        # yêu cầu nguyên đơn
        f"ý kiến trình bày của {defendant} về {dispute}"[:200],         # ý kiến bị đơn
        f"lời khai, chứng cứ, tài liệu liên quan đến {dispute}"[:200],  # chứng cứ
        f"nhận định của Hội đồng xét xử về {dispute}"[:200],            # phần nhận định tòa
        f"quyết định của Tòa án về {dispute}"[:200],                    # phần quyết định
    ]
    return queries[:MAX_CALLS_PER_CASE]


def get_case_evidence(case_id: str, query: str, api_key: str | None = None, multi_query: bool = True) -> list[dict | str]:
    """
    Retrieve case evidence results (containing chunk_id, text, score) for a given case using disk cache.
    Nếu chưa có trong cache, sử dụng chiến lược truy vấn đa chiều + Early Stop để thu thập đủ góc nhìn vụ án.
    """
    cid = str(case_id).strip()
    cache = _load_cache()

    # 1. Check local disk cache first (0 penalty, 0 delay)
    if cid in cache:
        cached = cache[cid]
        # Nếu cache lưu format dict mới {"multi_query": True, "results": [...]}
        if isinstance(cached, dict) and "results" in cached:
            return cached["results"]
        # Nếu cache lưu list dict có text từ trước, dùng luôn
        if isinstance(cached, list) and cached and isinstance(cached[0], dict) and "text" in cached[0]:
            return cached

    # 2. Get API Token
    token = api_key or os.getenv("ALQAC_TOKEN") or os.getenv("ALQAC_API_KEY") or os.getenv("X_API_KEY")
    if not token:
        cached = cache.get(cid, [])
        return cached.get("results", []) if isinstance(cached, dict) else cached

    # 3. Call API với chiến lược đa chiều (multi-query) + Early Stop
    headers = {
        "X-API-Key": token.strip(),
        "Content-Type": "application/json",
    }

    queries = build_diverse_queries(query) if multi_query else [query[:200]]
    evidence_segments = []
    seen_chunk_ids = set()
    low_score_streak = 0

    for i, q in enumerate(queries):
        payload = {"query": q, "case_id": cid}
        max_retries = 3
        success_chunk = False

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
                if results and isinstance(results[0], dict):
                    top_res = results[0]
                    chunk_id = top_res.get("chunk_id")
                    score = top_res.get("score", 0.0) or 0.0

                    if chunk_id and chunk_id not in seen_chunk_ids:
                        evidence_segments.append(top_res)
                        seen_chunk_ids.add(chunk_id)
                        low_score_streak = 0
                    else:
                        low_score_streak += 1

                    if score < MIN_SCORE:
                        low_score_streak += 1
                else:
                    low_score_streak += 1

                success_chunk = True
                if i < len(queries) - 1:
                    time.sleep(5.1)
                break

            except Exception as e:
                print(f"  [WARN] Lỗi gọi API Retrieval cho {cid} query {i+1} (lần {attempt+1}/{max_retries}): {e}")
                time.sleep(4)

        if low_score_streak >= 2 and i >= 2:
            break

    # Cache lại toàn bộ object results vào disk
    cache[cid] = evidence_segments if not multi_query else {"multi_query": True, "results": evidence_segments}
    _save_cache(cache)
    return evidence_segments
