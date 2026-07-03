import json
import os
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

from configs.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

API_URL = "https://alqac-api.ngrok.pro/retrieve"
CACHE_FILE = PROJECT_ROOT / "data" / "cache" / "case_evidence_cache.json"


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


def get_case_evidence(case_id: str, query: str, api_key: str | None = None) -> list[dict | str]:
    """
    Retrieve case evidence results (containing chunk_id, text, score) for a given case using disk cache.
    Returns a list of result dicts, e.g. [{"chunk_id": "...", "text": "...", "score": 0.886}].
    """
    cid = str(case_id).strip()
    cache = _load_cache()

    # 1. Check local disk cache first (0 penalty, 0 delay)
    if cid in cache:
        cached = cache[cid]
        # Nếu cache đã lưu dict có chứa 'text', trả về luôn
        if cached and isinstance(cached[0], dict) and "text" in cached[0]:
            return cached
        # Nếu cache cũ chỉ lưu list[str], tiếp tục gọi API bên dưới để lấy đầy đủ text và score

    # 2. Get API Token
    token = api_key or os.getenv("ALQAC_TOKEN") or os.getenv("ALQAC_API_KEY") or os.getenv("X_API_KEY")
    if not token:
        # Nếu chưa cấu hình token, trả về cache cũ (nếu có) hoặc rỗng
        return cache.get(cid, [])

    # 3. Call API with rate-limit handling (max 1 req per 5s)
    headers = {
        "X-API-Key": token.strip(),
        "Content-Type": "application/json",
    }
    payload = {
        "query": query[:200],  # Gửi keywords trọng tâm
        "case_id": cid,
    }

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
                return cache.get(cid, [])
            
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            # Cache lại toàn bộ object results (gồm chunk_id, text, score) vào disk
            cache[cid] = results
            _save_cache(cache)

            # Nghỉ 5.1s sau mỗi lần gọi network thành công đúng luật 1 req/5s của BTC
            time.sleep(5.1)
            return results

        except Exception as e:
            print(f"  [WARN] Lỗi gọi API Retrieval cho {cid} (lần {attempt+1}/{max_retries}): {e}")
            time.sleep(5)

    return cache.get(cid, [])
