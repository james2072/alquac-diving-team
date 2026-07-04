"""
llm_client.py – Thin OpenAI-compatible wrapper.

All supported endpoints (Google AI Studio, OpenAI, Ollama, LM Studio …)
speak the same OpenAI Chat Completions protocol, so a single client handles
all of them. To switch source, only change LLM_API_KEY / LLM_BASE_URL /
LLM_MODEL in .env — no code change needed.
"""
from __future__ import annotations

from openai import OpenAI
from configs.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

import random
import time

# Single client instance (re-used across calls)
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def chat(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> str:
    """
    Send a prompt to the configured endpoint and return the text response.
    Includes automatic retry with backoff for 503 (High Demand) & 429 (Rate Limit).
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            res = response.choices[0].message.content or ""
            time.sleep(2)  # Nghỉ 2s sau mỗi lần gọi LLM thành công để giảm tải
            return res
        except Exception as e:
            err_str = str(e)
            if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries - 1:
                base_wait = min(60, (2 ** attempt) * 4)  # 4s, 8s, 16s, 32s, 60s...
                jitter = random.uniform(1, 4)
                wait_time = base_wait + jitter
                print(f"\n  [LLM RETRY {attempt+1}/{max_retries}] Server LLM tạm thời bận hoặc limit ({err_str[:60]}...). Chờ {wait_time:.1f}s thử lại...")
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    raise e
    return ""

