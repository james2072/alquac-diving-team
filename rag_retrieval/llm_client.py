"""
llm_client.py – OpenAI-compatible LLM client wrapper.

All supported endpoints (Google AI Studio, OpenAI, Ollama, LM Studio, etc.)
speak the standard OpenAI Chat Completions protocol. Switch providers simply by
changing LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env without code changes.
"""
from __future__ import annotations

import random
import time
from typing import Any, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel

from configs.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CHAT_TEMPERATURE,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_SLEEP_SUCCESS,
)

T = TypeVar("T", bound=BaseModel)

# Singleton client instance re-used across calls
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-load the OpenAI client singleton."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def chat(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_CHAT_TEMPERATURE,
) -> str:
    """
    Send a prompt to the configured LLM endpoint and return the text response.
    
    Includes automatic retry with exponential backoff and jitter for 503 (High Demand),
    429 (Rate Limit), and temporary network unavailability errors.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            res = response.choices[0].message.content or ""
            # Brief pause after successful requests to avoid rate limits
            time.sleep(LLM_RETRY_SLEEP_SUCCESS)
            return res
        except Exception as e:
            err_str = str(e)
            is_transient = any(code in err_str for code in ["503", "429", "UNAVAILABLE"])
            
            if is_transient and attempt < LLM_MAX_RETRIES - 1:
                base_wait = min(60, (2 ** attempt) * 4)  # 4s, 8s, 16s, 32s, 60s...
                jitter = random.uniform(1, 4)
                wait_time = base_wait + jitter
                print(
                    f"\n  [LLM RETRY {attempt + 1}/{LLM_MAX_RETRIES}] "
                    f"Server temporarily busy or rate limited. Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
            else:
                if attempt == LLM_MAX_RETRIES - 1:
                    raise e
    return ""


def chat_structured(
    prompt: str,
    response_format: Type[T],
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_CHAT_TEMPERATURE,
) -> T | None:
    """
    Send a prompt and enforce strict Zod-like schema output matching the Pydantic response_format.
    
    Uses OpenAI/Gemini Structured Outputs (`client.beta.chat.completions.parse`) to guarantee
    exact JSON schema adherence at the token generation level without manual regex parsing.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = _get_client().beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            res = response.choices[0].message.parsed
            time.sleep(LLM_RETRY_SLEEP_SUCCESS)
            return res
        except Exception as e:
            err_str = str(e)
            is_transient = any(code in err_str for code in ["503", "429", "UNAVAILABLE"])
            
            if is_transient and attempt < LLM_MAX_RETRIES - 1:
                base_wait = min(60, (2 ** attempt) * 4)
                jitter = random.uniform(1, 4)
                wait_time = base_wait + jitter
                print(
                    f"\n  [LLM RETRY {attempt + 1}/{LLM_MAX_RETRIES}] "
                    f"Server temporarily busy or rate limited. Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
            else:
                if attempt == LLM_MAX_RETRIES - 1:
                    raise e
    return None


class LMStudioClient:
    """Compatible wrapper class for legacy code requiring an instantiated client object."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generate response from structured chat messages."""
        system = ""
        prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            elif msg.get("role") == "user":
                prompt = msg.get("content", "")
        return chat(prompt=prompt, system=system, **kwargs)
