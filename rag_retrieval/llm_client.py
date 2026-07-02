"""
llm_client.py – Thin OpenAI-compatible wrapper.

All supported endpoints (Google AI Studio, OpenAI, Ollama, LM Studio …)
speak the same OpenAI Chat Completions protocol, so a single client handles
all of them. To switch source, only change LLM_API_KEY / LLM_BASE_URL /
LLM_MODEL in .env — no code change needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add root folder to sys.path to allow importing config from rag_runner
_root_parent = Path(__file__).resolve().parent.parent
if str(_root_parent) not in sys.path:
    sys.path.append(str(_root_parent))

from openai import OpenAI
from rag_runner.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

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
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """
    Send a prompt to the configured endpoint and return the text response.

    Args:
        prompt      – the user message
        system      – optional system instruction
        model       – model name (defaults to LLM_MODEL from .env)
        max_tokens  – upper bound on response length
        temperature – sampling temperature (0 = deterministic)
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""

