"""
utils.py – Shared utility functions for text cleaning, JSON extraction, and formatting.

This module provides clean, reusable helpers to avoid code duplication across
retrieval and submission pipelines.
"""
from __future__ import annotations

import json
import re
from typing import Any

from configs.config import (
    RULE_KEYWORDS_ACCEPT_ALL,
    RULE_KEYWORDS_PARTIAL,
    RULE_KEYWORDS_REJECT_ALL,
)


def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks produced by reasoning LLMs.
    """
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def clean_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters and newlines into single spaces for clean text processing.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_json_from_text(text: str) -> Any | None:
    """
    Locate and parse a valid JSON object or array from LLM output text.
    
    Supports both JSON dicts ({...}) and arrays ([...]), handling markdown code blocks seamlessly.
    Returns None if no valid JSON object or array can be extracted.
    """
    if not text:
        return None

    # Step 1: Check for markdown JSON code blocks (e.g., ```json ... ```)
    md_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, flags=re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Step 2 & 3: Try outermost balanced extract, then scan inner balanced pairs
    for open_char, close_char in [("[", "]"), ("{", "}")]:
        start_idx = text.find(open_char)
        end_idx = text.rfind(close_char)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx : end_idx + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        pos = 0
        while True:
            start = text.find(open_char, pos)
            if start == -1:
                break
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_char:
                    depth += 1
                elif text[i] == close_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except (json.JSONDecodeError, TypeError):
                            break
            pos = start + 1

    return None


def rule_override(text: str) -> str | None:
    """
    Apply deterministic keyword rules and negation filtering to classify court decisions.
    
    Returns A_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN, B_WIN, or None if undecided.
    """
    if not text:
        return None
    t = text.lower()

    has_reject_all = any(kw in t for kw in RULE_KEYWORDS_REJECT_ALL)
    has_accept_all = any(
        kw in t
        and f"không {kw}" not in t
        and f"chưa {kw}" not in t
        and f"để {kw}" not in t
        and f"cứ {kw}" not in t
        for kw in RULE_KEYWORDS_ACCEPT_ALL
    )
    has_partial = any(kw in t for kw in RULE_KEYWORDS_PARTIAL)

    if has_reject_all and not has_accept_all and not has_partial:
        return "B_WIN"
    if has_accept_all and not has_reject_all and not has_partial:
        return "A_WIN"
    if has_partial or (has_accept_all and has_reject_all):
        accept_count = sum(1 for kw in ["chấp nhận"] if kw in t)
        reject_count = sum(
            1
            for kw in ["không chấp nhận", "bác", "không có căn cứ", "chưa có căn cứ"]
            if kw in t
        )
        return "PARTIAL_A_WIN" if accept_count > reject_count else "PARTIAL_B_WIN"

    return None

