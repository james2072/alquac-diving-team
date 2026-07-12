"""
utils.py – Shared utility functions for text cleaning, JSON extraction, and formatting.

This module provides clean, reusable helpers to avoid code duplication across
retrieval and submission pipelines.
"""
from __future__ import annotations

import json
import re
from typing import Any


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





