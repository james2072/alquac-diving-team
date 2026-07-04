"""
corpus_loader.py – Load and pre-process the Vietnamese law JSON corpus.

JSON schema expected:
[
  {
    "id": 33,
    "law_id": "66/2014/QH13",
    "content": [
      {"aid": 819, "content_Article": "bla bla bla"},
      ...
    ]
  },
  ...
]

Each article (content_Article) becomes one or more *chunk* candidates.
Long articles are split into overlapping windows to improve embedding quality.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Tokenizer (singleton) — uses the same model as the embedding pipeline
# ---------------------------------------------------------------------------
_tokenizer: AutoTokenizer | None = None


def _get_tokenizer() -> AutoTokenizer:
    """Lazy-load the tokenizer matching the embedding model."""
    global _tokenizer
    if _tokenizer is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Light clean-up: collapse whitespace, normalise newlines."""
    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _token_count(text: str) -> int:
    """Accurate token count using the embedding model's tokenizer."""
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_long_text(
    text: str,
    max_tokens: int = 512,
    stride: int = 256,
) -> list[str]:
    """
    Split a long text into overlapping windows of at most `max_tokens` tokens.

    If the text fits within `max_tokens`, returns [text] unchanged.
    Otherwise, splits into chunks with `stride` overlap to preserve context
    at chunk boundaries.

    Args:
        text: the cleaned article text.
        max_tokens: maximum tokens per chunk.
        stride: step size between chunk starts.

    Returns:
        List of text chunks.
    """
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return [text]

    chunks: list[str] = []
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start : start + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text.strip())
        # Stop if we've reached the end
        if start + max_tokens >= len(token_ids):
            break

    return chunks


# ---------------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------------

def load_law_corpus(
    json_path: Path,
    max_chunk_tokens: int = 512,
    chunk_stride: int = 256,
) -> list[dict]:
    """
    Parse the law JSON file and return a flat list of article chunks.

    Long articles (> max_chunk_tokens) are split into overlapping windows
    to improve embedding quality for downstream FAISS retrieval.

    Each returned dict has:
        law_id          – e.g. "66/2014/QH13"
        law_db_id       – integer id of the parent law object
        aid             – article id
        text            – cleaned article text (or window thereof)
        char_count      – character count
        word_count      – rough word count
        token_count     – accurate token count (via model tokenizer)
        chunk_part      – 0-based window index (0 if not split)
    """
    with open(json_path, encoding="utf-8") as f:
        corpus: list[dict] = json.load(f)

    chunks: list[dict] = []
    for law in corpus:
        law_id    = law.get("law_id", "unknown")
        law_db_id = law.get("id", -1)
        articles  = law.get("content", [])

        for article in articles:
            raw_text = article.get("content_Article", "")
            if not raw_text:
                continue
            text = _clean_text(raw_text)

            # Split long articles into overlapping windows
            windows = _split_long_text(text, max_chunk_tokens, chunk_stride)

            for part_idx, window_text in enumerate(windows):
                tc = _token_count(window_text)
                chunks.append({
                    "law_id":      law_id,
                    "law_db_id":   law_db_id,
                    "aid":         article.get("aid", -1),
                    "text":        window_text,
                    "char_count":  len(window_text),
                    "word_count":  len(window_text.split()),
                    "token_count": tc,
                    "chunk_part":  part_idx,
                })

    return chunks


def iter_law_corpus(json_path: Path) -> Iterator[dict]:
    """Memory-efficient generator version of load_law_corpus."""
    with open(json_path, encoding="utf-8") as f:
        corpus: list[dict] = json.load(f)

    for law in corpus:
        law_id    = law.get("law_id", "unknown")
        law_db_id = law.get("id", -1)
        for article in law.get("content", []):
            raw_text = article.get("content_Article", "")
            if not raw_text:
                continue
            text = _clean_text(raw_text)
            yield {
                "law_id":    law_id,
                "law_db_id": law_db_id,
                "aid":       article.get("aid", -1),
                "text":      text,
                "char_count":  len(text),
                "word_count":  len(text.split()),
                "token_count": _token_count(text),
            }
