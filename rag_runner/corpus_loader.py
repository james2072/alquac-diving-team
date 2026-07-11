"""
corpus_loader.py – Load and pre-process the Vietnamese legal JSON corpus.

Long articles exceeding configured token limits are automatically split into
overlapping windows to preserve semantic context and improve embedding quality.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from transformers import AutoTokenizer

from configs.config import CHUNK_STRIDE, EMBEDDING_MODEL, MAX_CHUNK_TOKENS
from rag_retrieval.utils import clean_whitespace

# Singleton tokenizer matching the embedding model
_tokenizer: AutoTokenizer | None = None


def _get_tokenizer() -> AutoTokenizer:
    """Lazy-load the tokenizer matching the configured embedding model."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def _token_count(text: str) -> int:
    """Calculate accurate token count using the embedding model tokenizer."""
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_long_text(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    stride: int = CHUNK_STRIDE,
) -> list[str]:
    """
    Split long text into overlapping windows of at most `max_tokens` tokens.
    
    Preserves context across chunk boundaries using `stride` overlap.
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
        if start + max_tokens >= len(token_ids):
            break

    return chunks


def load_law_corpus(
    json_path: Path,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
    chunk_stride: int = CHUNK_STRIDE,
) -> list[dict[str, Any]]:
    """
    Parse legal corpus JSON and return a flat list of article chunks.
    
    Long articles are split into overlapping windows to optimize downstream
    FAISS vector retrieval quality.
    """
    with open(json_path, encoding="utf-8") as f:
        corpus: list[dict[str, Any]] = json.load(f)

    chunks: list[dict[str, Any]] = []
    for law in corpus:
        law_id = str(law.get("law_id", "unknown")).strip()
        law_db_id = int(law.get("id", -1))
        
        for article in law.get("content", []):
            raw_text = str(article.get("content_Article", "")).strip()
            if not raw_text:
                continue
            
            text = clean_whitespace(raw_text)
            windows = _split_long_text(text, max_chunk_tokens, chunk_stride)

            for part_idx, window_text in enumerate(windows):
                tc = _token_count(window_text)
                chunks.append({
                    "law_id":      law_id,
                    "law_db_id":   law_db_id,
                    "aid":         int(article.get("aid", -1)),
                    "text":        window_text,
                    "char_count":  len(window_text),
                    "word_count":  len(window_text.split()),
                    "token_count": tc,
                    "chunk_part":  part_idx,
                })

    return chunks


def iter_law_corpus(json_path: Path) -> Iterator[dict[str, Any]]:
    """Memory-efficient generator yielding processed legal articles."""
    with open(json_path, encoding="utf-8") as f:
        corpus: list[dict[str, Any]] = json.load(f)

    for law in corpus:
        law_id = str(law.get("law_id", "unknown")).strip()
        law_db_id = int(law.get("id", -1))
        
        for article in law.get("content", []):
            raw_text = str(article.get("content_Article", "")).strip()
            if not raw_text:
                continue
            
            text = clean_whitespace(raw_text)
            yield {
                "law_id":      law_id,
                "law_db_id":   law_db_id,
                "aid":         int(article.get("aid", -1)),
                "text":        text,
                "char_count":  len(text),
                "word_count":  len(text.split()),
                "token_count": _token_count(text),
            }
