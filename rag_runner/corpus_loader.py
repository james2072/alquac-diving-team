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

Each article (content_Article) becomes one *chunk* candidate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Light clean-up: collapse whitespace, normalise newlines."""
    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _token_count(text: str) -> float:
    """Rough estimate: 1 token ≈ 4 characters (works for Vietnamese too)."""
    return len(text) / 4


# ---------------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------------

def load_law_corpus(json_path: Path) -> list[dict]:
    """
    Parse the law JSON file and return a flat list of article chunks.

    Each returned dict has:
        law_id          – e.g. "66/2014/QH13"
        law_db_id       – integer id of the parent law object
        aid             – article id
        text            – cleaned article text
        char_count      – character count
        word_count      – rough word count
        token_count     – rough token count
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
            chunks.append({
                "law_id":    law_id,
                "law_db_id": law_db_id,
                "aid":       article.get("aid", -1),
                "text":      text,
                "char_count":  len(text),
                "word_count":  len(text.split()),
                "token_count": _token_count(text),
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
