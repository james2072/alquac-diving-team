"""
indexer.py – Build FAISS vector index and BM25 keyword index from Parquet.

Reads cached Parquet embeddings and builds:
    1. FAISS IndexFlatIP (vector similarity) → data/output/law.faiss
    2. BM25Okapi (lexical matching)         → data/output/law_bm25.pkl

These indexes power Reciprocal Rank Fusion (RRF) in the HybridRetriever.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from configs.config import (
    BM25_INDEX,
    EMBEDDING_MODEL,
    EMBEDDINGS_SAVE,
    FAISS_INDEX,
)


def build_faiss_index(
    embeddings: np.ndarray,
    save_path: Path = FAISS_INDEX,
) -> faiss.IndexFlatIP:
    """Create and persist a FAISS IndexFlatIP (inner-product / cosine similarity)."""
    t0 = time.perf_counter()
    emb = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(emb)  # L2 normalization ensures inner product equals cosine similarity

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_path))
    dt = time.perf_counter() - t0
    print(f"[INDEX] FAISS index built: {index.ntotal} vectors, dim={dim} ({dt:.2f}s)")
    print(f"[INDEX] Saved to {save_path}")
    return index


def load_faiss_index(save_path: Path = FAISS_INDEX) -> faiss.IndexFlatIP:
    """Load a persisted FAISS index from disk."""
    index = faiss.read_index(str(save_path))
    print(f"[INDEX] FAISS index loaded: {index.ntotal} vectors")
    return index


def _tokenize_vietnamese(text: str) -> list[str]:
    """
    Tokenize Vietnamese text for BM25 lexical indexing.
    
    Uses `underthesea.word_tokenize` for accurate word segmentation if available;
    otherwise falls back to regex-based word splitting.
    """
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text.lower(), format="text")
        tokens = re.split(r"\s+", segmented)
        return [t for t in tokens if t and re.search(r"\w", t)]
    except ImportError:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in cleaned.split() if t]


def build_bm25_index(
    texts: list[str],
    save_path: Path = BM25_INDEX,
) -> BM25Okapi:
    """Create and persist a BM25Okapi lexical index."""
    t0 = time.perf_counter()
    tokenized = [_tokenize_vietnamese(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)
    dt = time.perf_counter() - t0
    print(f"[INDEX] BM25 index built: {len(texts)} documents ({dt:.2f}s)")
    print(f"[INDEX] Saved to {save_path}")
    return bm25


def load_bm25_index(save_path: Path = BM25_INDEX) -> BM25Okapi:
    """Load a persisted BM25 index from disk."""
    with open(save_path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"[INDEX] BM25 index loaded: {bm25.corpus_size} documents")
    return bm25


def _file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for sync detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    """Read index metadata JSON."""
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    """Write index metadata JSON."""
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_all_indexes(
    parquet_path: Path = EMBEDDINGS_SAVE,
    faiss_path: Path = FAISS_INDEX,
    bm25_path: Path = BM25_INDEX,
    force: bool = False,
) -> tuple[faiss.IndexFlatIP, BM25Okapi, pd.DataFrame]:
    """
    Load Parquet embeddings and build or load FAISS and BM25 indexes.
    
    Automatically rebuilds indexes if the underlying Parquet file hash has changed.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}\n"
            "Run `python -m rag_runner.build_index` first to generate embeddings."
        )

    print(f"[INDEX] Reading Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"[INDEX] Loaded {len(df)} articles")

    meta_path = faiss_path.with_suffix(".meta.json")
    parquet_hash = _file_hash(parquet_path)
    meta = _read_meta(meta_path) or {}
    hash_mismatch = meta.get("parquet_hash") != parquet_hash

    if hash_mismatch and not force:
        print("[INDEX] WARNING: Parquet hash changed — automatically rebuilding indexes.")

    need_rebuild = force or hash_mismatch

    if faiss_path.exists() and not need_rebuild:
        faiss_idx = load_faiss_index(faiss_path)
    else:
        embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
        faiss_idx = build_faiss_index(embeddings, faiss_path)

    if bm25_path.exists() and not need_rebuild:
        bm25_idx = load_bm25_index(bm25_path)
    else:
        bm25_idx = build_bm25_index(df["text"].tolist(), bm25_path)

    if need_rebuild:
        new_meta = {
            "parquet_hash": parquet_hash,
            "created": datetime.datetime.utcnow().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "num_chunks": len(df),
        }
        _write_meta(meta_path, new_meta)
        print(f"[INDEX] Successfully saved metadata to {meta_path}")

    return faiss_idx, bm25_idx, df
