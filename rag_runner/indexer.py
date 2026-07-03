"""
indexer.py – Build FAISS vector index and BM25 keyword index from Parquet.

Reads the cached law_embeddings.parquet and creates:
    1. FAISS IndexFlatIP  → data/output/law.faiss
    2. BM25Okapi pickle   → data/output/law_bm25.pkl

These indexes power the Hybrid Search (RRF) in the retrieval stage.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from configs.config import (
    EMBEDDINGS_SAVE,
    FAISS_INDEX,
    BM25_INDEX,
)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
    save_path: Path = FAISS_INDEX,
) -> faiss.IndexFlatIP:
    """
    Create a FAISS IndexFlatIP (inner-product = cosine after L2 norm).

    Args:
        embeddings: (N, D) float32 array of document embeddings.
        save_path: where to persist the index file.

    Returns:
        The populated FAISS index.
    """
    t0 = time.perf_counter()
    emb = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(emb)  # normalize so IP ≡ cosine similarity

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
    """Load a persisted FAISS index."""
    index = faiss.read_index(str(save_path))
    print(f"[INDEX] FAISS index loaded: {index.ntotal} vectors")
    return index


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def _tokenize_vietnamese(text: str) -> list[str]:
    """
    Simple whitespace tokenizer for Vietnamese.

    For better results, consider using `underthesea` or `pyvi` word segmentation.
    This basic approach still works well for BM25 on legal text because legal
    terms are often multi-syllable words that partially match via unigrams.
    """
    return text.lower().split()


def build_bm25_index(
    texts: list[str],
    save_path: Path = BM25_INDEX,
) -> BM25Okapi:
    """
    Create a BM25Okapi index from document texts.

    Args:
        texts: list of document text strings.
        save_path: where to persist the pickled BM25 object.

    Returns:
        The fitted BM25Okapi instance.
    """
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
    """Load a persisted BM25 index."""
    with open(save_path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"[INDEX] BM25 index loaded: {bm25.corpus_size} documents")
    return bm25


# ---------------------------------------------------------------------------
# Build all indexes from Parquet
# ---------------------------------------------------------------------------

def build_all_indexes(
    parquet_path: Path = EMBEDDINGS_SAVE,
    faiss_path: Path = FAISS_INDEX,
    bm25_path: Path = BM25_INDEX,
    force: bool = False,
) -> tuple[faiss.IndexFlatIP, BM25Okapi, pd.DataFrame]:
    """
    Read the cached Parquet file and build both FAISS + BM25 indexes.

    If index files already exist and force=False, loads from disk instead.

    Returns:
        (faiss_index, bm25_index, dataframe)
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}\n"
            "Run `python -m rag_runner.build_index` first to generate embeddings."
        )

    print(f"[INDEX] Reading Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"[INDEX] Loaded {len(df)} articles")

    # --- FAISS ---
    if faiss_path.exists() and not force:
        faiss_idx = load_faiss_index(faiss_path)
    else:
        embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
        faiss_idx = build_faiss_index(embeddings, faiss_path)

    # --- BM25 ---
    if bm25_path.exists() and not force:
        bm25_idx = load_bm25_index(bm25_path)
    else:
        bm25_idx = build_bm25_index(df["text"].tolist(), bm25_path)

    return faiss_idx, bm25_idx, df
