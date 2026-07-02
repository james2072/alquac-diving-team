"""
embedder.py – Build or load the embedding index for the law corpus.

Mirrors the notebook's "Embedding our text chunks" section but reads from
JSON instead of a PDF.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

from config import (
    CHUNK_MIN_TOKENS,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDINGS_SAVE,
    CORPUS_JSON,
)
from corpus_loader import load_law_corpus


# ---------------------------------------------------------------------------
# Embedding model (singleton)
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
        _embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_embeddings(
    corpus_json: Path = CORPUS_JSON,
    save_path: Path = EMBEDDINGS_SAVE,
    min_tokens: int = CHUNK_MIN_TOKENS,
    batch_size: int = 32,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load corpus → filter short chunks → embed → save Parquet.

    Parquet stores the embedding column as a native binary array — no string
    parsing on load, ~5-10× smaller file, ~15× faster to read than CSV.

    Returns:
        df         : DataFrame with columns [law_id, law_db_id, aid, text, …, embedding]
        embeddings : (N, D) torch.Tensor on the embedding device
    """
    save_path = Path(save_path)

    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading cached embeddings from {save_path}")
        return _load_embeddings(save_path)

    # 1. Load corpus
    print(f"[INFO] Loading corpus from {corpus_json}")
    chunks = load_law_corpus(corpus_json)
    df = pd.DataFrame(chunks)
    print(f"[INFO] Total articles loaded: {len(df)}")

    # 2. Filter short chunks
    df = df[df["token_count"] > min_tokens].reset_index(drop=True)
    print(f"[INFO] Chunks after min-token filter ({min_tokens}): {len(df)}")

    # 3. Embed
    model = get_embedding_model()
    texts = df["text"].tolist()

    print(f"[INFO] Embedding {len(texts)} chunks …")
    t0 = time.perf_counter()
    emb_tensor = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    print(f"[INFO] Embedding done in {time.perf_counter() - t0:.1f}s")

    # 4. Store embeddings as native float32 arrays and save as Parquet
    df["embedding"] = [e.tolist() for e in emb_tensor.cpu()]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False, engine="pyarrow")
    print(f"[INFO] Saved to {save_path}")

    # Move tensor to device
    embeddings = emb_tensor.to(EMBEDDING_DEVICE)
    return df, embeddings


def _load_embeddings(save_path: Path) -> tuple[pd.DataFrame, torch.Tensor]:
    """Read cached Parquet and reconstruct the embedding tensor.

    Parquet stores the embedding column as a native list-of-floats (no string
    parsing needed), so loading is fast and type-safe.
    """
    df = pd.read_parquet(save_path, engine="pyarrow")
    # embedding column comes back as list[float] — convert directly to tensor
    embeddings = torch.tensor(df["embedding"].tolist(), dtype=torch.float32).to(EMBEDDING_DEVICE)
    print(f"[INFO] Loaded {len(df)} embeddings (shape {embeddings.shape})")
    return df, embeddings


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer | None = None,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Embed *query* and return top-k (scores, indices) via dot-product search.
    """
    if model is None:
        model = get_embedding_model()

    query_emb = model.encode(query, convert_to_tensor=True).to(EMBEDDING_DEVICE)
    dot_scores = util.dot_score(query_emb, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=min(k, len(embeddings)))
    return scores, indices
