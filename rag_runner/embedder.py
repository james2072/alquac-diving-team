"""
embedder.py – Build or load the embedding index for the legal corpus.

Embeds text chunks using the configured SentenceTransformer model and stores
them as high-performance, type-safe Parquet files.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from configs.config import (
    CHUNK_MIN_TOKENS,
    CORPUS_JSON,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDINGS_SAVE,
)
from rag_runner.corpus_loader import load_law_corpus

# Singleton embedding model
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
        _embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
        )
    return _embedding_model


def build_embeddings(
    corpus_json: Path = CORPUS_JSON,
    save_path: Path = EMBEDDINGS_SAVE,
    min_tokens: int = CHUNK_MIN_TOKENS,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load corpus, filter short chunks, compute embeddings, and save as Parquet.
    
    Storing embeddings in Parquet format provides type-safe binary storage,
    significantly faster read speeds, and smaller file sizes compared to CSV.
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
    print(f"[INFO] Chunks after minimum token filter ({min_tokens}): {len(df)}")

    # 3. Compute embeddings
    model = get_embedding_model()
    texts = df["text"].tolist()

    print(f"[INFO] Embedding {len(texts)} chunks...")
    t0 = time.perf_counter()
    emb_tensor = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    print(f"[INFO] Embedding completed in {time.perf_counter() - t0:.1f}s")

    # 4. Save to Parquet
    df["embedding"] = list(emb_tensor.cpu().numpy())
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False, engine="pyarrow")
    print(f"[INFO] Successfully saved embeddings to {save_path}")

    embeddings = emb_tensor.to(EMBEDDING_DEVICE)
    return df, embeddings


def _load_embeddings(save_path: Path) -> tuple[pd.DataFrame, torch.Tensor]:
    """Read cached Parquet file and reconstruct the embedding tensor."""
    df = pd.read_parquet(save_path, engine="pyarrow")
    embeddings = torch.tensor(
        np.array(df["embedding"].tolist()), dtype=torch.float32
    ).to(EMBEDDING_DEVICE)
    print(f"[INFO] Loaded {len(df)} embeddings (shape {embeddings.shape})")
    return df, embeddings


def retrieve_top_k(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer | None = None,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dot-product vector similarity between query and corpus embeddings."""
    if model is None:
        model = get_embedding_model()

    query_emb = model.encode(query, convert_to_tensor=True).to(EMBEDDING_DEVICE)
    dot_scores = util.dot_score(query_emb, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=min(k, len(embeddings)))
    return scores, indices
