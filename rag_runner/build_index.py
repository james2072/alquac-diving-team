"""
build_index.py – One-time script to pre-compute and cache embeddings.

Run this FIRST before querying:
    python build_index.py

Options:
    --force   Rebuild even if cached embeddings file already exists.
    --corpus  Path to the law JSON file (overrides .env CORPUS_JSON).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import CORPUS_JSON, EMBEDDINGS_SAVE
from embedder import build_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build law corpus embeddings.")
    parser.add_argument("--force",  action="store_true", help="Force rebuild.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_JSON,
                        help="Path to corpus JSON file.")
    parser.add_argument("--save",   type=Path, default=EMBEDDINGS_SAVE,
                        help="Where to save the embeddings CSV.")
    args = parser.parse_args()

    df, embeddings = build_embeddings(
        corpus_json=args.corpus,
        save_path=args.save,
        force_rebuild=args.force,
    )
    print(f"\n✅  Index ready — {len(df)} chunks, embedding shape {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
