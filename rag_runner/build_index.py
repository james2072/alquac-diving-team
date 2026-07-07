"""
build_index.py – One-time script to pre-compute embeddings and build indexes.

Run this script FIRST before querying:
    python -m rag_runner.build_index

This script executes 3 sequential steps:
    1. Embed corpus → save law_embeddings.parquet
    2. Build FAISS vector index → save law.faiss
    3. Build BM25 keyword index → save law_bm25.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from configs.config import BM25_INDEX, CORPUS_JSON, EMBEDDINGS_SAVE, FAISS_INDEX
from rag_runner.embedder import build_embeddings
from rag_runner.indexer import build_all_indexes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build legal corpus embeddings and search indexes."
    )
    parser.add_argument("--force",  action="store_true", help="Force rebuild all indexes.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_JSON,
        help="Path to corpus JSON file.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=EMBEDDINGS_SAVE,
        help="Path to save the generated Parquet embeddings.",
    )
    args = parser.parse_args()

    # Step 1: Build embeddings (Parquet format)
    print("=" * 60)
    print("Building embeddings (Parquet)")
    print("=" * 60)
    df, embeddings = build_embeddings(
        corpus_json=args.corpus,
        save_path=args.save,
        force_rebuild=args.force,
    )
    print(f"✅ Embeddings ready — {len(df)} chunks, shape {tuple(embeddings.shape)}")

    # Step 2 & 3: Build FAISS + BM25 search indexes
    print("\n" + "=" * 60)
    print("Building FAISS + BM25 search indexes")
    print("=" * 60)
    faiss_idx, bm25_idx, _ = build_all_indexes(
        parquet_path=args.save,
        faiss_path=FAISS_INDEX,
        bm25_path=BM25_INDEX,
        force=args.force,
    )

    print("\n" + "=" * 60)
    print("✅ All search indexes built successfully!")
    print(f"    Parquet : {args.save}")
    print(f"    FAISS   : {FAISS_INDEX}")
    print(f"    BM25    : {BM25_INDEX}")
    print("=" * 60)


if __name__ == "__main__":
    main()
