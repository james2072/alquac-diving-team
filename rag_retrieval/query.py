"""
query.py – Interactive CLI to query the law RAG system.

Usage:
    python query.py "Điều kiện để thành lập tổ chức tín dụng là gì?"

Options:
    --k      Number of retrieved articles (default from .env NUM_RESULTS).
    --quiet  Don't print retrieved sources, only the final answer.
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

from configs.config import EMBEDDINGS_SAVE, NUM_RESULTS
from rag_runner.embedder import build_embeddings
from rag_retrieval.rag import answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the law RAG system.")
    parser.add_argument("query", nargs="?", default=None, help="Question to ask.")
    parser.add_argument("--k",     type=int, default=NUM_RESULTS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load (or build) the index
    df, embeddings = build_embeddings()

    # Single query mode
    if args.query:
        result = answer_query(args.query, df, embeddings, k=args.k, verbose=not args.quiet)
        print("\n" + "=" * 60)
        print("ANSWER:\n")
        print(result["answer"])
        return

    # Interactive loop
    print("\nLaw RAG – type your question (or 'quit' to exit)\n")
    while True:
        try:
            query = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        result = answer_query(query, df, embeddings, k=args.k, verbose=not args.quiet)
        print("\n" + "=" * 60)
        print("ANSWER:\n")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
