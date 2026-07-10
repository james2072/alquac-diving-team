"""
prefetch_cache.py – Automated pre-fetcher for case evidence chunks.

Crawls and caches evidence chunks from the competition API for all test cases ahead of time,
eliminating API latency and rate-limit bottlenecks during final submission generation.
Complies strictly with the 1 request/5s rate limit and incremental checkpointing.

Usage:
    python -m rag_retrieval.prefetch_cache --test-file data/test/ALQAC2026_public_test.json --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from configs.config import CACHE_FILE, RETRY_DELAY_NORMAL, TEST_FILE
from rag_retrieval.evidence_api_client import _load_cache, get_case_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch and cache ALQAC 2026 case evidence.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=TEST_FILE,
        help="Path to the JSON test file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh existing cache entries by re-querying the API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N cases (for debugging).",
    )
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"[ERROR] Test file not found: {args.test_file}")
        sys.exit(1)

    print(f"[INFO] Reading test data from: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if args.limit:
        cases = cases[:args.limit]
        print(f"[INFO] Debug mode: Processing only the first {len(cases)} cases.")

    cache = _load_cache()
    print(f"[INFO] Initial cache contains {len(cache)} cases at {CACHE_FILE}")
    print(f"[INFO] Using rate-limit delay of {RETRY_DELAY_NORMAL}s between queries.")
    print(f"\n[START] Pre-fetching case evidence for {len(cases)} cases...")

    t0 = time.perf_counter()
    fetched_count = 0
    skipped_count = 0

    for i, case in enumerate(cases, 1):
        cid = str(case.get("case_id", f"case_{i}")).strip()
        query = str(case.get("case_query", ""))
        fact = str(case.get("case_fact", ""))

        if not args.force and cid in cache:
            print(f"[{i}/{len(cases)}] Case: {cid} -> [SKIP] Already cached.")
            skipped_count += 1
            continue

        print(f"[{i}/{len(cases)}] Fetching evidence for case: {cid} ...", end=" ", flush=True)
        t_case = time.perf_counter()

        chunks = get_case_evidence(cid, query, case_fact=fact, force_refresh=args.force, case=case)
        
        dt = time.perf_counter() - t_case
        print(f"-> Retrieved {len(chunks)} chunks ({dt:.1f}s)")
        fetched_count += 1

    total_time = time.perf_counter() - t0
    print(f"\n[DONE] Pre-fetch completed in {total_time:.1f}s.")
    print(f"Summary: {fetched_count} cases fetched, {skipped_count} cases skipped.")
    print(f"Total cached cases: {len(_load_cache())} stored in {CACHE_FILE}")


if __name__ == "__main__":
    main()
