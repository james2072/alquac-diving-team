"""
rag.py – Retrieval-Augmented Generation pipeline for Vietnamese law corpus.

Uses Hybrid Search (FAISS + BM25 + RRF) for retrieval, then sends context
to the LLM for answer generation.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from configs.config import NUM_RESULTS
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Bạn là một trợ lý pháp lý thông thái và chính xác, chuyên về pháp luật Việt Nam.
Trả lời ONLY dựa trên các điều khoản pháp luật được cung cấp trong phần CONTEXT.
Nếu context không đủ thông tin, hãy nói rõ điều đó.
Trả lời bằng tiếng Việt, rõ ràng và đầy đủ.
"""

PROMPT_TEMPLATE = """\
CONTEXT – Các điều khoản pháp luật liên quan:
{context}

---
Câu hỏi: {query}

Trả lời:"""


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    retriever: HybridRetriever,
    k: int = NUM_RESULTS,
    alpha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Full RAG cycle:
      1. Hybrid Search: retrieve top-k articles via FAISS + BM25 + RRF.
      2. Build a context string with law_id and article text.
      3. Ask the LLM to answer based on that context.

    Args:
        query: the user's question in Vietnamese.
        retriever: a loaded HybridRetriever instance.
        k: number of articles to retrieve.
        alpha: weight for semantic vs keyword search (0.5 = balanced).
        verbose: print retrieval details to stdout.

    Returns a dict with keys:
        answer      – LLM response string
        sources     – list of dicts with [rank, law_id, aid, score, faiss_score, bm25_score, text]
    """
    # 1. Hybrid retrieval
    sources = retriever.search(query, k=k, alpha=alpha)

    # 2. Build context
    context_parts = []
    for s in sources:
        context_parts.append(
            f"[{s['rank']}] Luật {s['law_id']} – Điều {s['aid']}:\n{s['text']}"
        )
    context = "\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Top-{k} retrieved articles (Hybrid FAISS+BM25):")
        for s in sources:
            print(f"  [{s['rank']}] rrf={s['score']:.6f}  "
                  f"faiss={s['faiss_score']:.4f}  bm25={s['bm25_score']:.4f}  "
                  f"law={s['law_id']}  aid={s['aid']}")

    # 3. Generate answer
    answer = chat(prompt, system=SYSTEM_PROMPT)

    return {"answer": answer, "sources": sources}
