"""
rag.py – Retrieval-Augmented Generation pipeline for Vietnamese law corpus.

Mirrors the notebook's "Generate an answer" section.
"""
from __future__ import annotations

import pandas as pd
import torch

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from configs.config import NUM_RESULTS
from rag_runner.embedder import get_embedding_model, retrieve_top_k
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
# English fallback:
# SYSTEM_PROMPT = """
# You are a precise legal assistant specialising in Vietnamese law.
# Answer ONLY based on the legal articles supplied in CONTEXT.
# If the context is insufficient, say so clearly.
# """

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
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    k: int = NUM_RESULTS,
    verbose: bool = True,
) -> dict:
    """
    Full RAG cycle:
      1. Retrieve top-k relevant articles via embedding similarity.
      2. Build a context string with law_id and article text.
      3. Ask the LLM to answer based on that context.

    Returns a dict with keys:
        answer      – LLM response string
        sources     – list of dicts with [law_id, aid, score, text]
    """
    model = get_embedding_model()
    scores, indices = retrieve_top_k(query, embeddings, model=model, k=k)

    # Build context
    sources = []
    context_parts = []
    for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        row = df.iloc[idx]
        sources.append({
            "rank":   rank,
            "law_id": row["law_id"],
            "aid":    row["aid"],
            "score":  round(score, 4),
            "text":   row["text"],
        })
        context_parts.append(
            f"[{rank}] Luật {row['law_id']} – Điều {row['aid']}:\n{row['text']}"
        )

    context = "\n\n".join(context_parts)
    prompt  = PROMPT_TEMPLATE.format(context=context, query=query)

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Top-{k} retrieved articles:")
        for s in sources:
            print(f"  [{s['rank']}] score={s['score']:.4f}  "
                  f"law={s['law_id']}  aid={s['aid']}")

    answer = chat(prompt, system=SYSTEM_PROMPT)

    return {"answer": answer, "sources": sources}
