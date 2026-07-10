"""
rag.py – Main Retrieval-Augmented Generation (RAG) pipeline for legal cases.

Coordinates evidence retrieval (API/Cache), legal text retrieval (Hybrid Search),
and LLM generation to predict court case verdicts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs.config import DEFAULT_TOP_K_LAWS, MAX_LAW_CONTENT_LENGTH
from rag_retrieval.case_api import CaseAPIClient
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat, LMStudioClient


def answer_query(
    query: str,
    retriever: HybridRetriever,
    k: int,
    alpha: float,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Interactive legal query answering using Hybrid Search and LLM generation.
    Used by the query.py CLI tool.
    """
    if verbose:
        print(f"\n[RAG] Searching for: '{query}'")

    # 1. Retrieve laws
    rel_laws = retriever.search(query, k=k, alpha=alpha)
    
    if verbose:
        print(f"[RAG] Retrieved {len(rel_laws)} laws.")
        for l in rel_laws:
            print(f"  - Luật {l['law_id']} Điều {l['aid']} (Score: {l['score']})")

    # 2. Build Prompt
    laws_str = "\n".join(
        f"- Luật {l['law_id']} Điều {l['aid']}: {l['text'][:MAX_LAW_CONTENT_LENGTH]}"
        for l in rel_laws
    )
    
    system_prompt = (
        "Bạn là một chuyên gia pháp lý Việt Nam. Hãy trả lời câu hỏi dựa trên "
        "các điều luật được cung cấp một cách ngắn gọn, chính xác."
    )
    
    user_prompt = f"""CÂU HỎI:
{query}

ĐIỀU LUẬT THAM KHẢO:
{laws_str}

Hãy trả lời câu hỏi trên dựa vào các điều luật tham khảo."""

    # 3. Call LLM
    if verbose:
        print("[RAG] Generating answer...")
    
    # We use the raw `chat` from llm_client since this isn't structured JSON output
    response = chat(prompt=user_prompt, system=system_prompt)
    
    return {
        "query": query,
        "answer": response,
        "retrieved_laws": rel_laws,
    }


class LegalRAGPipeline:
    """Crisp, DRY RAG pipeline coordinating HybridRetriever and unified case prediction."""

    def __init__(
        self,
        llm: LMStudioClient | None = None,
        api: CaseAPIClient | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.llm = llm
        self.api = api or CaseAPIClient()
        self.retriever = retriever or HybridRetriever.from_disk()

    def search_laws(self, query: str, top_k: int = DEFAULT_TOP_K_LAWS) -> list[dict[str, Any]]:
        """Search legal corpus using active Hybrid FAISS + BM25 retrieval."""
        return self.retriever.search(query, k=top_k)

    def predict(self, case: dict[str, Any], top_k: int = DEFAULT_TOP_K_LAWS) -> dict[str, Any]:
        """Predict case outcome by delegating to the unified prediction pipeline (DRY)."""
        from rag_retrieval.generate_submission import predict_case
        return predict_case(case, self.retriever, top_k=top_k)