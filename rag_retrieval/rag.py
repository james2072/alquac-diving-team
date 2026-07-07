"""
rag.py – Main Retrieval-Augmented Generation (RAG) pipeline for legal cases.

Coordinates evidence retrieval (API/Cache), legal text retrieval (Hybrid Search),
and LLM generation to predict court case verdicts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs.config import (
    DEFAULT_TOP_K_LAWS,
    MAX_EVIDENCE_LENGTH_FOR_PROMPT,
    MAX_EVIDENCE_LENGTH_FOR_SEARCH,
    MAX_LAW_CONTENT_LENGTH,
    MIN_SEARCH_TEXT_LENGTH,
)
from rag_retrieval.case_api import CaseAPIClient
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat, LMStudioClient
from rag_retrieval.utils import extract_json_from_text, rule_override

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}


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
    """RAG pipeline integrating base laws and external laws."""

    def __init__(
        self,
        llm: LMStudioClient,
        api: CaseAPIClient,
        corpus_path: str = "data/corpus/corpus_law_pub.json",
        output_dir: str = "data/output",
        external_dir: str = "data/external",
    ) -> None:
        self.llm = llm
        self.api = api
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.external_dir = Path(external_dir)

        if not self.corpus_path.exists():
            self.corpus_path = Path("corpus_law_pub.json")

        self.base_laws = self._load_base_laws()
        self.valid_law_aids = {(l["law_id"], l["aid"]) for l in self.base_laws}
        
        self.external_laws = self._load_external_laws()
        self.all_laws = self.base_laws + self.external_laws
        
        # Note: Pre-built index initialization is delegated to caller or Retriever classes in clean architecture
        print(f"[RAG] Ready: {len(self.base_laws)} base + {len(self.external_laws)} external laws")

    def _load_base_laws(self) -> list[dict[str, Any]]:
        """Parse base corpus cleanly using standard JSON module."""
        print(f"[RAG] Loading corpus from {self.corpus_path}")
        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
        except Exception as e:
            print(f"[RAG] Failed to load {self.corpus_path}: {e}")
            return []

        laws: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        for law in corpus:
            law_id = str(law.get("law_id", "unknown")).strip()
            for article in law.get("content", []):
                aid = int(article.get("aid", -1))
                content = str(article.get("content_Article", "")).strip()
                
                if content and aid != -1:
                    key = (law_id, aid)
                    if key not in seen:
                        laws.append({
                            "aid": aid,
                            "content": content,
                            "law_id": law_id,
                            "is_external": False,
                        })
                        seen.add(key)
        
        print(f"[RAG] Parsed {len(laws)} base laws")
        return laws

    def _load_external_laws(self) -> list[dict[str, Any]]:
        """Load external laws for reasoning support."""
        if not self.external_dir.exists():
            print(f"[RAG] External dir not found: {self.external_dir}")
            return []
        
        laws: list[dict[str, Any]] = []
        for json_file in self.external_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                items = data if isinstance(data, list) else data.get("laws", [data])
                for law in items:
                    if not isinstance(law, dict):
                        continue
                    lid = law.get("law_id", "external")
                    for art in law.get("articles", []):
                        content = art.get("content", "").strip()
                        if content:
                            laws.append({
                                "aid": art.get("aid", 900000 + len(laws)),
                                "content": content,
                                "law_id": lid,
                                "is_external": True,
                            })
            except Exception as e:
                print(f"[RAG] Failed to load external laws from {json_file}: {e}")
                
        print(f"[RAG] Loaded {len(laws)} external laws")
        return laws

    def search_laws(self, query: str, top_k: int = DEFAULT_TOP_K_LAWS) -> list[dict[str, Any]]:
        """
        Placeholder for legacy method. In the refactored architecture,
        use HybridRetriever explicitly.
        """
        print("[RAG] Warning: Using placeholder search_laws. Switch to HybridRetriever for accuracy.")
        return []

    def rule_override(self, text: str) -> str | None:
        """Apply deterministic keyword rules to override predictions."""
        return rule_override(text)

    def predict(self, case: dict[str, Any]) -> dict[str, Any]:
        """Predict case outcome using RAG."""
        cid = str(case.get("case_id", "")).strip()
        query = str(case.get("case_query", ""))
        
        print(f"\n[Case] {cid}")
        
        ev_segs = self.api.get_adaptive_evidence(case)
        ev_text = "\n".join([s.get("text", "") for s in ev_segs])
        
        rule_pred = self.rule_override(ev_text)
        if rule_pred:
            print(f"  -> Rule Override: {rule_pred}")
            pred = rule_pred
            laws = self.search_laws(query, top_k=DEFAULT_TOP_K_LAWS)
        else:
            search_text = ev_text[-MAX_EVIDENCE_LENGTH_FOR_SEARCH:] if len(ev_text) > MAX_EVIDENCE_LENGTH_FOR_SEARCH else query
            if len(search_text.strip()) < MIN_SEARCH_TEXT_LENGTH:
                search_text = query
            
            laws = self.search_laws(search_text, top_k=DEFAULT_TOP_K_LAWS)
            laws_str = "\n".join(
                f"- Điều {l['aid']} ({l['law_id']}): {l['content'][:MAX_LAW_CONTENT_LENGTH]}"
                for l in laws
            )
            
            # Note: System prompt remains in Vietnamese for LLM performance on local legal cases
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là Thẩm phán Tòa án nhân dân tối cao Việt Nam. "
                        "Hãy phân tích kỹ bằng chứng và điều luật để dự đoán kết quả.\n\n"
                        "Định nghĩa:\n"
                        "- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu nguyên đơn\n"
                        "- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN (phần được chấp nhận > 50%)\n"
                        "- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN (phần được chấp nhận ≤ 50%)\n"
                        "- B_WIN: Tòa BÁC TOÀN BỘ yêu cầu nguyên đơn\n\n"
                        "Trả về DUY NHẤT JSON: {\"prediction\": \"...\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"THÔNG TIN VỤ ÁN: {query}\n\n"
                        f"BẰNG CHỨNG:\n{ev_text[:MAX_EVIDENCE_LENGTH_FOR_PROMPT]}\n\n"
                        f"ĐIỀU LUẬT LIÊN QUAN:\n{laws_str}"
                    ),
                },
            ]
            
            resp = self.llm.generate(msgs)
            
            pred = "B_WIN"
            parsed = extract_json_from_text(resp)
            if parsed:
                p = parsed.get("prediction", "B_WIN")
                if p in VALID_LABELS:
                    pred = p
            else:
                print("  [WARN] JSON parse failed, falling back to B_WIN.")
            
            print(f"  -> LLM Prediction: {pred}")
        
        valid_laws = [
            l for l in laws
            if (l["law_id"], l["aid"]) in self.valid_law_aids
        ]
        
        return {
            "case_id": cid,
            "prediction": pred,
            "law_evidence": [
                {"law_id": l["law_id"], "aid": l["aid"]}
                for l in valid_laws
            ],
            "case_evidence": [
                s.get("chunk_id")
                for s in ev_segs
                if s.get("chunk_id")
            ],
        }