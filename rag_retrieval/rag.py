from __future__ import annotations
import json
import re
import pickle
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from collections import defaultdict

from rag_retrieval.llm_client import LMStudioClient
from rag_retrieval.case_api import CaseAPIClient


VALID_LABELS = ["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]


class LegalRAGPipeline:
    """RAG pipeline with pre-built indexes + external laws."""
    
    def __init__(
        self,
        llm: LMStudioClient,
        api: CaseAPIClient,
        corpus_path: str = "data/corpus/corpus_law_pub.json",
        output_dir: str = "data/output",
        external_dir: str = "data/external",
    ):
        self.llm = llm
        self.api = api
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.external_dir = Path(external_dir)
        
        # Fallback to root corpus if not in data/corpus/
        if not self.corpus_path.exists():
            self.corpus_path = Path("corpus_law_pub.json")
        
        # Load base laws and pre-built indexes
        self.base_laws = self._load_base_laws()
        self.valid_law_aids = {(l["law_id"], l["aid"]) for l in self.base_laws}
        
        # Try pre-built indexes first, fallback to in-memory build
        if not self._load_prebuilt_indexes():
            self._build_indexes_in_memory()
        
        # Load external laws (for LLM reasoning only)
        self.external_laws = self._load_external_laws()
        self.all_laws = self.base_laws + self.external_laws
        
        print(f"[RAG] Ready: {len(self.base_laws)} base + {len(self.external_laws)} external laws")
    
    def _load_base_laws(self) -> list[dict]:
        """Parse base corpus with robust regex."""
        print(f"[RAG] Loading corpus from {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        aids = list(re.finditer(r'"aid\s*"\s*:\s*(\d+)', text))
        law_ids = [
            (m.start(), m.group(1).strip())
            for m in re.finditer(r'"law_id\s*"\s*:\s*"([^"]+)"', text)
        ]
        
        laws = []
        for i, am in enumerate(aids):
            aid = int(am.group(1))
            lid = "unknown"
            for lp, l_id in law_ids:
                if lp < am.start():
                    lid = l_id
                else:
                    break
            
            end = aids[i + 1].start() if i + 1 < len(aids) else len(text)
            chunk = text[am.end():end]
            ca = re.search(r'"content_Article\s*"\s*:\s*"', chunk)
            if not ca:
                continue
            
            start = ca.end()
            pos = start
            while pos < len(chunk):
                if chunk[pos] == '"' and (pos == 0 or chunk[pos - 1] != '\\'):
                    break
                pos += 1
            
            content = chunk[start:pos].replace('\\n', '\n').replace('\\"', '"').strip()
            if content:
                laws.append({
                    "aid": aid,
                    "content": content,
                    "law_id": lid,
                    "is_external": False,
                })
        
        # Deduplicate
        seen, unique = set(), []
        for l in laws:
            key = (l["law_id"], l["aid"])
            if key not in seen:
                unique.append(l)
                seen.add(key)
        
        print(f"[RAG] Parsed {len(unique)} base laws")
        return unique
    
    def _load_external_laws(self) -> list[dict]:
        """Load external laws from data/external/ (for LLM reasoning only)."""
        if not self.external_dir.exists():
            print(f"[RAG] External dir not found: {self.external_dir}")
            return []
        
        laws = []
        for json_file in self.external_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle both list and dict formats
                items = data if isinstance(data, list) else data.get("laws", [data])
                
                for law in items:
                    if not isinstance(law, dict):
                        continue
                    lid = law.get("law_id", "external")
                    articles = law.get("articles", [])
                    if isinstance(articles, list):
                        for art in articles:
                            content = art.get("content", "").strip()
                            if content:
                                laws.append({
                                    "aid": art.get("aid", 900000 + len(laws)),
                                    "content": content,
                                    "law_id": lid,
                                    "is_external": True,
                                })
            except Exception as e:
                print(f"[RAG] Failed to load {json_file}: {e}")
        
        print(f"[RAG] Loaded {len(laws)} external laws")
        return laws
    
    def _load_prebuilt_indexes(self) -> bool:
        """Load pre-built BM25 + FAISS indexes from data/output/."""
        bm25_path = self.output_dir / "law_bm25.pkl"
        faiss_path = self.output_dir / "law.faiss"
        
        if not (bm25_path.exists() and faiss_path.exists()):
            print(f"[RAG] Pre-built indexes not found in {self.output_dir}")
            return False
        
        try:
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            print(f"[RAG] Loaded BM25 from {bm25_path}")
            
            self.index = faiss.read_index(str(faiss_path))
            print(f"[RAG] Loaded FAISS from {faiss_path}")
            
            # Still need embedding model for query encoding
            self.embedder = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
            print("[RAG] Loaded BGE-M3 for query encoding")
            return True
        
        except Exception as e:
            print(f"[RAG] Failed to load pre-built indexes: {e}")
            return False
    
    def _build_indexes_in_memory(self):
        """Fallback: build BM25 + FAISS indexes in memory."""
        print("[RAG] Building indexes in memory (fallback)...")
        
        # BM25 on all laws (base + external)
        corpus = [
            f"Điều {l['aid']} {l['content']}".lower().split()
            for l in self.all_laws
        ]
        self.bm25 = BM25Okapi(corpus)
        
        # Dense with BGE-M3
        self.embedder = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
        law_texts = [
            f"passage: Điều {l['aid']}: {l['content'][:1200]}"
            for l in self.all_laws
        ]
        embeddings = self.embedder.encode(
            law_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        print("[RAG] Indexes built in memory")
    
    def search_laws(self, query: str, top_k: int = 6) -> list[dict]:
        """Hybrid search: BM25 + dense with RRF fusion."""
        # BM25
        tokenized_q = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_ranks = np.argsort(-bm25_scores)
        
        # Dense
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        k_search = min(50, len(self.all_laws))
        _, ann = self.index.search(np.array(q_emb), k_search)
        
        # RRF fusion
        k = 60
        rrf_scores = defaultdict(float)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
        for rank, idx in enumerate(ann[0]):
            if idx != -1 and idx < len(self.all_laws):
                rrf_scores[idx] += 1.0 / (k + rank + 1)
        
        sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [self.all_laws[i] for i in sorted_idx[:top_k]]
    
    def rule_override(self, text: str):
        """Override prediction using strong keywords from evidence."""
        t = text.lower()
        
        # Strong signals
        has_accept_all = any(kw in t for kw in [
            "chấp nhận toàn bộ",
            "chấp nhận yêu cầu của nguyên đơn",
            "chấp nhận yêu cầu khởi kiện",
        ])
        has_reject_all = any(kw in t for kw in [
            "không chấp nhận yêu cầu",
            "bác toàn bộ",
            "bác yêu cầu",
            "không có căn cứ chấp nhận",
        ])
        has_partial = any(kw in t for kw in [
            "chấp nhận một phần",
            "một phần yêu cầu",
        ])
        
        # Decision logic
        if has_reject_all and not has_accept_all and not has_partial:
            return "B_WIN"
        if has_accept_all and not has_reject_all and not has_partial:
            return "A_WIN"
        if has_partial or (has_accept_all and has_reject_all):
            # Determine if >50% or <=50%
            # Heuristic: count occurrences
            accept_count = sum(1 for kw in ["chấp nhận"] if kw in t)
            reject_count = sum(1 for kw in ["không chấp nhận", "bác"] if kw in t)
            if accept_count > reject_count:
                return "PARTIAL_A_WIN"
            else:
                return "PARTIAL_B_WIN"
        
        return None
    
    def predict(self, case: dict) -> dict:
        """Predict case outcome using RAG pipeline."""
        cid = str(case.get("case_id", "")).strip()
        query = case.get("case_query", "")
        
        print(f"\n[Case] {cid}")
        
        # 1. Get evidence from cache (no API calls)
        ev_segs = self.api.get_adaptive_evidence(case)
        ev_text = "\n".join([s.get("text", "") for s in ev_segs])
        
        # 2. Try rule override
        rule_pred = self.rule_override(ev_text)
        if rule_pred:
            print(f"  -> Rule Override: {rule_pred}")
            pred = rule_pred
            # Still need laws for submission
            laws = self.search_laws(query, top_k=6)
        else:
            # 3. Hybrid law search (use last 1500 chars of evidence for better context)
            search_text = ev_text[-1500:] if len(ev_text) > 1500 else query
            if len(search_text.strip()) < 50:
                search_text = query
            
            laws = self.search_laws(search_text, top_k=6)
            
            # Format laws for prompt (include external for reasoning)
            laws_str = "\n".join([
                f"- Điều {l['aid']} ({l['law_id']}): {l['content'][:200]}"
                for l in laws
            ])
            
            # 4. LLM prediction
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
                        f"BẰNG CHỨNG:\n{ev_text[:3000]}\n\n"
                        f"ĐIỀU LUẬT LIÊN QUAN:\n{laws_str}"
                    ),
                },
            ]
            
            resp = self.llm.generate(msgs)
            
            # Parse JSON
            pred = "B_WIN"
            try:
                js = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
                if js:
                    parsed = json.loads(js.group(0))
                    p = parsed.get("prediction", "B_WIN")
                    if p in VALID_LABELS:
                        pred = p
            except Exception as e:
                print(f"  [WARN] JSON parse failed: {e}")
            
            print(f"  -> LLM: {pred}")
        
        # 5. Filter law_evidence (only base laws, no external)
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