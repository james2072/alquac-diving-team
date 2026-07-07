"""
generate_submission.py – Automated submission pipeline for ALQAC 2026.

Workflow:
  1. Load test cases from JSON (ALQAC2026_public_test.json).
  2. For each case:
     - Retrieve evidence chunks using API or cache.
     - Retrieve relevant laws using Hybrid Search (FAISS + BM25 + RRF).
     - Format and query the LLM for a legal prediction.
     - Validate and parse the output JSON.
  3. Export results incrementally to submission.json.

Usage:
    python -m rag_retrieval.generate_submission --test-file data/test/ALQAC2026_public_test.json --output submission.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from configs.config import (
    DEFAULT_ALPHA,
    DEFAULT_SUBMISSION_TOP_K,
    LLM_TEMPERATURE,
    MAX_CASE_EVIDENCE_CHUNK_LEN,
    MAX_CONTEXT_CHUNK_LEN_FOR_SEARCH,
    MAX_CONTEXT_CHUNKS_FOR_SEARCH,
    MAX_FACT_LEN_FOR_SEARCH,
    MAX_LAW_TEXT_LEN_FOR_PROMPT,
    PROJECT_ROOT,
    SUBMISSION_FILE,
    TEST_FILE,
)
from rag_retrieval.evidence_api_client import get_case_evidence
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat
from rag_retrieval.utils import extract_json_from_text, rule_override, strip_think_tags

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}

# Note: The system prompt remains in Vietnamese to instruct the LLM correctly 
# for Vietnamese legal reasoning.
SYSTEM_PROMPT = """Bạn là một Chuyên gia Pháp lý và Thẩm phán Tối cao tại Việt Nam.
Nhiệm vụ của bạn là phân tích thông tin vụ án (Yêu cầu khởi kiện), các bằng chứng thu thập được, và các điều luật liên quan để đưa ra phán quyết chính xác nhất.

PHÂN LOẠI PHÁN QUYẾT (bắt buộc chọn 1 trong 4 nhãn sau):
- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu của nguyên đơn.
- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, phần chấp nhận > 50%.
- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, nhưng phần chấp nhận <= 50%.
- B_WIN: Tòa BÁC TOÀN BỘ yêu cầu của nguyên đơn.

NGUYÊN TẮC SUY LUẬN BẮT BUỘC:
1. TUYỆT ĐỐI CHỈ suy luận dựa trên các bằng chứng tình tiết (Case Evidence) và quy định pháp luật (Law Evidence) được cung cấp trong prompt. Không được tự ý suy diễn hay giả định thêm chi tiết không có trong dữ liệu.
2. CHỌN LỌC BẰNG CHỨNG (PICK): Bạn phải chọn ra chính xác từ danh sách chứng cứ (`chunk_id`) và điều luật (`law_id`, `aid`) đã cung cấp những đoạn nào thực sự được bạn dùng làm căn cứ quyết định phán quyết để nộp bài.

VÍ DỤ SUY LUẬN (FEW-SHOT EXAMPLES):
- Ví dụ 1: Nếu chứng cứ cho thấy "Tòa án quyết định không có căn cứ để chấp nhận yêu cầu của nguyên đơn", hãy chọn prediction là "B_WIN".
- Ví dụ 2: Nếu chứng cứ cho thấy "Chấp nhận yêu cầu khởi kiện, buộc bị đơn trả đủ số tiền nguyên đơn đòi", hãy chọn prediction là "A_WIN".
- Ví dụ 3: Nếu chứng cứ cho thấy "Chấp nhận một phần yêu cầu khởi kiện, buộc bị đơn bồi thường 70 triệu trên tổng số 100 triệu nguyên đơn yêu cầu (> 50%)", hãy chọn prediction là "PARTIAL_A_WIN".

Bạn PHẢI trả lời CHỈ DUY NHẤT một object JSON hợp lệ theo đúng định dạng (không giải thích ngoài JSON):
{
  "prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN",
  "reasoning": "Giải thích ngắn gọn căn cứ phán quyết dựa trên chứng cứ và điều luật",
  "selected_case_evidence": ["danh_sách_mã_chunk_id_đã_dùng"],
  "selected_law_evidence": [{"law_id": "mã_luật", "aid": số_điều}]
}"""


def _retrieve_evidence(
    cid: str, query: str, fact: str, retriever: HybridRetriever, top_k: int, alpha: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Retrieve both case evidence (API/Cache) and legal text (Hybrid Search)."""
    # 1. Retrieve case evidence
    case_ev_data = get_case_evidence(cid, query, case_fact=fact)
    
    case_ev_str = "\n".join(
        f"- [{r.get('chunk_id', '')}] {str(r.get('text', ''))[:MAX_CASE_EVIDENCE_CHUNK_LEN]}"
        for r in case_ev_data if isinstance(r, dict) and r.get("text")
    )

    # 2. Retrieve laws by combining query, fact, and case evidence context
    ev_context = " ".join(
        [str(r.get("text", ""))[:MAX_CONTEXT_CHUNK_LEN_FOR_SEARCH] 
         for r in case_ev_data if isinstance(r, dict) and r.get("text")][:MAX_CONTEXT_CHUNKS_FOR_SEARCH]
    )
    
    search_query = f"{query}\n{fact[:MAX_FACT_LEN_FOR_SEARCH]}\n{ev_context}"
    rel_laws = retriever.search(search_query, k=top_k, alpha=alpha)
    
    laws_str = "\n".join(
        f"- Luật {l['law_id']} Điều {l['aid']}: {str(l['text'])[:MAX_LAW_TEXT_LEN_FOR_PROMPT]}..."
        for l in rel_laws
    )
    
    return case_ev_data, rel_laws, case_ev_str, laws_str


def _build_prompt(query: str, case_ev_str: str, laws_str: str) -> str:
    """Construct the user prompt for the LLM."""
    return f"""THÔNG TIN VỤ ÁN (Yêu cầu khởi kiện):
{query}

BẰNG CHỨNG THU THẬP ĐƯỢC (Case Evidence):
{case_ev_str if case_ev_str else "(Không có chứng cứ bổ sung)"}

ĐIỀU LUẬT LIÊN QUAN:
{laws_str if laws_str else "(Không tìm thấy điều luật liên quan)"}

Hãy phân tích, dự đoán kết quả xét xử và chọn ra các bằng chứng/điều luật làm căn cứ, trả về đúng định dạng JSON yêu cầu."""


def _parse_and_validate(
    raw_resp: str,
    cid: str,
    case_ev_data: list[dict[str, Any]],
    rel_laws: list[dict[str, Any]],
    valid_law_aids: set[tuple[str, int]],
) -> dict[str, Any]:
    """Parse JSON and validate selected evidence and labels."""
    clean_resp = strip_think_tags(raw_resp)
    
    pred = "B_WIN"  # Default fallback
    selected_laws: list[Any] = []
    
    parsed = extract_json_from_text(clean_resp)
    if parsed:
        candidate = str(parsed.get("prediction", "")).strip()
        if candidate in VALID_LABELS:
            pred = candidate
        else:
            print(f"\n  [WARN] Case {cid}: Invalid label '{candidate}'. Falling back to B_WIN.")
        
        selected_laws = parsed.get("selected_law_evidence", [])
    else:
        print(f"\n  [WARN] Case {cid}: No valid JSON block found in LLM output.")

    # Submit ALL valid case evidence chunks retrieved (to maximize Case Recall)
    sub_case_ev = [str(r["chunk_id"]) for r in case_ev_data if isinstance(r, dict) and "chunk_id" in r]

    # Validate selected laws
    sub_law_ev: list[dict[str, Any]] = []
    if isinstance(selected_laws, list):
        for le in selected_laws:
            if isinstance(le, dict) and "law_id" in le and "aid" in le:
                try:
                    lid = str(le["law_id"]).strip()
                    aid = int(le["aid"])
                    if (lid, aid) in valid_law_aids:
                        sub_law_ev.append({"law_id": lid, "aid": aid})
                except (ValueError, TypeError):
                    pass

    # Fallback to top retrieved laws if LLM failed to pick correctly
    if not sub_law_ev:
        sub_law_ev = [
            {"law_id": str(l["law_id"]).strip(), "aid": int(l["aid"])}
            for l in rel_laws[:4]
            if (str(l["law_id"]).strip(), int(l["aid"])) in valid_law_aids
        ]

    return {
        "case_id": str(cid),
        "prediction": str(pred),
        "case_evidence": sub_case_ev,
        "law_evidence": sub_law_ev,
    }


def predict_case(
    case: dict[str, Any],
    retriever: HybridRetriever,
    top_k: int = DEFAULT_SUBMISSION_TOP_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Execute prediction pipeline for a single case."""
    cid = str(case.get("case_id", "unknown")).strip()
    query = str(case.get("case_query", ""))
    fact = str(case.get("case_fact", ""))

    if retriever.valid_law_aids is None:
        retriever.valid_law_aids = {
            (str(row["law_id"]).strip(), int(row["aid"]))
            for _, row in retriever.df.iterrows()
        }

    case_ev_data, rel_laws, case_ev_str, laws_str = _retrieve_evidence(
        cid, query, fact, retriever, top_k, alpha
    )

    user_prompt = _build_prompt(query, case_ev_str, laws_str)
    raw_resp = chat(prompt=user_prompt, system=SYSTEM_PROMPT, temperature=LLM_TEMPERATURE)
    
    res = _parse_and_validate(raw_resp, cid, case_ev_data, rel_laws, retriever.valid_law_aids)

    # Apply deterministic rule override (Outcome Accuracy boost)
    rule_pred = rule_override(fact)
    if rule_pred:
        print(f"  [RuleOverride] Case {cid}: Deterministic rule override applied -> {rule_pred}")
        res["prediction"] = rule_pred

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ALQAC 2026 submission file.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=TEST_FILE,
        help="Path to the JSON test file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUBMISSION_FILE,
        help="Path to save the resulting submission JSON.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_SUBMISSION_TOP_K, help="Number of laws to retrieve per case.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="RRF weight balancing FAISS and BM25.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cases (for debugging).")
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

    print("[INFO] Loading Hybrid Retriever index (FAISS + BM25)...")
    retriever = HybridRetriever.from_disk()

    # Read incremental checkpoint
    submission: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    
    if args.output.exists():
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                submission = json.load(f)
                completed_ids = {str(item["case_id"]).strip() for item in submission}
            print(f"[INFO] Found checkpoint '{args.output}' with {len(completed_ids)} completed cases. Resuming...")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to read checkpoint: {e}. Starting fresh.")
            submission = []
            completed_ids = set()

    print(f"\n[START] Processing {len(cases)} cases...")
    t0 = time.perf_counter()

    for i, case in enumerate(cases, 1):
        cid = str(case.get("case_id", f"case_{i}")).strip()
        if cid in completed_ids:
            print(f"[{i}/{len(cases)}] Case: {cid} → [SKIP] Already completed.")
            continue

        print(f"[{i}/{len(cases)}] Processing case: {cid} ...", end=" ", flush=True)
        t_case = time.perf_counter()
        
        result = predict_case(case, retriever, top_k=args.top_k, alpha=args.alpha)
        submission.append(result)
        completed_ids.add(cid)
        
        dt = time.perf_counter() - t_case
        print(f"→ Prediction: {result['prediction']} ({dt:.1f}s)")

        # Save checkpoint incrementally
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2, default=str)

    total_time = time.perf_counter() - t0
    print(f"\n✅ Completed! {len(submission)} predictions saved to: {args.output}")
    print(f"⏱️ Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
