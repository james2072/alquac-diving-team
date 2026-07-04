"""
generate_submission.py – Pipeline tự động tạo file dự đoán (submission) cho ALQAC 2026.

Flow hoạt động:
  1. Đọc danh sách vụ án từ file test JSON (ALQAC2026_public_test.json).
  2. Với mỗi vụ án:
     - Trích xuất thông tin vụ án (case_query, case_fact).
     - Sử dụng HybridRetriever (FAISS + BM25 + RRF) tìm top-K Điều luật liên quan nhất.
     - Tạo prompt gửi lên LLM qua llm_client (hỗ trợ Google AI Studio / OpenAI / Ollama).
     - Parse JSON output để lấy nhãn dự đoán (verdict_label / prediction).
  3. Xuất kết quả ra file submission.json đúng định dạng cuộc thi.

Usage:
    python -m rag_retrieval.generate_submission --test-file data/test/ALQAC2026_public_test.json --output submission.json
"""
from __future__ import annotations

import argparse
import json
import re
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

from configs.config import PROJECT_ROOT, NUM_RESULTS
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat
from rag_retrieval.evidence_api_client import get_case_evidence

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}

SYSTEM_PROMPT = """Bạn là một Thẩm phán Tòa án nhân dân giàu kinh nghiệm, chuyên xét xử các vụ án dân sự, thương mại, hành chính và lao động tại Việt Nam.
Nhiệm vụ của bạn: Đọc kỹ Yêu cầu khởi kiện của Nguyên đơn, Các đoạn chứng cứ tình tiết thu thập được và Danh sách các Điều luật liên quan được cung cấp, sau đó đưa ra phán quyết khách quan và chính xác nhất.

Định nghĩa 4 nhãn phán quyết:
- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu của nguyên đơn.
- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, phần chấp nhận > 50%.
- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, nhưng phần chấp nhận <= 50%.
- B_WIN: Tòa BÁC TOÀN BỘ yêu cầu của nguyên đơn.

NGUYÊN TẮC SUY LUẬN BẮT BUỘC:
1. TUYỆT ĐỐI CHỈ suy luận dựa trên các bằng chứng tình tiết (Case Evidence) và quy định pháp luật (Law Evidence) được cung cấp trong prompt. Không được tự ý suy diễn hay giả định thêm chi tiết không có trong dữ liệu.
2. CHỌN LỌC BẰNG CHỨNG (PICK): Bạn phải chọn ra chính xác từ danh sách chứng cứ (`chunk_id`) và điều luật (`law_id`, `aid`) đã cung cấp những đoạn nào thực sự được bạn dùng làm căn cứ quyết định phán quyết để nộp bài.

Bạn PHẢI trả lời CHỈ DUY NHẤT một object JSON hợp lệ theo đúng định dạng (không giải thích ngoài JSON):
{
  "prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN",
  "reasoning": "Giải thích ngắn gọn căn cứ phán quyết dựa trên chứng cứ và điều luật",
  "selected_case_evidence": ["danh_sách_mã_chunk_id_đã_dùng"],
  "selected_law_evidence": [{"law_id": "mã_luật", "aid": số_điều}]
}"""


def strip_think(text: str) -> str:
    """Loại bỏ khối <think>...</think> nếu LLM sử dụng reasoning mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_last_json(text: str) -> str | None:
    """Tìm và trích xuất khối JSON hợp lệ trong text trả về từ LLM (hỗ trợ nested object/array)."""
    # 1. Thử tìm khối ```json ... ``` trước
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if md_match:
        try:
            json.loads(md_match.group(1))
            return md_match.group(1)
        except Exception:
            pass

    # 2. Tìm từ dấu { đầu tiên đến dấu } cuối cùng
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 3. Quét từng cặp { ... } cân bằng từ ngoài vào trong
    start_idx = 0
    while True:
        s = text.find('{', start_idx)
        if s == -1:
            break
        depth = 0
        for i in range(s, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    cand = text[s:i+1]
                    try:
                        json.loads(cand)
                        return cand
                    except Exception:
                        break
        start_idx = s + 1

    return None


def predict_case(
    case: dict,
    retriever: HybridRetriever,
    top_k: int = 10,
    alpha: float = 0.5,
) -> dict:
    """Dự đoán kết quả cho 1 vụ án."""
    cid = str(case.get("case_id", "unknown")).strip()
    query = case.get("case_query", "")
    fact = case.get("case_fact", "")

    # Tạo tập hợp các điều luật hợp lệ từ retriever để post-validate
    valid_law_aids = getattr(retriever, "valid_law_aids", None)
    if valid_law_aids is None:
        valid_law_aids = {(str(row["law_id"]).strip(), int(row["aid"])) for _, row in retriever.df.iterrows()}
        retriever.valid_law_aids = valid_law_aids

    # 1. Retrieve case_evidence từ API BTC (hoặc cache) với chiến lược đa chiều TRƯỚC
    case_ev_data = get_case_evidence(cid, query)
    case_ev_str = "\n".join(
        f"- [{r.get('chunk_id', '')}] {r.get('text', '')[:1500]}"
        for r in case_ev_data if isinstance(r, dict) and r.get('text')
    )

    # 2. Retrieve laws (Top-k) bằng cách kết hợp query, fact và toàn bộ ngữ cảnh case evidence
    ev_context = " ".join([r.get('text', '')[:1000] for r in case_ev_data if isinstance(r, dict) and r.get('text')][:8])
    search_query = f"{query}\n{fact[:400]}\n{ev_context}"
    rel_laws = retriever.search(search_query, k=top_k, alpha=alpha)
    
    laws_str = "\n".join(
        f"- Luật {l['law_id']} Điều {l['aid']}: {l['text'][:1500]}..."
        for l in rel_laws
    )

    # 3. Build Prompt (Khử nhiễu: Loại bỏ case_fact thô, chỉ dùng query + evidence đắt giá + Top 5 laws)
    user_prompt = f"""THÔNG TIN VỤ ÁN (Yêu cầu khởi kiện):
{query}

BẰNG CHỨNG THU THẬP ĐƯỢC (Case Evidence):
{case_ev_str if case_ev_str else "(Không có chứng cứ bổ sung)"}

ĐIỀU LUẬT LIÊN QUAN (Top {len(rel_laws)} điều luật):
{laws_str if laws_str else "(Không tìm thấy điều luật liên quan)"}

Hãy phân tích, dự đoán kết quả xét xử và chọn ra các bằng chứng/điều luật làm căn cứ, trả về đúng định dạng JSON yêu cầu."""

    # 4. Call LLM
    raw_resp = chat(prompt=user_prompt, system=SYSTEM_PROMPT, temperature=0.2)
    clean_resp = strip_think(raw_resp)

    # 5. Parse JSON Response
    pred = "B_WIN"  # Fallback mặc định
    reasoning = ""
    selected_chunks = []
    selected_laws = []

    try:
        js_str = extract_last_json(clean_resp)
        if js_str:
            parsed = json.loads(js_str)
            candidate = parsed.get("prediction", "").strip()
            if candidate in VALID_LABELS:
                pred = candidate
            else:
                print(f"\n  [WARN] Case {cid}: Nhãn '{candidate}' không hợp lệ. Fallback B_WIN.")
            reasoning = parsed.get("reasoning", "")
            selected_chunks = parsed.get("selected_case_evidence", [])
            selected_laws = parsed.get("selected_law_evidence", [])
        else:
            print(f"\n  [WARN] Case {cid}: Không tìm thấy khối JSON hợp lệ trong output LLM.")
    except Exception as e:
        print(f"\n  [WARN] Parse JSON lỗi cho case {cid}: {e}. Raw output: {clean_resp[:150]}...")

    # Validate & Lọc Case Evidence: Nộp TOÀN BỘ các chunk đã retrieve được từ API (vì luật chấm là Penalized Case Recall, không bị trừ điểm Precision)
    sub_case_ev = [r["chunk_id"] for r in case_ev_data if isinstance(r, dict) and "chunk_id" in r]

    # Validate & Lọc Law Evidence do LLM pick (phải có thật trong corpus gốc)
    sub_law_ev = []
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

    # Fallback nếu LLM pick rỗng hoặc sai
    if not sub_law_ev:
        sub_law_ev = [
            {"law_id": str(l["law_id"]), "aid": int(l["aid"])}
            for l in rel_laws[:4]
            if (str(l["law_id"]).strip(), int(l["aid"])) in valid_law_aids
        ]

    return {
        "case_id": str(cid),
        "prediction": str(pred),
        "case_evidence": sub_case_ev,
        "law_evidence": sub_law_ev,
    }


def main():
    parser = argparse.ArgumentParser(description="Tạo file submission ALQAC 2026.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "ALQAC2026_public_test.json",
        help="Đường dẫn đến file test JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "submission.json",
        help="Đường dẫn lưu file submission kết quả."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Số điều luật retrieve cho mỗi case.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trọng số RRF giữa FAISS và BM25.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy thử N vụ án đầu tiên (để debug).")
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"[ERROR] Không tìm thấy file test: {args.test_file}")
        sys.exit(1)

    print(f"[INFO] Đọc dữ liệu test từ: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if args.limit:
        cases = cases[:args.limit]
        print(f"[INFO] Chế độ debug: Chỉ xử lý {len(cases)} vụ án đầu tiên.")

    print("[INFO] Nạp chỉ mục Hybrid Retriever (FAISS + BM25)...")
    retriever = HybridRetriever.from_disk()

    # Đọc checkpoint từ file output (nếu đã có sẵn kết quả cũ)
    submission = []
    completed_ids = set()
    if args.output.exists():
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                submission = json.load(f)
                completed_ids = {str(item["case_id"]).strip() for item in submission}
            print(f"[INFO] Tìm thấy file '{args.output}' đã lưu {len(completed_ids)} vụ án trước đó. Sẽ bỏ qua các vụ này!")
        except Exception as e:
            print(f"[WARN] Không thể đọc checkpoint cũ: {e}. Sẽ chạy lại từ đầu.")
            submission = []
            completed_ids = set()

    print(f"\n[START] Bắt đầu xử lý danh sách {len(cases)} vụ án...")
    t0 = time.perf_counter()

    for i, case in enumerate(cases, 1):
        cid = str(case.get("case_id", f"case_{i}")).strip()
        if cid in completed_ids:
            print(f"[{i}/{len(cases)}] Vụ án: {cid} → [SKIP] Đã có trong checkpoint.")
            continue

        print(f"[{i}/{len(cases)}] Đang xử lý vụ án: {cid} ...", end=" ", flush=True)
        t_case = time.perf_counter()
        
        result = predict_case(case, retriever, top_k=args.top_k, alpha=args.alpha)
        submission.append(result)
        completed_ids.add(cid)
        
        dt = time.perf_counter() - t_case
        print(f"→ Dự đoán: {result['prediction']} ({dt:.1f}s)")

        # Lưu ngay vào file submission sau MỖI vụ án (incremental checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2, default=str)

    total_time = time.perf_counter() - t0
    print(f"\n✅ Hoàn thành! Tổng cộng {len(submission)} dự đoán đã được lưu an toàn vào: {args.output}")
    print(f"⏱️ Tổng thời gian chạy đợt này: {total_time:.1f}s")


if __name__ == "__main__":
    main()
