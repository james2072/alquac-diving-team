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

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}

SYSTEM_PROMPT = """Bạn là một luật sư và thẩm phán rất giỏi, am hiểu sâu sắc pháp luật Việt Nam.
Nhiệm vụ của bạn: đọc thông tin vụ án, nội dung sự kiện và điều luật liên quan, sau đó dự đoán kết quả xét xử.

Định nghĩa các nhãn (dựa trên mức độ Tòa án chấp nhận yêu cầu của nguyên đơn):
- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu của nguyên đơn.
- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, và phần được chấp nhận > 50%.
- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, nhưng phần được chấp nhận <= 50%.
- B_WIN: Tòa BÁC TOÀN BỘ yêu cầu của nguyên đơn.

Bạn PHẢI trả lời CHỈ DUY NHẤT một object JSON hợp lệ, không thêm bất kỳ văn bản giải thích nào khác bên ngoài JSON, đúng định dạng:
{"prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN", "reasoning": "Lý do ngắn gọn"}"""


def strip_think(text: str) -> str:
    """Loại bỏ khối <think>...</think> nếu LLM sử dụng reasoning mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_last_json(text: str) -> str | None:
    """Tìm khối {...} cuối cùng trong text trả về từ LLM."""
    matches = list(re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL))
    if not matches:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return None
    return matches[-1].group(0)


def predict_case(
    case: dict,
    retriever: HybridRetriever,
    top_k: int = 5,
    alpha: float = 0.5,
) -> dict:
    """Dự đoán kết quả cho 1 vụ án."""
    cid = str(case.get("case_id", "unknown")).strip()
    query = case.get("case_query", "")
    fact = case.get("case_fact", "")

    # 1. Retrieve laws (Hybrid Search)
    search_query = f"{query}\n{fact[:500]}"
    rel_laws = retriever.search(search_query, k=top_k, alpha=alpha)
    
    laws_str = "\n".join(
        f"- Luật {l['law_id']} Điều {l['aid']}: {l['text'][:400]}..."
        for l in rel_laws
    )

    # 2. Build Prompt
    user_prompt = f"""THÔNG TIN VỤ ÁN (Yêu cầu khởi kiện):
{query}

TÓM TẮT NỘI DUNG SỰ KIỆN VỤ ÁN:
{fact[:2000]}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐƯỢC TÌM THẤY:
{laws_str if laws_str else "(Không tìm thấy điều luật liên quan)"}

Hãy phân tích và dự đoán kết quả xét xử, trả về đúng định dạng JSON yêu cầu."""

    # 3. Call LLM
    raw_resp = chat(prompt=user_prompt, system=SYSTEM_PROMPT, temperature=0.2)
    clean_resp = strip_think(raw_resp)

    # 4. Parse JSON Response
    pred = "B_WIN"  # Fallback mặc định
    reasoning = ""
    try:
        js_str = extract_last_json(clean_resp)
        if js_str:
            parsed = json.loads(js_str)
            candidate = parsed.get("prediction", "").strip()
            if candidate in VALID_LABELS:
                pred = candidate
            reasoning = parsed.get("reasoning", "")
    except Exception as e:
        print(f"  [WARN] Parse JSON lỗi cho case {cid}: {e}. Raw output: {clean_resp[:100]}...")

    # Format chuẩn hóa cho submission theo đúng yêu cầu schema ALQAC 2026
    return {
        "case_id": str(cid),
        "prediction": str(pred),
        "case_evidence": [],  # Bắt buộc theo schema cuộc thi (được phép rỗng [])
        "law_evidence": [
            {
                "law_id": str(l["law_id"]),
                "aid": int(l["aid"]),
            }
            for l in rel_laws
        ],
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
    parser.add_argument("--top-k", type=int, default=NUM_RESULTS, help="Số điều luật retrieve cho mỗi case.")
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
