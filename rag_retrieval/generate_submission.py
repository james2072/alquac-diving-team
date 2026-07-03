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

SYSTEM_PROMPT = """Bạn là một Thẩm phán và Luật sư xét xử cấp cao, chuyên sâu về pháp luật dân sự, hành chính và thương mại Việt Nam.
Nhiệm vụ của bạn: Đọc kỹ yêu cầu khởi kiện, tóm tắt sự kiện vụ án và danh sách điều luật liên quan, sau đó đưa ra phán quyết chính xác nhất.

Định nghĩa 4 nhãn phán quyết (Dựa trên mức độ Tòa án chấp nhận yêu cầu khởi kiện của Nguyên đơn):
- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu khởi kiện của nguyên đơn (Nguyên đơn đúng 100%, bị đơn sai hoàn toàn).
- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu khởi kiện của nguyên đơn (> 50%). Thường xảy ra khi: có lỗi hỗn hợp (cả nguyên đơn và bị đơn đều có lỗi dẫn đến thiệt hại), hoặc một phần số tiền đòi bồi thường không có hóa đơn/căn cứ hợp lý.
- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu khởi kiện của nguyên đơn (<= 50%).
- B_WIN: Tòa BÁC TOÀN BỘ yêu cầu khởi kiện của nguyên đơn (Nguyên đơn không có căn cứ pháp lý hoặc lỗi hoàn toàn do nguyên đơn).

QUY TRÌNH PHÂN TÍCH PHÁP LÝ TỪNG BƯỚC (CHAIN-OF-THOUGHT):
Trong tâm trí, hãy thực hiện phân tích IRAC (Issue - Rule - Application - Conclusion):
1. Hành vi vi phạm: Bị đơn có hành vi vi phạm pháp luật hay gây ra thiệt hại không?
2. Lỗi hỗn hợp (Đặc biệt quan trọng): Nguyên đơn có sơ suất hay vi phạm quy tắc an toàn nào góp phần làm xảy ra tai nạn/thiệt hại không? Nếu CẢ HAI BÊN CÙNG CÓ LỖI (Lỗi hỗn hợp theo Điều 584/585 Bộ luật Dân sự), Tòa bắt buộc phải chia đôi hoặc giảm trừ trách nhiệm bồi thường -> tuyên PARTIAL_A_WIN hoặc PARTIAL_B_WIN!
3. Chốt kết quả: Chọn 1 trong 4 nhãn phản ánh đúng tỷ lệ chấp nhận yêu cầu khởi kiện.

Bạn PHẢI trả lời CHỈ DUY NHẤT một object JSON hợp lệ, không thêm bất kỳ văn bản giải thích nào bên ngoài JSON, đúng định dạng:
{"prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN", "reasoning": "Phân tích ngắn gọn lỗi từng bên 1-2 câu"}"""


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
    top_k: int = 15,
    alpha: float = 0.5,
) -> dict:
    """Dự đoán kết quả cho 1 vụ án."""
    cid = str(case.get("case_id", "unknown")).strip()
    query = case.get("case_query", "")
    fact = case.get("case_fact", "")

    # 1. Retrieve laws (Hybrid Search với top_k = 15 để độ phủ sâu rộng hơn)
    search_query = f"{query}\n{fact[:1500]}"
    rel_laws = retriever.search(search_query, k=top_k, alpha=alpha)
    
    laws_str = "\n".join(
        f"- Luật {l['law_id']} Điều {l['aid']}: {l['text'][:600]}..."
        for l in rel_laws
    )

    # 2. Build Prompt
    user_prompt = f"""THÔNG TIN VỤ ÁN (Yêu cầu khởi kiện):
{query}

TÓM TẮT NỘI DUNG SỰ KIỆN VỤ ÁN:
{fact[:2500]}

CÁC ĐIỀU LUẬT LIÊN QUAN ĐƯỢC TÌM THẤY (Top {len(rel_laws)} điều luật):
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
    parser.add_argument("--top-k", type=int, default=15, help="Số điều luật retrieve cho mỗi case.")
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
