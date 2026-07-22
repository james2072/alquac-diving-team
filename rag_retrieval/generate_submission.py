"""
generate_submission.py – Automated submission pipeline for ALQAC 2026.

Workflow:
  1. Load test cases from JSON (data/test/ALQUAC_test.json).
  2. For each case:
     - Retrieve evidence chunks using API or cache.
     - Retrieve relevant laws using Hybrid Search (FAISS + BM25 + RRF).
     - Format and query the LLM for a legal prediction.
     - Validate and parse the output JSON.
  3. Export results incrementally to submission.json.

Usage:
    python -m rag_retrieval.generate_submission
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
    CORPUS_JSON,
    DEFAULT_ALPHA,
    DEFAULT_SUBMISSION_TOP_K,
    LLM_TEMPERATURE,
    MAX_CASE_EVIDENCE_CHUNK_LEN,
    MAX_CONTEXT_CHUNK_LEN_FOR_SEARCH,
    MAX_CONTEXT_CHUNKS_FOR_SEARCH,
    MAX_FACT_LEN_FOR_SEARCH,
    MAX_LAW_TEXT_LEN_FOR_PROMPT,
    SUBMISSION_FILE,
    TEST_FILE,
)
from rag_retrieval.evidence_api_client import (
    CACHE_FILE,
    _load_cache,
    get_cached_case_queries,
    get_case_evidence,
)
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat_structured
from rag_retrieval.schemas import CasePredictionSchema

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}

# Note: The system prompt remains in Vietnamese to instruct the LLM correctly 
# for Vietnamese legal reasoning.
SYSTEM_PROMPT = """Bạn là Thẩm phán và Chuyên gia Pháp lý Tối cao tại Việt Nam.
Nhiệm vụ của bạn là phân tích toàn diện thông tin yêu cầu khởi kiện, các diễn biến tình tiết (nếu có), các BẰNG CHỨNG THU THẬP ĐƯỢC (đây là các đoạn trích từ hồ sơ/bản án thực tế) và ĐIỀU LUẬT LIÊN QUAN để ra phán quyết chính xác tuyệt đối. Đặc biệt đối với các vụ án bị khuyết thông tin tình tiết, bạn phải "đãi cát tìm vàng" từ chính các đoạn BẰNG CHỨNG THU THẬP ĐƯỢC để tìm ra Quyết định hoặc Nhận định của Tòa án.

PHÂN LOẠI PHÁN QUYẾT (bắt buộc chọn đúng 1 trong 4 nhãn sau):
- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu khởi kiện CHÍNH của nguyên đơn (phần Tòa xét trên thực tế). Lưu ý: nếu nguyên đơn TỰ RÚT một số yêu cầu phụ trước khi xét xử, những phần rút đó KHÔNG được tính vào tỷ lệ bị bác — chỉ tính những phần Tòa BÁC sau khi xem xét.
- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn (phần Tòa bác ≠ nguyên đơn tự rút), với tỷ lệ hoặc giá trị được chấp nhận > 50% tổng yêu cầu Tòa xem xét.
- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, nhưng tỷ lệ hoặc giá trị được chấp nhận <= 50% tổng yêu cầu Tòa xem xét.
- B_WIN: Tòa BÁC TOÀN BỘ (0%) yêu cầu khởi kiện.

2. PHÂN BIỆT "GHI NHẬN THỎA THUẬN" VÀ "CHẤP NHẬN YÊU CẦU": Khi bản án "Ghi nhận sự tự nguyện thỏa thuận...", đây là thỏa thuận ngoài tòa, KHÔNG phải Tòa chấp nhận yêu cầu khởi kiện. Nếu Tòa bác phần còn lại -> B_WIN. Nếu toàn bộ vụ án được giải quyết bằng thỏa thuận (hòa giải thành) mà Tòa không có phán quyết chấp nhận/bác nội dung -> Trả về nhãn B_WIN.
3. PHÂN BIỆT "CHẤP NHẬN" THỦ TỤC VÀ NỘI DUNG: Việc Tòa "chấp nhận" [rút yêu cầu / đơn vắng mặt / thẩm quyền...] là thủ tục. Chỉ đánh giá dựa trên từ khóa "chấp nhận / bác YÊU CẦU KHỞI KIỆN" hoặc "chấp nhận / bác YÊU CẦU của nguyên đơn".
4. ĐỘC LẬP VỚI YÊU CẦU PHẢN TỐ & BÙ TRỪ NGHĨA VỤ: Phân tích CHỈ dựa trên kết quả giải quyết "Yêu cầu khởi kiện của NGUYÊN ĐƠN". Bỏ qua hoàn toàn việc Tòa chấp nhận hay bác "Yêu cầu phản tố của Bị đơn" hoặc "Yêu cầu độc lập của Người liên quan". CHÚ Ý: Nếu Nguyên đơn được Tòa chấp nhận 100% số tiền đòi, nhưng bị Tòa "cấn trừ/bù trừ" vào nợ của Bị đơn (do phản tố) khiến số tiền thực nhận cuối cùng trên bản án ít hơn, thì nhãn của Nguyên đơn VẪN LÀ A_WIN.
5. ĐỊNH LƯỢNG TOÁN HỌC (ĐỐI VỚI YÊU CẦU TÀI SẢN): Tính tỷ lệ chấp nhận = (Giá trị Tòa chấp nhận) / (Tổng giá trị Nguyên đơn đòi mà Tòa xét). Chấp nhận > 50% -> PARTIAL_A_WIN; <= 50% -> PARTIAL_B_WIN. LƯU Ý QUAN TRỌNG: Nếu Tòa chấp nhận TIỀN GỐC đầy đủ nhưng tính tiền lãi THẤP HƠN yêu cầu, đây là PARTIAL. Phải so sánh tổng số tiền Tòa phán quyết với tổng số tiền nguyên đơn đòi. CHÚ Ý BẪY KHẤU TRỪ TIỀN ĐÃ NHẬN (Deduction Trap): Trong án đòi bồi thường hoặc nợ, nếu Tòa xác định tổng giá trị chấp nhận cho nguyên đơn (VD: xác định giá trị bồi thường căn nhà là 42.560.000đ trên tổng đòi 60.000.000đ), sau đó KHẤU TRỪ đi khoản tiền nguyên đơn đã tạm ứng/nhận trước từ bên thứ ba hoặc bị đơn (VD: khấu trừ 26.880.000đ đã nhận) và tuyên buộc bị đơn c
6. ĐỊNH TÍNH (ĐỐI VỚI YÊU CẦU PHI TÀI SẢN): Đối với các yêu cầu không bằng tiền (đòi đất, hủy hợp đồng, xin lỗi công khai, ly hôn...):
   - A_WIN: Tòa chấp nhận toàn bộ các yêu cầu cốt lõi.
   - PARTIAL_A_WIN: Tòa chấp nhận yêu cầu quan trọng nhất (VD: Hủy hợp đồng, đòi được đất) nhưng bác các yêu cầu phụ kiện đi kèm.
   - PARTIAL_B_WIN: Tòa bác yêu cầu cốt lõi, chỉ chấp nhận yêu cầu phụ.
   - B_WIN: Tòa bác toàn bộ các yêu cầu phi tài sản.
7. QUY TẮC SOI LỆCH SỐ LIỆU & PARTIAL ẨN (NUMERICAL MISMATCH -> PARTIAL_A_WIN): Khi Tòa viết "Chấp nhận yêu cầu khởi kiện của nguyên đơn..." nhưng trong chi tiết phán quyết có bất kỳ sự điều chỉnh/cắt giảm nào về tiền lãi suất, tiền phạt vi phạm, hoặc diện tích đất đo đạc thực tế lệch nhẹ so với đơn kiện ban đầu -> BẮT BUỘC gán nhãn PARTIAL_A_WIN. TUYỆT ĐỐI không được gán A_WIN!
8. CHỌN LỌC BẰNG CHỨNG & TRÍCH XUẤT TỐI ĐA ĐIỀU LUẬT (`selected_case_evidence` và `selected_law_evidence`):
   - `selected_case_evidence`: BẮT BUỘC liệt kê chính xác các mã `chunk_id` có chứa phán quyết làm căn cứ cho nhãn.
   - `selected_law_evidence`: BẮT BUỘC liệt kê TỐI ĐA TẤT CẢ các cặp `{"law_id": "...", "aid": ...}` có xuất hiện trong phần Căn cứ hoặc Áp dụng của Tòa án (thường từ 6 đến 15 điều luật). Bạn PHẦR PHẢI trích xuất đầy đủ cả các điều luật nội dung (BLDS, Luật Đất đai...) VÀ toàn bộ các điều luật thủ tục tố tụng/án phí (BLTTDS 2015 Điều 26, 35, 39, 147, 227, Nghị quyết án phí...). LƯU Ý CỰC KỲ QUAN TRỌNG: Trong field `aid`, bạn BẮT BUỘC phải ghi con số Mã hệ thống AID (con số trong ngoặc vuông `[Mã hệ thống AID: ...]`), TUYỆT ĐỐI KHÔNG GHI SỐ ĐIỀU LUẬT!
9. QUY TẮC THỐNG NHẤT VỀ CHẤP NHẬN MỘT PHẦN (PARTIAL) CHO MỌI LOẠI ÁN:
   - Bất kể là vụ án tranh chấp tài sản, đất đai, hợp đồng hay thừa kế/chia tài sản chung: Nếu Tòa án tuyên phán quyết có từ khóa "Chấp nhận một phần yêu cầu khởi kiện" hoặc thực tế Tòa án BÁC bất kỳ phần yêu cầu nội dung nào của nguyên đơn (như yêu cầu bồi thường, chia theo tỷ lệ, yêu cầu phụ trợ kèm theo...), thì BẮT BUỘC phải phân loại là PARTIAL_A_WIN (khi phần được chấp nhận chiếm ưu thế > 50% hoặc là yêu cầu cốt lõi nhất) hoặc PARTIAL_B_WIN (khi phần được chấp nhận <= 50% hoặc chỉ là phần phụ). TUYỆT ĐỐI KHÔNG được gán nhãn A_WIN cho các bản án mà Tòa tuyên chấp nhận một phần hoặc có bác bỏ một phần nội dung.
10. QUY TẮC LỖI HỖN HỢP (CHÍNH XÁC 50%):
    - Trong án bồi thường thiệt hại ngoài hợp đồng, nếu Tòa xác định "lỗi hỗn hợp" và buộc bị đơn bồi thường ĐÚNG BẰNG 50% (tỷ lệ >= 50%) số tiền nguyên đơn yêu cầu -> Gán nhãn PARTIAL_A_WIN.
11. QUY TẮC TRỌNG SỐ YÊU CẦU CỐT LÕI vs PHỤ KHI KHÔNG CÓ TỔNG TIỀN:
    - Nếu nguyên đơn thắng được yêu cầu mục đích chính (đòi được đất/nhà, trả được nợ gốc, hủy sổ đỏ) và chỉ bị bác các khoản phụ (lãi suất, bồi thường thêm) -> PARTIAL_A_WIN (> 50%).
    - Nếu nguyên đơn bị bác yêu cầu chính, chỉ được chấp nhận yêu cầu phụ (như hoàn trả tiền sửa chữa, tiền cọc) -> PARTIAL_B_WIN (<= 50%).

VÍ DỤ SUY LUẬN MẪU (COMPREHENSIVE FEW-SHOT EXAMPLES):

[VÍ DỤ 1 - A_WIN: CHẤP NHẬN TOÀN BỘ]
- Đầu vào: Nguyên đơn đòi bị đơn trả số tiền vay 500.000.000 đồng và tiền lãi suất 50.000.000 đồng theo Hợp đồng vay ngày 10/01/2020.
- Bằng chứng: `[case_101_chunk_3]` Tòa án nhận định hợp đồng vay là hợp pháp. Bị đơn vi phạm nghĩa vụ thanh toán theo thỏa thuận. Quyết định: Chấp nhận toàn bộ yêu cầu khởi kiện của nguyên đơn, buộc bị đơn trả cho nguyên đơn đủ 550.000.000 đồng.
- Điều luật: `- Luật 91/2015/QH13 | Điều 466 [Mã hệ thống AID: 53236]: Nghĩa vụ trả nợ của bên vay...`, `- Luật 92/2015/QH13 | Điều 147 [Mã hệ thống AID: 52917]: Vị phí tố tụng...`
- Đầu ra JSON:
{
  "reasoning": "Tòa án xác định hợp đồng hợp pháp và bị đơn vi phạm nghĩa vụ thanh toán theo Điều 466 BLDS 2015. Tòa chấp nhận toàn bộ 100% yêu cầu trả nợ gốc và lãi (550 triệu đồng).",
  "selected_case_evidence": ["case_101_chunk_3"],
  "selected_law_evidence": [
    {"law_id": "91/2015/QH13", "aid": 53236},
    {"law_id": "92/2015/QH13", "aid": 52917}
  ],
  "prediction": "A_WIN"
}

[VÍ DỤ 2 - PARTIAL_A_WIN: CHẤP NHẬN MỘT PHẦN > 50% VÀ TRÍCH XUẤT TỐI ĐA ĐIỀU LUẬT]
- Đầu vào: Nguyên đơn khởi kiện đòi bồi thường thiệt hại do tai nạn tổng số tiền 100.000.000 đồng (gồm viện phí 70.000.000 đồng và tổn thất tinh thần 30.000.000 đồng).
- Bằng chứng: `[case_202_chunk_2]` Áp dụng các Điều 26, 35, 39, 147, 227 Bộ luật Tố tụng Dân sự 2015; Điều 584, 590 Bộ luật Dân sự 2015; Nghị quyết số 326/2016/UBTVQH14. Quyết định: Chấp nhận một phần yêu cầu khởi kiện, buộc bị đơn bồi thường viện phí 70.000.000 đồng, bác yêu cầu bồi thường tổn thất tinh thần.
- Điều luật: `- Luật 91/2015/QH13 | Điều 584 [Mã hệ thống AID: 53354]...`, `- Luật 91/2015/QH13 | Điều 590 [Mã hệ thống AID: 53360]...`, `- Luật 92/2015/QH13 | Điều 26 [Mã hệ thống AID: 52796]...`, `- Luật 92/2015/QH13 | Điều 35 [Mã hệ thống AID: 52805]...`, `- Luật 92/2015/QH13 | Điều 39 [Mã hệ thống AID: 52809]...`, `- Luật 92/2015/QH13 | Điều 147 [Mã hệ thống AID: 52917]...`, `- Luật 92/2015/QH13 | Điều 227 [Mã hệ thống AID: 52997]...`, `- Luật 326/2016/UBTVQH14 | Điều 26 [Mã hệ thống AID: 50691]...`
- Đầu ra JSON:
{
  "reasoning": "Tòa chấp nhận phần bồi thường viện phí 70.000.000 đồng trên tổng yêu cầu 100.000.000 đồng (đạt tỷ lệ 70% > 50%), bác phần yêu cầu tổn thất tinh thần. Trích xuất đầy đủ 8 căn cứ pháp lý nội dung và thủ tục Tòa đã áp dụng.",
  "selected_case_evidence": ["case_202_chunk_2"],
  "selected_law_evidence": [
    {"law_id": "91/2015/QH13", "aid": 53354},
    {"law_id": "91/2015/QH13", "aid": 53360},
    {"law_id": "92/2015/QH13", "aid": 52796},
    {"law_id": "92/2015/QH13", "aid": 52805},
    {"law_id": "92/2015/QH13", "aid": 52809},
    {"law_id": "92/2015/QH13", "aid": 52917},
    {"law_id": "92/2015/QH13", "aid": 52997},
    {"law_id": "326/2016/UBTVQH14", "aid": 50691}
  ],
  "prediction": "PARTIAL_A_WIN"
}

[VÍ DỤ 3 - PARTIAL_B_WIN: CHẤP NHẬN MỘT PHẦN <= 50%]
- Đầu vào: Nguyên đơn yêu cầu thanh toán tiền phạt vi phạm 200.000.000 đồng và bồi thường thiệt hại 300.000.000 đồng (tổng 500.000.000 đồng).
- Bằng chứng: `[case_303_chunk_1]` Tòa chấp nhận tiền phạt vi phạm 150.000.000 đồng, bác phần bồi thường thiệt hại do thiếu chứng cứ. Quyết định: Chấp nhận một phần yêu cầu khởi kiện.
- Điều luật: `- Luật 36/2005/QH11 | Điều 300 [Mã hệ thống AID: 52870]...`, `- Luật 92/2015/QH13 | Điều 147 [Mã hệ thống AID: 52917]...`
- Đầu ra JSON:
{
  "prediction": "PARTIAL_B_WIN",
  "reasoning": "Tòa chấp nhận yêu cầu phạt vi phạm 150.000.000 đồng trên tổng số 500.000.000 đồng nguyên đơn đòi (đạt tỷ lệ 30% <= 50%), bác phần bồi thường theo Điều 300 Luật Thương mại 2005.",
  "selected_case_evidence": ["case_303_chunk_1"],
  "selected_law_evidence": [
    {"law_id": "36/2005/QH11", "aid": 52870},
    {"law_id": "92/2015/QH13", "aid": 52917}
  ]
}

[VÍ DỤ 4 - B_WIN: BÁC TOÀN BỘ YÊU CẦU / YÊU CẦU PHI TÀI SẢN]
- Đầu vào: Nguyên đơn yêu cầu công nhận di chúc viết tay ngày 12/05/2018 là hợp pháp và chia di sản thừa kế theo di chúc.
- Bằng chứng: `[case_404_chunk_5]` Tòa kết luận di chúc viết tay không tuân thủ quy định pháp luật. Quyết định: Bác toàn bộ yêu cầu khởi kiện của nguyên đơn.
- Điều luật: `- Luật 91/2015/QH13 | Điều 630 [Mã hệ thống AID: 53393]...`
- Đầu ra JSON:
{
  "prediction": "B_WIN",
  "reasoning": "Di chúc viết tay vi phạm điều kiện có hiệu lực pháp luật theo Điều 630 BLDS 2015. Tòa bác toàn bộ 100% yêu cầu cốt lõi của nguyên đơn.",
  "selected_case_evidence": ["case_404_chunk_5"],
  "selected_law_evidence": [
    {"law_id": "91/2015/QH13", "aid": 53393}
  ]
}

[VÍ DỤ 5 - A_WIN: ĐỘC LẬP VỚI PHẢN TỐ & BÙ TRỪ NGHĨA VỤ]
- Đầu vào: Nguyên đơn đòi tiền hàng 1.000.000.000 đồng. Bị đơn phản tố đòi bồi thường 300.000.000 đồng.
- Bằng chứng: `[case_505_chunk_4]` Tòa chấp nhận yêu cầu của nguyên đơn (1 tỷ), chấp nhận yêu cầu phản tố của bị đơn (300 triệu). Đối trừ nghĩa vụ, buộc bị đơn thanh toán cho nguyên đơn 700.000.000 đồng.
- Điều luật: `- Luật 91/2015/QH13 | Điều 378 [Mã hệ thống AID: 53241]...`
- Đầu ra JSON:
{
  "prediction": "A_WIN",
  "reasoning": "Tòa án chấp nhận 100% yêu cầu khởi kiện của nguyên đơn (1.000.000.000 đồng). Số tiền thực nhận giảm xuống 700.000.000 đồng là do bù trừ nghĩa vụ với yêu cầu phản tố theo Điều 378 BLDS 2015, không phải do Tòa bác yêu cầu.",
  "selected_case_evidence": ["case_505_chunk_4"],
  "selected_law_evidence": [
    {"law_id": "91/2015/QH13", "aid": 53241}
  ]
}

Bạn PHẢI trả lời CHỈ DUY NHẤT một object JSON hợp lệ theo đúng định dạng (tuyệt đối không thêm bất kỳ văn bản giải thích nào ngoài JSON):
{
  "prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN",
  "reasoning": "Giải thích ngắn gọn căn cứ phán quyết và định lượng tỷ lệ",
  "selected_case_evidence": ["danh_sách_mã_chunk_id_đã_dùng"],
  "selected_law_evidence": [{"law_id": "mã_luật", "aid": Mã_hệ_thống_AID}]
}
Lưu ý quan trọng: Trong field 'aid' của selected_law_evidence, BẮT BUỘC ghi giá trị Mã hệ thống AID (con số nguyên trong ngoặc vuông [Mã hệ thống AID: ...]) tương ứng với Điều luật áp dụng, tuyệt đối không ghi số Điều luật vào field aid!

CHẾ ĐỘ SUY LUẬN TỪ EVIDENCE (khi không có TÌNH TIẾT CHI TIẾT hoặc QUYẾT ĐỊNH TÒA):
Khi dữ liệu đầu vào CHỈ CÓ tóm tắt yêu cầu khởi kiện và bằng chứng thu thập được (evidence chunks), bạn phải:
1. ƯU TIÊN TÌM PHÁN QUYẾT TRONG EVIDENCE CHUNKS: Evidence chunks thường chứa trích đoạn bản án gốc. Tìm keyword: "Quyết định", "Chấp nhận toàn bộ", "Chấp nhận một phần", "Bác toàn bộ", "Buộc", "Tuyên xử", "đình chỉ xét xử".
2. NẾU TÌM THẤY PHÁN QUYẾT TRONG EVIDENCE → áp dụng quy tắc phân loại nhãn (A_WIN/PARTIAL_A_WIN/PARTIAL_B_WIN/B_WIN) như bình thường.
3. NẾU KHÔNG TÌM THẤY PHÁN QUYẾT → suy luận dựa trên tình tiết trong evidence, loại tranh chấp, chứng cứ hợp đồng/giấy tờ, và điều luật áp dụng.
4. Evidence chunks có thể chứa cả nhận định của Hội đồng xét xử → đây là nguồn quan trọng nhất để xác định nhãn."""

_AID_TO_ARTICLE: dict[int, tuple[str, int]] = {}
_ARTICLE_TO_AID: dict[tuple[str, int], int] = {}

def _load_article_mappings() -> None:
    if _AID_TO_ARTICLE:
        return
    corpus_path = CORPUS_JSON
    if corpus_path.exists():
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for law in data:
                lid = str(law.get("law_id", "")).strip()
                for idx, art in enumerate(law.get("content", [])):
                    try:
                        aid = int(art["aid"])
                        art_num = idx + 1
                        _AID_TO_ARTICLE[aid] = (lid, art_num)
                        _ARTICLE_TO_AID[(lid, art_num)] = aid
                    except (ValueError, KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"  [WARN] Could not load {CORPUS_JSON.name} mappings: {e}")

def _retrieve_evidence(
    case: dict[str, Any], retriever: HybridRetriever, top_k: int, alpha: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Retrieve both case evidence (API/Cache) and legal text (Hybrid Search)."""
    _load_article_mappings()
    cid = str(case.get("case_id", "")).strip()
    query = str(case.get("case_query", ""))
    fact = str(case.get("case_fact", ""))
    a_desc = str(case.get("A_description", ""))
    b_desc = str(case.get("B_description", ""))

    # 1. Retrieve case evidence strictly from local disk cache
    case_ev_data = get_case_evidence(cid, query, case_fact=fact, case=case, strict_cache=True)
    
    if isinstance(case_ev_data, list):
        case_ev_data = sorted(
            [r for r in case_ev_data if isinstance(r, dict) and r.get("text")],
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )

    case_ev_str = "\n".join(
        f"- [{r.get('chunk_id', '')}] {str(r.get('text', '')).strip()[:MAX_CASE_EVIDENCE_CHUNK_LEN]}"
        for r in case_ev_data
    )

    # 2. Retrieve laws using focused, high-precision legal query (avoiding noisy 60k character evidence text flooding)
    cached_queries = get_cached_case_queries(cid)
    llm_questions_str = "\n".join(cached_queries[1:]) if len(cached_queries) > 1 else ""
    party_context = f"{a_desc[:300]} {b_desc[:300]}".strip()
    
    # Combined search query: Substantive claim + LLM investigative query variations + fact summary
    if fact or party_context:
        q_combined = f"{query}\n{party_context}\n{llm_questions_str}\n{fact[:1000]}".strip()
    else:
        q_combined = f"{query}\n{llm_questions_str}".strip()
    rel_laws = retriever.search(q_combined, k=top_k, alpha=alpha)
    
    laws_formatted = []
    for l in rel_laws:
        lid = str(l["law_id"]).strip()
        aid = int(l["aid"])
        art_info = _AID_TO_ARTICLE.get(aid)
        if art_info and art_info[0] == lid:
            art_label = f"Điều {art_info[1]} [Mã hệ thống AID: {aid}]"
        else:
            art_label = f"Điều (Mã hệ thống AID: {aid})"
        snippet = str(l.get("text", "")).strip()[:MAX_LAW_TEXT_LEN_FOR_PROMPT]
        laws_formatted.append(f"- Luật {lid} | {art_label}: {snippet}")
    laws_str = "\n".join(laws_formatted)
    
    return case_ev_data, rel_laws, case_ev_str, laws_str


def _build_prompt(case: dict[str, Any], case_ev_str: str, laws_str: str) -> str:
    """Construct the user prompt for the LLM.
    
    Supports two modes:
    - Full mode: When case has rich metadata (fact, verdict, reasoning, party descriptions).
    - Evidence-only mode: When case only has case_id + case_query (e.g., private test).
    """
    query = str(case.get("case_query", ""))
    fact = str(case.get("case_fact", "")).strip()
    case_type = str(case.get("case_type", "")).strip()
    court_level = str(case.get("court_level", "")).strip()
    a_role = str(case.get("A_role", "")).strip()
    a_desc = str(case.get("A_description", "")).strip()
    b_role = str(case.get("B_role", "")).strip()
    b_desc = str(case.get("B_description", "")).strip()
    court_verdict = str(case.get("court_verdict", "")).strip()
    court_reasoning = str(case.get("court_reasoning", "")).strip()

    # Detect evidence-only mode: no fact, no verdict, no reasoning
    has_rich_context = bool(fact or court_verdict or court_reasoning)

    if has_rich_context:
        # === FULL MODE (public test with 20 fields) ===
        party_info = f"- {a_role or 'Nguyên đơn'} (Bên A - tương ứng nhãn A_WIN): {a_desc if a_desc else '(Không có chi tiết)'}\n- {b_role or 'Bị đơn'} (Bên B - tương ứng nhãn B_WIN): {b_desc if b_desc else '(Không có chi tiết)'}"

        verdict_section = ""
        if court_verdict:
            verdict_section += f"\nQUYẾT ĐỊNH CỦA TÒA ÁN (Court Verdict):\n{court_verdict[:MAX_CASE_EVIDENCE_CHUNK_LEN]}"
        if court_reasoning:
            verdict_section += f"\n\nNHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ (Court Reasoning):\n{court_reasoning[:MAX_CASE_EVIDENCE_CHUNK_LEN]}"

        return f"""THÔNG TIN TỔNG QUAN VỤ ÁN:
- Loại vụ án: {case_type or 'Dân sự'} | Cấp xét xử: {court_level or 'Sơ thẩm'}
{party_info}

TÓM TẮT YÊU CẦU KHỞI KIỆN / TRANH CHẤP:
{query}

TÌNH TIẾT VỤ ÁN CHI TIẾT:
{fact if fact else "(Không có tình tiết bổ sung)"}
{verdict_section}

BẰNG CHỨNG THU THẬP ĐƯỢC:
{case_ev_str if case_ev_str else "(Không có chứng cứ bổ sung)"}

ĐIỀU LUẬT LIÊN QUAN:
{laws_str if laws_str else "(Không tìm thấy điều luật liên quan)"}

Hãy phân tích kỹ QUYẾT ĐỊNH và NHẬN ĐỊNH của Tòa án (nếu có), tư cách các bên, yêu cầu khởi kiện, tình tiết chi tiết và quy định pháp luật để xác định kết quả xét xử. Trả về đúng định dạng JSON yêu cầu."""
    else:
        # === EVIDENCE-ONLY MODE (private test with 2 fields) ===
        return f"""THÔNG TIN VỤ ÁN (Chế độ Evidence-Only — chỉ có tóm tắt yêu cầu khởi kiện):
{query}

LƯU Ý QUAN TRỌNG: Bạn KHÔNG có tình tiết chi tiết, KHÔNG có quyết định hay nhận định trực tiếp của Tòa.
Bạn phải phân tích DỰA HOÀN TOÀN vào:
1. NỘI DUNG YÊU CẦU KHỞI KIỆN ở trên (xác định loại tranh chấp, các bên, số tiền, yêu cầu cụ thể)
2. BẰNG CHỨNG THU THẬP ĐƯỢC (evidence chunks từ hồ sơ vụ án — có thể chứa phán quyết/nhận định Tòa)
3. ĐIỀU LUẬT LIÊN QUAN

BẰNG CHỨNG THU THẬP ĐƯỢC:
{case_ev_str if case_ev_str else "(Không có chứng cứ bổ sung)"}

ĐIỀU LUẬT LIÊN QUAN:
{laws_str if laws_str else "(Không tìm thấy điều luật liên quan)"}

HƯỚNG DẪN SUY LUẬN:
- TÌM PHÁN QUYẾT TRONG EVIDENCE: Tìm keyword "Chấp nhận", "Bác", "Quyết định", "Buộc", "Tuyên xử" trong evidence chunks
- Nếu evidence chứa phán quyết/nhận định Tòa → dùng chúng để xác định nhãn chính xác theo các quy tắc phân loại
- Nếu evidence không có phán quyết → suy luận dựa trên tình tiết, chứng cứ, và luật pháp áp dụng
- Trả về đúng định dạng JSON yêu cầu."""


def _parse_and_validate(
    parsed: CasePredictionSchema | None,
    cid: str,
    case_ev_data: list[dict[str, Any]],
    rel_laws: list[dict[str, Any]],
    valid_law_aids: set[tuple[str, int]],
) -> dict[str, Any]:
    """Validate structured Pydantic output (`CasePredictionSchema`)."""
    pred: str | None = parsed.prediction if parsed and parsed.prediction in VALID_LABELS else None

    # Prioritize LLM-selected case evidence chunks, falling back to all retrieved chunks if empty
    valid_cids = {str(r["chunk_id"]) for r in case_ev_data if isinstance(r, dict) and "chunk_id" in r}
    sub_case_ev: list[str] = []
    if parsed and parsed.selected_case_evidence:
        for cid_item in parsed.selected_case_evidence:
            clean_cid = str(cid_item).strip()
            if clean_cid in valid_cids and clean_cid not in sub_case_ev:
                sub_case_ev.append(clean_cid)
    if not sub_case_ev:
        sub_case_ev = [str(r["chunk_id"]) for r in case_ev_data if isinstance(r, dict) and "chunk_id" in r]

    # Validate and recover selected laws
    _load_article_mappings()
    sub_law_ev: list[dict[str, Any]] = []

    def _normalize_lid(raw_lid: str) -> str:
        clean = raw_lid.strip()
        for _, (real_l, _) in _AID_TO_ARTICLE.items():
            if clean == real_l:
                return real_l
        lower = clean.lower()
        aliases = {
            "91/2015/QH13": ["dân sự năm 2015", "dân sư\u0323 năm 2015", "dân sự 2015", "bộ luật dân sự", "blds 2015", "blds"],
            "92/2015/QH13": ["tố tụng dân sự năm 2015", "tố tụng dân sự", "bộ luật tố tụng dân sự", "blttds", "blttds 2015"],
            "33/2005/QH11": ["dân sự năm 2005", "dân sự 2005"],
            "36/2005/QH11": ["thương mại năm 2005", "thương mại 2005", "thương mại"],
            "26/2008/QH12": ["thi hành án dân sự"],
            "52/2014/QH13": ["hôn nhân và gia đình", "hôn nhân gia đình", "hngđ", "luật hôn nhân gia đình"],
            "45/2013/QH13": ["lao động", "luật lao động 2013", "bộ luật lao động"],
            "66/2014/QH13": ["kinh doanh bất động sản"],
            "45/2019/QH14": ["lao động năm 2019", "lao động 2019"],
            "326/2016/UBTVQH14": ["326/2016", "nghị quyết số 326", "án phí"],
        }
        for true_l, keywords in aliases.items():
            if true_l.lower() == lower:
                return true_l
            for kw in keywords:
                if kw in lower:
                    return true_l
        return clean

    if parsed and parsed.selected_law_evidence:
        for le in parsed.selected_law_evidence:
            try:
                lid = _normalize_lid(str(le.law_id).strip())
                raw_aid = int(le.aid)

                # Check 1: Direct unique AID match across entire system index
                if raw_aid in _AID_TO_ARTICLE:
                    real_lid, _ = _AID_TO_ARTICLE[raw_aid]
                    if not any(x["aid"] == raw_aid for x in sub_law_ev):
                        sub_law_ev.append({"law_id": real_lid, "aid": raw_aid})
                    continue

                # Check 2: If LLM outputted Article Number (Điều X) instead of AID -> recover via normalized lid + art_num
                if (lid, raw_aid) in _ARTICLE_TO_AID:
                    mapped_aid = _ARTICLE_TO_AID[(lid, raw_aid)]
                    if not any(x["aid"] == mapped_aid for x in sub_law_ev):
                        sub_law_ev.append({"law_id": lid, "aid": mapped_aid})
                    continue

                # Check 3: If normalized lid didn't match exactly, check against valid retrieved candidate articles
                for v_lid, v_aid in valid_law_aids:
                    if v_aid in _AID_TO_ARTICLE and _AID_TO_ARTICLE[v_aid][1] == raw_aid:
                        if (v_lid == lid) or ("91/2015" in lid and "91/2015" in v_lid) or ("92/2015" in lid and "92/2015" in v_lid):
                            if not any(x["aid"] == v_aid for x in sub_law_ev):
                                sub_law_ev.append({"law_id": v_lid, "aid": v_aid})
                            break
            except (ValueError, TypeError):
                pass

    return {
        "case_id": str(cid),
        "prediction": pred,
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

    if retriever.valid_law_aids is None:
        retriever.valid_law_aids = {
            (str(row["law_id"]).strip(), int(row["aid"]))
            for _, row in retriever.df.iterrows()
        }

    case_ev_data, rel_laws, case_ev_str, laws_str = _retrieve_evidence(
        case, retriever, top_k, alpha
    )

    user_prompt = _build_prompt(case, case_ev_str, laws_str)
    
    max_retries = 3
    res: dict[str, Any] = _parse_and_validate(None, cid, case_ev_data, rel_laws, retriever.valid_law_aids)
    for attempt in range(max_retries):
        try:
            parsed = chat_structured(
                prompt=user_prompt,
                response_format=CasePredictionSchema,
                system=SYSTEM_PROMPT,
                temperature=LLM_TEMPERATURE,
            )
            res = _parse_and_validate(parsed, cid, case_ev_data, rel_laws, retriever.valid_law_aids)
            if res.get("prediction") in VALID_LABELS:
                return res
        except Exception as e:
            print(f"\n  [WARN] Attempt {attempt + 1}/{max_retries} encountered API error: {e}")
            res = _parse_and_validate(None, cid, case_ev_data, rel_laws, retriever.valid_law_aids)
            
        if attempt < max_retries - 1:
            print(f"\n  [RETRY {attempt + 1}/{max_retries}] Case {cid} returned invalid prediction or error. Waiting 4.0s before retry...")
            time.sleep(4.0)

    # If all retries exhausted and LLM still returned None/invalid, prevent script crash by defaulting prediction
    if res.get("prediction") not in VALID_LABELS:
        print(f"\n  [ERROR] All {max_retries} LLM attempts failed to produce a valid label for {cid}. Setting fallback prediction to B_WIN to prevent crash.")
        res["prediction"] = "B_WIN"

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

    print(f"[INFO] Verifying strict offline cache at {CACHE_FILE}...")
    cache = _load_cache()
    missing_cases = [
        str(c.get("case_id", "")).strip()
        for c in cases
        if str(c.get("case_id", "")).strip() not in cache
        or not (
            isinstance(cache.get(str(c.get("case_id", "")).strip()), (dict, list))
            and (
                (isinstance(cache.get(str(c.get("case_id", "")).strip()), dict) and "results" in cache.get(str(c.get("case_id", "")).strip()))
                or (isinstance(cache.get(str(c.get("case_id", "")).strip()), list))
            )
        )
    ]
    if missing_cases:
        raise RuntimeError(
            f"\n================================================================================\n"
            f"[ERROR] Strict Offline Pipeline Enforcement: Missing case evidence in cache!\n"
            f"Found {len(missing_cases)} case(s) without cached evidence (e.g., {missing_cases[:5]}).\n"
            f"To prevent API calls during label prediction and maintain high Case Recall,\n"
            f"you MUST prefetch all cases before running generate_submission.\n\n"
            f"Please run the prefetch step first:\n"
            f"    python -m rag_retrieval.prefetch_cache\n"
            f"================================================================================\n"
        )

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
        
        result = predict_case(case, retriever)
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
