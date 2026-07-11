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
from rag_retrieval.evidence_api_client import (
    CACHE_FILE,
    _load_cache,
    get_cached_case_queries,
    get_case_evidence,
)
from rag_retrieval.hybrid_retriever import HybridRetriever
from rag_retrieval.llm_client import chat
from rag_retrieval.utils import extract_json_from_text, strip_think_tags

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}

# Note: The system prompt remains in Vietnamese to instruct the LLM correctly 
# for Vietnamese legal reasoning.
SYSTEM_PROMPT = """Bạn là Thẩm phán và Chuyên gia Pháp lý Tối cao tại Việt Nam.
Nhiệm vụ của bạn là phân tích toàn diện thông tin yêu cầu khởi kiện (THÔNG TIN VỤ ÁN), diễn biến tình tiết (TÌNH TIẾT VỤ ÁN CHI TIẾT), chứng cứ bổ sung (BẰNG CHỨNG THU THẬP ĐƯỢC) và quy định pháp luật (ĐIỀU LUẬT LIÊN QUAN) để ra phán quyết chính xác tuyệt đối.

PHÂN LOẠI PHÁN QUYẾT (bắt buộc chọn đúng 1 trong 4 nhãn sau):
- A_WIN: Tòa chấp nhận TOÀN BỘ yêu cầu khởi kiện CHÍNH của nguyên đơn (phần Tòa xét trên thực tế). Lưu ý: nếu nguyên đơn TỰ RÚT một số yêu cầu phụ trước khi xét xử, những phần rút đó KHÔNG được tính vào tỷ lệ bị bác — chỉ tính những phần Tòa BÁC sau khi xem xét.
- PARTIAL_A_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn (phần Tòa bác ≠ nguyên đơn tự rút), với tỷ lệ hoặc giá trị được chấp nhận > 50% tổng yêu cầu Tòa xem xét.
- PARTIAL_B_WIN: Tòa chấp nhận MỘT PHẦN yêu cầu của nguyên đơn, nhưng tỷ lệ hoặc giá trị được chấp nhận <= 50% tổng yêu cầu Tòa xem xét.
- B_WIN: Tòa BÁC TOÀN BỘ (0%) yêu cầu khởi kiện của nguyên đơn (hoặc không có căn cứ chấp nhận).

NGUYÊN TẮC SUY LUẬN & ĐỊNH LƯỢNG BẮT BUỘC:
1. PHÂN BIỆT TỰ RÚT VÀ BỊ BÁC: Khi nguyên đơn tự rút yêu cầu tại phiên tòa (đình chỉ xét xử một phần), phần đó KHÔNG được tính là Tòa bác. Chỉ xét tỷ lệ trên phần Tòa thực sự giải quyết bằng phán quyết nội dung.
2. PHÂN BIỆT "GHI NHẬN THỎA THUẬN" VÀ "CHẤP NHẬN YÊU CẦU": Khi bản án "Ghi nhận sự tự nguyện thỏa thuận...", đây là thỏa thuận ngoài tòa, KHÔNG phải Tòa chấp nhận yêu cầu khởi kiện. Nếu Tòa bác phần còn lại -> B_WIN. Nếu toàn bộ vụ án được giải quyết bằng thỏa thuận (hòa giải thành) mà Tòa không có phán quyết chấp nhận/bác nội dung -> Trả về nhãn B_WIN.
3. PHÂN BIỆT "CHẤP NHẬN" THỦ TỤC VÀ NỘI DUNG: Việc Tòa "chấp nhận" [rút yêu cầu / đơn vắng mặt / thẩm quyền...] là thủ tục. Chỉ đánh giá dựa trên từ khóa "chấp nhận / bác YÊU CẦU KHỞI KIỆN" hoặc "chấp nhận / bác YÊU CẦU của nguyên đơn".
4. ĐỘC LẬP VỚI YÊU CẦU PHẢN TỐ & BÙ TRỪ NGHĨA VỤ: Phân tích CHỈ dựa trên kết quả giải quyết "Yêu cầu khởi kiện của NGUYÊN ĐƠN". Bỏ qua hoàn toàn việc Tòa chấp nhận hay bác "Yêu cầu phản tố của Bị đơn" hoặc "Yêu cầu độc lập của Người liên quan". CHÚ Ý: Nếu Nguyên đơn được Tòa chấp nhận 100% số tiền đòi, nhưng bị Tòa "cấn trừ/bù trừ" vào nợ của Bị đơn (do phản tố) khiến số tiền thực nhận cuối cùng trên bản án ít hơn, thì nhãn của Nguyên đơn VẪN LÀ A_WIN.
5. ĐỊNH LƯỢNG TOÁN HỌC (ĐỐI VỚI YÊU CẦU TÀI SẢN): Tính tỷ lệ chấp nhận = (Giá trị Tòa chấp nhận) / (Tổng giá trị Nguyên đơn đòi mà Tòa xét). Chấp nhận > 50% -> PARTIAL_A_WIN; <= 50% -> PARTIAL_B_WIN. LƯU Ý QUAN TRỌNG: Nếu Tòa chấp nhận TIỀN GỐC đầy đủ nhưng tính tiền lãi THẤP HƠN yêu cầu, đây là PARTIAL. Phải so sánh tổng số tiền Tòa phán quyết với tổng số tiền nguyên đơn đòi.
6. ĐỊNH TÍNH (ĐỐI VỚI YÊU CẦU PHI TÀI SẢN): Đối với các yêu cầu không bằng tiền (đòi đất, hủy hợp đồng, xin lỗi công khai, ly hôn...):
   - A_WIN: Tòa chấp nhận toàn bộ các yêu cầu cốt lõi.
   - PARTIAL_A_WIN: Tòa chấp nhận yêu cầu quan trọng nhất (VD: Hủy hợp đồng, đòi được đất) nhưng bác các yêu cầu phụ kiện đi kèm.
   - PARTIAL_B_WIN: Tòa bác yêu cầu cốt lõi, chỉ chấp nhận yêu cầu phụ.
   - B_WIN: Tòa bác toàn bộ các yêu cầu phi tài sản.
7. NHẬN DIỆN PARTIAL ẨN (IMPLICIT PARTIAL): Khi Tòa viết "Chấp nhận yêu cầu khởi kiện" nhưng số tiền phán quyết (hoặc diện tích đất cấp) THẤP HƠN yêu cầu ban đầu -> Đây là PARTIAL.
8. CHỌN LỌC BẰNG CHỨNG CHUẨN XÁC (`selected_case_evidence` và `selected_law_evidence`):
   - `selected_case_evidence`: BẮT BUỘC liệt kê chính xác các mã `chunk_id` có chứa phán quyết, nhận định của Tòa án làm căn cứ cho nhãn bạn chọn.
   - `selected_law_evidence`: BẮT BUỘC liệt kê chính xác các cặp `{"law_id": "...", "aid": ...}` được Tòa áp dụng.

VÍ DỤ SUY LUẬN MẪU (COMPREHENSIVE FEW-SHOT EXAMPLES):

[VÍ DỤ 1 - A_WIN: CHẤP NHẬN TOÀN BỘ]
- Đầu vào: Nguyên đơn đòi bị đơn trả số tiền vay 500.000.000 đồng và tiền lãi suất 50.000.000 đồng theo Hợp đồng vay ngày 10/01/2020.
- Bằng chứng: `[case_101_chunk_3]` Tòa án nhận định hợp đồng vay là hợp pháp. Bị đơn vi phạm nghĩa vụ thanh toán theo thỏa thuận. Quyết định: Chấp nhận toàn bộ yêu cầu khởi kiện của nguyên đơn, buộc bị đơn trả cho nguyên đơn đủ 550.000.000 đồng.
- Điều luật: `- Luật 91/2015/QH13 Điều 466: Nghĩa vụ trả nợ của bên vay...`
- Đầu ra JSON:
{
  "prediction": "A_WIN",
  "reasoning": "Tòa án xác định hợp đồng hợp pháp và bị đơn vi phạm nghĩa vụ thanh toán theo Điều 466 BLDS 2015. Tòa chấp nhận toàn bộ 100% yêu cầu trả nợ gốc và lãi (550 triệu đồng).",
  "selected_case_evidence": ["case_101_chunk_3"],
  "selected_law_evidence": [{"law_id": "91/2015/QH13", "aid": 466}]
}

[VÍ DỤ 2 - PARTIAL_A_WIN: CHẤP NHẬN MỘT PHẦN > 50%]
- Đầu vào: Nguyên đơn khởi kiện đòi bồi thường thiệt hại do tai nạn tổng số tiền 100.000.000 đồng (gồm chi phí điều trị viện phí 70.000.000 đồng và bồi thường tổn thất tinh thần 30.000.000 đồng).
- Bằng chứng: `[case_202_chunk_2]` Tòa xét thấy bị đơn có lỗi gây ra tai nạn nên phải chịu trách nhiệm bồi thường chi phí điều trị y tế thực tế là 70.000.000 đồng. Đối với khoản tổn thất tinh thần 30.000.000 đồng không có cơ sở chứng minh nên không được chấp nhận. Quyết định: Chấp nhận một phần yêu cầu khởi kiện của nguyên đơn, buộc bị đơn bồi thường 70.000.000 đồng.
- Điều luật: `- Luật 91/2015/QH13 Điều 584: Căn cứ phát sinh trách nhiệm bồi thường...`, `- Luật 91/2015/QH13 Điều 590: Thiệt hại do sức khỏe bị xâm phạm...`
- Đầu ra JSON:
{
  "prediction": "PARTIAL_A_WIN",
  "reasoning": "Tòa chấp nhận phần bồi thường viện phí hợp lệ là 70.000.000 đồng trên tổng yêu cầu 100.000.000 đồng (đạt tỷ lệ 70% > 50%), bác phần yêu cầu tổn thất tinh thần. Căn cứ theo Điều 584 và 590 BLDS 2015.",
  "selected_case_evidence": ["case_202_chunk_2"],
  "selected_law_evidence": [{"law_id": "91/2015/QH13", "aid": 584}, {"law_id": "91/2015/QH13", "aid": 590}]
}

[VÍ DỤ 3 - PARTIAL_B_WIN: CHẤP NHẬN MỘT PHẦN <= 50%]
- Đầu vào: Nguyên đơn yêu cầu buộc bị đơn thanh toán tiền phạt vi phạm hợp đồng 200.000.000 đồng và bồi thường thiệt hại kinh doanh 300.000.000 đồng (tổng yêu cầu là 500.000.000 đồng).
- Bằng chứng: `[case_303_chunk_1]` Tòa nhận định thỏa thuận phạt vi phạm 150.000.000 đồng là đúng quy định. Tuy nhiên nguyên đơn không cung cấp được chứng cứ chứng minh thiệt hại thực tế 300.000.000 đồng nên bác yêu cầu này. Quyết định: Chấp nhận một phần yêu cầu khởi kiện, buộc bị đơn thanh toán tiền phạt vi phạm 150.000.000 đồng.
- Điều luật: `- Luật 36/2005/QH11 Điều 300: Phạt vi phạm...`
- Đầu ra JSON:
{
  "prediction": "PARTIAL_B_WIN",
  "reasoning": "Tòa chấp nhận yêu cầu phạt vi phạm 150.000.000 đồng trên tổng số 500.000.000 đồng nguyên đơn đòi (đạt tỷ lệ 30% <= 50%), bác phần bồi thường thiệt hại do thiếu chứng cứ theo Điều 300 Luật Thương mại 2005.",
  "selected_case_evidence": ["case_303_chunk_1"],
  "selected_law_evidence": [{"law_id": "36/2005/QH11", "aid": 300}]
}

[VÍ DỤ 4 - B_WIN: BÁC TOÀN BỘ YÊU CẦU / YÊU CẦU PHI TÀI SẢN]
- Đầu vào: Nguyên đơn yêu cầu công nhận di chúc viết tay ngày 12/05/2018 là hợp pháp và chia di sản thừa kế theo di chúc.
- Bằng chứng: `[case_404_chunk_5]` Tòa án giám định và kết luận di chúc viết tay không tuân thủ quy định pháp luật, người lập di chúc khi đó mất năng lực hành vi dân sự. Quyết định: Bác toàn bộ yêu cầu khởi kiện của nguyên đơn.
- Điều luật: `- Luật 91/2015/QH13 Điều 630: Điều kiện hợp pháp của di chúc...`
- Đầu ra JSON:
{
  "prediction": "B_WIN",
  "reasoning": "Di chúc viết tay vi phạm điều kiện có hiệu lực pháp luật theo Điều 630 BLDS 2015. Tòa bác toàn bộ 100% yêu cầu cốt lõi (phi tài sản) của nguyên đơn.",
  "selected_case_evidence": ["case_404_chunk_5"],
  "selected_law_evidence": [{"law_id": "91/2015/QH13", "aid": 630}]
}

[VÍ DỤ 5 - A_WIN: ĐỘC LẬP VỚI PHẢN TỐ & BÙ TRỪ NGHĨA VỤ]
- Đầu vào: Nguyên đơn yêu cầu bị đơn trả tiền hàng 1.000.000.000 đồng. Bị đơn phản tố yêu cầu nguyên đơn bồi thường vi phạm hợp đồng 300.000.000 đồng.
- Bằng chứng: `[case_505_chunk_4]` Tòa xét thấy yêu cầu trả tiền hàng 1.000.000.000 đồng của nguyên đơn là có cơ sở. Yêu cầu phản tố 300.000.000 đồng của bị đơn cũng có cơ sở. Quyết định: Chấp nhận yêu cầu của nguyên đơn (1 tỷ), chấp nhận yêu cầu phản tố của bị đơn (300 triệu). Đối trừ nghĩa vụ, buộc bị đơn thanh toán cho nguyên đơn 700.000.000 đồng.
- Điều luật: `- Luật 91/2015/QH13 Điều 378: Bù trừ nghĩa vụ...`
- Đầu ra JSON:
{
  "prediction": "A_WIN",
  "reasoning": "Tòa án chấp nhận 100% yêu cầu khởi kiện của nguyên đơn (1.000.000.000 đồng). Việc số tiền thực nhận giảm xuống 700.000.000 đồng là do bù trừ nghĩa vụ với yêu cầu phản tố của bị đơn theo Điều 378 BLDS 2015, không phải do Tòa bác yêu cầu của nguyên đơn.",
  "selected_case_evidence": ["case_505_chunk_4"],
  "selected_law_evidence": [{"law_id": "91/2015/QH13", "aid": 378}]
}

Bạn PHẢI trả lời CHỈ DUY NHẤT một object JSON hợp lệ theo đúng định dạng (tuyệt đối không thêm bất kỳ văn bản giải thích nào ngoài JSON):
{
  "prediction": "A_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN" | "B_WIN",
  "reasoning": "Giải thích ngắn gọn căn cứ phán quyết và định lượng tỷ lệ",
  "selected_case_evidence": ["danh_sách_mã_chunk_id_đã_dùng"],
  "selected_law_evidence": [{"law_id": "mã_luật", "aid": Mã_hệ_thống_AID}]
}
Lưu ý quan trọng: Trong field 'aid' của selected_law_evidence, BẮT BUỘC ghi giá trị Mã hệ thống AID (con số nguyên trong ngoặc vuông [Mã hệ thống AID: ...]) tương ứng với Điều luật áp dụng, tuyệt đối không ghi số Điều luật vào field aid!"""

_AID_TO_ARTICLE: dict[int, tuple[str, int]] = {}
_ARTICLE_TO_AID: dict[tuple[str, int], int] = {}

def _load_article_mappings() -> None:
    if _AID_TO_ARTICLE:
        return
    corpus_path = PROJECT_ROOT / "data" / "corpus" / "corpus_law_pub.json"
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
            print(f"  [WARN] Could not load corpus_law_pub.json mappings: {e}")

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

    # 2. Retrieve laws by combining query, party descriptions, LLM investigative questions, fact, and retrieved evidence context
    ev_context = " ".join(
        [str(r.get("text", '')).strip()[:MAX_CONTEXT_CHUNK_LEN_FOR_SEARCH] 
         for r in case_ev_data if isinstance(r, dict) and r.get("text")][:MAX_CONTEXT_CHUNKS_FOR_SEARCH]
    )
    cached_queries = get_cached_case_queries(cid)
    llm_questions_str = "\n".join(cached_queries[1:]) if len(cached_queries) > 1 else ""
    party_context = f"{a_desc[:300]} {b_desc[:300]}".strip()
    
    search_query = f"{query}\n{party_context}\n{llm_questions_str}\n{fact[:MAX_FACT_LEN_FOR_SEARCH]}\n{ev_context}"
    rel_laws = retriever.search(search_query, k=top_k, alpha=alpha)
    
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
    """Construct the user prompt for the LLM including full case metadata, party roles, query, and detailed facts."""
    query = str(case.get("case_query", ""))
    fact = str(case.get("case_fact", ""))
    case_type = str(case.get("case_type", "Dân sự"))
    court_level = str(case.get("court_level", "Sơ thẩm"))
    a_role = str(case.get("A_role", "Nguyên đơn"))
    a_desc = str(case.get("A_description", ""))
    b_role = str(case.get("B_role", "Bị đơn"))
    b_desc = str(case.get("B_description", ""))
    court_verdict = str(case.get("court_verdict", "")).strip()
    court_reasoning = str(case.get("court_reasoning", "")).strip()

    party_info = f"- {a_role} (Bên A - tương ứng nhãn A_WIN): {a_desc if a_desc else '(Không có chi tiết)'}\n- {b_role} (Bên B - tương ứng nhãn B_WIN): {b_desc if b_desc else '(Không có chi tiết)'}"

    verdict_section = ""
    if court_verdict:
        verdict_section += f"\nQUYẾT ĐỊNH CỦA TÒA ÁN (Court Verdict):\n{court_verdict[:MAX_CASE_EVIDENCE_CHUNK_LEN]}"
    if court_reasoning:
        verdict_section += f"\n\nNHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ (Court Reasoning):\n{court_reasoning[:MAX_CASE_EVIDENCE_CHUNK_LEN]}"

    return f"""THÔNG TIN TỔNG QUAN VỤ ÁN:
- Loại vụ án: {case_type} | Cấp xét xử: {court_level}
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
    if isinstance(parsed, list) and parsed:
        parsed = next((item for item in parsed if isinstance(item, dict)), {})
    
    if isinstance(parsed, dict) and parsed:
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

    # Validate selected laws (supporting both direct AID matches and auto-recovery from Article numbers)
    _load_article_mappings()
    sub_law_ev: list[dict[str, Any]] = []
    if isinstance(selected_laws, list):
        for le in selected_laws:
            if isinstance(le, dict) and "law_id" in le and "aid" in le:
                try:
                    lid = str(le["law_id"]).strip()
                    raw_aid = int(le["aid"])
                    # Check 1: Direct AID match
                    if (lid, raw_aid) in valid_law_aids:
                        if not any(x["law_id"] == lid and x["aid"] == raw_aid for x in sub_law_ev):
                            sub_law_ev.append({"law_id": lid, "aid": raw_aid})
                    # Check 2: If LLM outputted Article Number (Điều X) instead of AID -> Auto-recover
                    elif (lid, raw_aid) in _ARTICLE_TO_AID:
                        mapped_aid = _ARTICLE_TO_AID[(lid, raw_aid)]
                        if (lid, mapped_aid) in valid_law_aids:
                            if not any(x["law_id"] == lid and x["aid"] == mapped_aid for x in sub_law_ev):
                                sub_law_ev.append({"law_id": lid, "aid": mapped_aid})
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
        case, retriever, top_k, alpha
    )

    user_prompt = _build_prompt(case, case_ev_str, laws_str)
    raw_resp = chat(prompt=user_prompt, system=SYSTEM_PROMPT, temperature=LLM_TEMPERATURE)
    
    res = _parse_and_validate(raw_resp, cid, case_ev_data, rel_laws, retriever.valid_law_aids)

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
