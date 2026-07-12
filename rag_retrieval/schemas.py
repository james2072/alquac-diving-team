"""
schemas.py – Pydantic models for structured LLM output (Zod-like schema validation).

Enforces strict JSON schema at the token generation level using Structured Outputs,
preventing invalid labels, hallucinated fields, or markdown clutter.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class LawEvidenceSchema(BaseModel):
    """Structured item for selected statutory law evidence."""
    law_id: str = Field(..., description="Mã hiệu văn bản luật, ví dụ: '91/2015/QH13' hoặc '45/2013/QH13'.")
    aid: int = Field(..., description="Mã hệ thống AID (Article ID) của điều luật trong cơ sở dữ liệu.")


class CasePredictionSchema(BaseModel):
    """Strict Zod-like schema for court verdict prediction and legal evidence selection."""
    prediction: Literal["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"] = Field(
        ..., description="Nhãn phán quyết chính xác của vụ án. Bắt buộc chọn đúng 1 trong 4 nhãn."
    )
    reasoning: str = Field(
        ..., description="Lý do và phân tích ngắn gọn dẫn tới phán quyết."
    )
    selected_case_evidence: list[str] = Field(
        default_factory=list, description="Danh sách các mã chunk_id chứa phán quyết/nhận định Tòa án."
    )
    selected_law_evidence: list[LawEvidenceSchema] = Field(
        default_factory=list, description="Danh sách các điều luật áp dụng (chỉ chọn các mã AID có trong danh sách đã cung cấp)."
    )
