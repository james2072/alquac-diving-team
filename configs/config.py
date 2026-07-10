"""
config.py – Centralized configuration and hyperparameters loaded from .env.

To switch LLM source, only change these 3 variables in .env:
    LLM_API_KEY   – your personal API key for the endpoint
    LLM_BASE_URL  – base URL of the OpenAI-compatible API
    LLM_MODEL     – model name as recognized by that endpoint

Different providers (Google AI Studio, OpenAI, Ollama …) speak the same
OpenAI-compatible protocol, so no other code needs to change.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Project root directory (ALQUAC/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from the project root directory
load_dotenv(PROJECT_ROOT / ".env")

# ── LLM Configuration (OpenAI-compatible) ─────────────────────────────────────
LLM_API_KEY: str  = os.getenv("LLM_API_KEY", "")                 # Personal API key
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL",                     # Endpoint base URL
                               "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL: str    = os.getenv("LLM_MODEL", "gemini-2.5-flash")    # Model name

LLM_MAX_RETRIES: int         = int(os.getenv("LLM_MAX_RETRIES", "8"))
LLM_RETRY_SLEEP_SUCCESS: int = int(os.getenv("LLM_RETRY_SLEEP_SUCCESS", "2"))
LLM_MAX_TOKENS: int          = int(os.getenv("LLM_MAX_TOKENS", "8192"))
LLM_TEMPERATURE: float       = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_CHAT_TEMPERATURE: float  = float(os.getenv("LLM_CHAT_TEMPERATURE", "0.7"))

# ── Embedding Configuration (Local) ───────────────────────────────────────────
EMBEDDING_MODEL: str  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── RAG & Retrieval Hyperparameters ───────────────────────────────────────────
CHUNK_MIN_TOKENS: int = int(os.getenv("CHUNK_MIN_TOKENS", "10"))
NUM_RESULTS: int      = int(os.getenv("NUM_RESULTS", "5"))
DEFAULT_TOP_K_LAWS: int = int(os.getenv("DEFAULT_TOP_K_LAWS", "6"))
DEFAULT_SUBMISSION_TOP_K: int = int(os.getenv("DEFAULT_SUBMISSION_TOP_K", "10"))

# Reciprocal Rank Fusion (RRF) constants
RRF_K: int                = int(os.getenv("RRF_K", "60"))
DEFAULT_ALPHA: float      = float(os.getenv("DEFAULT_ALPHA", "0.5"))
CANDIDATE_MULTIPLIER: int = int(os.getenv("CANDIDATE_MULTIPLIER", "5"))

# API Retrieval settings
API_URL: str            = os.getenv("ALQAC_API_URL", "https://alqac-api.ngrok.pro/retrieve")
REQUEST_TIMEOUT: int    = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES: int        = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_429: float  = float(os.getenv("RETRY_DELAY_429", "6.0"))
RETRY_DELAY_NORMAL: float = float(os.getenv("RETRY_DELAY_NORMAL", "5.2"))
RETRY_DELAY_ERROR: float  = float(os.getenv("RETRY_DELAY_ERROR", "3.0"))
MAX_CALLS_PER_CASE: int = int(os.getenv("MAX_CALLS_PER_CASE", "6"))
MIN_SCORE: float        = float(os.getenv("MIN_SCORE", "0.5"))
DEFAULT_MAX_CHUNKS: int = int(os.getenv("DEFAULT_MAX_CHUNKS", "20"))

# Query & Context Truncation limits
MAX_QUERY_LENGTH: int               = int(os.getenv("MAX_QUERY_LENGTH", "1500"))
SUB_QUERY_LENGTH: int               = int(os.getenv("SUB_QUERY_LENGTH", "800"))
MAX_EVIDENCE_LENGTH_FOR_SEARCH: int = int(os.getenv("MAX_EVIDENCE_LENGTH_FOR_SEARCH", "6000"))
MIN_SEARCH_TEXT_LENGTH: int         = int(os.getenv("MIN_SEARCH_TEXT_LENGTH", "50"))
MAX_EVIDENCE_LENGTH_FOR_PROMPT: int = int(os.getenv("MAX_EVIDENCE_LENGTH_FOR_PROMPT", "12000"))
MAX_CASE_EVIDENCE_CHUNK_LEN: int    = int(os.getenv("MAX_CASE_EVIDENCE_CHUNK_LEN", "6000"))
MAX_LAW_CONTENT_LENGTH: int         = int(os.getenv("MAX_LAW_CONTENT_LENGTH", "3000"))
MAX_LAW_TEXT_LEN_FOR_PROMPT: int    = int(os.getenv("MAX_LAW_TEXT_LEN_FOR_PROMPT", "6000"))
MAX_CONTEXT_CHUNKS_FOR_SEARCH: int  = int(os.getenv("MAX_CONTEXT_CHUNKS_FOR_SEARCH", "15"))
MAX_CONTEXT_CHUNK_LEN_FOR_SEARCH: int = int(os.getenv("MAX_CONTEXT_CHUNK_LEN_FOR_SEARCH", "4000"))
MAX_FACT_LEN_FOR_SEARCH: int        = int(os.getenv("MAX_FACT_LEN_FOR_SEARCH", "6000"))

# Chunking & Splitting
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
CHUNK_STRIDE: int     = int(os.getenv("CHUNK_STRIDE", "256"))

# ── Scoring Bonuses & Keyword Lists ───────────────────────────────────────────
VERDICT_KEYWORD_BOOST: float  = float(os.getenv("VERDICT_KEYWORD_BOOST", "5.0"))
CITATION_KEYWORD_BOOST: float = float(os.getenv("CITATION_KEYWORD_BOOST", "2.0"))

VERDICT_KEYWORDS: list[str] = [
    "quyết định",
    "tuyên xử",
    "chấp nhận",
    "không chấp nhận",
    "bác yêu cầu",
    "bác toàn bộ",
    "bác đơn",
    "bác bỏ",
    "án phí",
    "nhận định của tòa án",
    "hội đồng xét xử nhận định",
    "nhận định của hội đồng xét xử",
]

RULE_KEYWORDS_ACCEPT_ALL: list[str] = [
    "chấp nhận toàn bộ",
    "chấp nhận yêu cầu của nguyên đơn",
    "chấp nhận yêu cầu khởi kiện",
    "chấp nhận đơn khởi kiện",
]

RULE_KEYWORDS_REJECT_ALL: list[str] = [
    "không chấp nhận yêu cầu",
    "không chấp nhận toàn bộ",
    "không chấp nhận đơn",
    "bác toàn bộ",
    "bác yêu cầu",
    "bác đơn",
    "bác bỏ yêu cầu",
    "không có căn cứ chấp nhận",
    "không có căn cứ để chấp nhận",
    "chưa có căn cứ để chấp nhận",
    "không đủ căn cứ để chấp nhận",
]

RULE_KEYWORDS_PARTIAL: list[str] = [
    "chấp nhận một phần",
    "chấp nhận 1 phần",
    "một phần yêu cầu",
    "1 phần yêu cầu",
    "bác một phần",
    "bác 1 phần",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
CORPUS_JSON: Path     = (PROJECT_ROOT / "data" / "corpus" / "corpus_law_pub.json").resolve()
EMBEDDINGS_SAVE: Path = (PROJECT_ROOT / "data" / "output" / "law_embeddings.parquet").resolve()
FAISS_INDEX: Path     = (PROJECT_ROOT / "data" / "output" / "law.faiss").resolve()
BM25_INDEX: Path      = (PROJECT_ROOT / "data" / "output" / "law_bm25.pkl").resolve()

CACHE_FILE: Path      = (PROJECT_ROOT / "data" / "cache" / "case_evidence_cache.json").resolve()
TEST_FILE: Path       = (PROJECT_ROOT / "data" / "test" / "ALQAC2026_public_test.json").resolve()
SUBMISSION_FILE: Path = (PROJECT_ROOT / "submission.json").resolve()
EXTERNAL_DIR: Path    = (PROJECT_ROOT / "data" / "external").resolve()
