"""
config.py – Centralised settings loaded from .env

To switch LLM source, only change these 3 variables in .env:
    LLM_API_KEY   – your personal API key for the endpoint
    LLM_BASE_URL  – base URL of the OpenAI-compatible API
    LLM_MODEL     – model name as recognised by that endpoint

Different providers (Google AI Studio, OpenAI, Ollama …) speak the same
OpenAI-compatible protocol, so no other code needs to change.
"""
from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
import os
import torch

# Project root directory (ALQUAC/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from the project root directory
load_dotenv(PROJECT_ROOT / ".env")

# ── LLM (OpenAI-compatible – works with Google AI Studio, OpenAI, Ollama …) ──
LLM_API_KEY: str  = os.getenv("LLM_API_KEY", "")                 # your personal key
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL",                     # endpoint base URL
                               "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL: str    = os.getenv("LLM_MODEL", "gemini-2.5-flash")    # model name

# ── Embedding (always local, no API needed) ───────────────────────────────────
EMBEDDING_MODEL: str  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_MIN_TOKENS: int = int(os.getenv("CHUNK_MIN_TOKENS", "30"))
NUM_RESULTS: int      = int(os.getenv("NUM_RESULTS", "5"))

# ── Paths ─────────────────────────────────────────────────────────────────────
CORPUS_JSON: Path     = (PROJECT_ROOT / "data" / "corpus" / "corpus_law_pub.json").resolve()
EMBEDDINGS_SAVE: Path = (PROJECT_ROOT / "data" / "output" / "law_embeddings.parquet").resolve()
FAISS_INDEX: Path     = (PROJECT_ROOT / "data" / "output" / "law.faiss").resolve()
BM25_INDEX: Path      = (PROJECT_ROOT / "data" / "output" / "law_bm25.pkl").resolve()
