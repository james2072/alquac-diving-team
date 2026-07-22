# ⚖️ ALQUAC 2026 – Vietnamese Legal RAG & Verdict Prediction

## General Information
- **Site**: https://sites.google.com/view/alqac2026/home?authuser=0
- **Leaderboard & Submissions**: https://alqac2026-leaderboard.ngrok.app 
---

A high-performance **Retrieval-Augmented Generation (RAG)** and **Court Verdict Prediction** pipeline for Vietnamese legal texts and civil/commercial case analysis.

The system combines **Hybrid Search** (FAISS dense vector similarity + BM25Okapi lexical keyword matching via **Reciprocal Rank Fusion - RRF**) with structured **LLM reasoning** to accurately predict court verdicts (`A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`) and retrieve governing statutory laws.

---

## Architecture & Pipeline Flow

The workflow is strictly decoupled into **2 mandatory steps** to eliminate runtime API latency and ensure maximum precision:

1. **Step 1: Evidence Pre-fetching (`rag_retrieval/prefetch_cache.py`)**
   - Crawls case evidence chunks from the competition API using rate-limit compliance (`5.2s` interval) and smart multi-query expansion ($c_i \le 5$).
   - Saves all evidence chunks to disk (`data/cache/case_evidence_cache.json`).

2. **Step 2: Verdict & Law Prediction (`rag_retrieval/generate_submission.py`)**
   - Reads evidence chunks strictly from local disk cache (`data/cache/case_evidence_cache.json`).
   - Retrieves top statutory law articles using **RRF Hybrid Search** (`alpha = 0.5`).
   - Generates legal verdict predictions via **Pydantic Structured Outputs** (`CasePredictionSchema`) to guarantee valid JSON labels without parsing errors.

---

## Directory Structure

```
ALQUAC/
├── .env.example                        # Template for environment configuration
├── README.md                           # Project documentation
├── configs/
│   └── config.py                       # configurations & hyperparameters
├── data/
│   ├── corpus/corpus_law.json      # Vietnamese statutory legal corpus
│   ├── cache/case_evidence_cache.json  # Cached case evidence (API)
│   ├── test/ALQUAC_test.json           # Benchmark evaluation test set
│   └── output/                         # Embeddings (Parquet), FAISS, and BM25 indexes
├── rag_runner/                         # Indexing module
│   ├── requirements.txt                # Dependencies for embedding & indexing
│   ├── build_index.py                  # CLI builder for FAISS & BM25 indexes
│   ├── corpus_loader.py                # Corpus loader & text chunking
│   ├── embedder.py                     # BGE-M3 embedding computation & Parquet storage
│   └── indexer.py                      # FAISS + BM25Okapi index builder
└── rag_retrieval/                      # Retrieval & inference module
    ├── requirements.txt                # Dependencies for RAG & LLM inference
    ├── prefetch_cache.py               # Step 1: Pre-fetch & cache evidence chunks
    ├── generate_submission.py          # Step 2: Prediction & submission generator
    ├── hybrid_retriever.py             # Hybrid search (FAISS + BM25Okapi via RRF)
    ├── evidence_api_client.py          # API client with rate-limit handling & cache fallback
    ├── schemas.py                      # Pydantic schemas for structured output validation
    ├── llm_client.py                   # LLM client wrapper
    └── test_connection.py              # LLM endpoint connectivity test utility
```

---

## How to run

### 1. Setup Environment & Configuration

Using `Python 3.14.3`, then install requirements:

```bash
pip install -r rag_retrieval/requirements.txt
```

Create `.env` from template:

```bash
cp .env.example .env
```

Configure your LLM credentials and competition token in `.env`:

```ini
LLM_API_KEY="your-llm-api-key"
LLM_BASE_URL="your-llm-api-provider-url"
LLM_MODEL="your-llm-model"
ALQAC_TOKEN="your-competition-api-token"
```

---

### 2. Verify LLM Connection

Test connectivity to your configured LLM endpoint:

```bash
python -m rag_retrieval.test_connection
```

---

### 3. Build Search Indexes (Run Once)

Generate FAISS dense index (`law.faiss`) and BM25 lexical index (`law_bm25.pkl`):

```bash
python -m rag_runner.build_index
```

---

### 4. Step 1: Pre-fetch Case Evidence (Mandatory)

Pre-fetch and cache evidence chunks from the competition API (uses default input `data/test/ALQUAC_test.json`):

```bash
python -m rag_retrieval.prefetch_cache
```

---

### 5. Step 2: Generate Submission File

Execute the structured prediction pipeline to generate `submission.json`:

```bash
python -m rag_retrieval.generate_submission
```

---

## ⚙️ Execution Modes & File Overrides

All scripts use pre-configured **smart defaults in `configs/config.py`**, allowing execution without flags. You can optionally override input test files or output target paths using CLI flags:

| Script | Default Command (Shortest) | Default Test File (Input) | Override Test File (`--test-file`) | Default Output File | Override Output File (`--output`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1: Pre-fetch Cache** | `python -m rag_retrieval.prefetch_cache` | `data/test/ALQUAC_test.json` | `--test-file <path/to/test.json>` | `data/cache/case_evidence_cache.json` | N/A *(saves to cache)* |
| **Step 2: Generate Submission** | `python -m rag_retrieval.generate_submission` | `data/test/ALQUAC_test.json` | `--test-file <path/to/test.json>` | `submission.json` | `--output <path/to/custom_submission.json>` |

