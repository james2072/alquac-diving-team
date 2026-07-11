# ALQUAC 2026 – Vietnamese Legal RAG & Verdict Prediction Pipeline

A streamlined **Retrieval-Augmented Generation (RAG)** and **Court Verdict Prediction** system designed for Vietnamese legal texts and civil/commercial case analysis.

The pipeline integrates **Hybrid Search** (combining **FAISS** vector similarity with **BM25** keyword matching via **Reciprocal Rank Fusion - RRF**) to retrieve relevant legal articles and court case evidence, powering an OpenAI-compatible LLM to predict court case outcomes (`A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`).

---

## 🏛️ System Architecture & Workflow

The system strictly decouples online evidence crawling from offline label prediction into a **2-Step Mandatory Pipeline**:

1. **Offline Corpus & Indexing (`rag_runner/`)**:
   - Chunks and embeds Vietnamese legal corpora (`data/corpus/corpus_law_pub.json`) using `BAAI/bge-m3`.
   - Builds dense **FAISS** (`law.faiss`) and lexical **BM25** (`law_bm25.pkl`) search indexes.
2. **Step 1: Pre-fetch Case Evidence (`prefetch_cache.py`)**:
   - Crawls evidence chunks from the competition API with rate-limit compliance (`5.2s` interval) and multi-query expansion ($c_i \le 4$).
   - Saves all retrieved evidence chunks to disk (`data/cache/case_evidence_cache.json`).
3. **Step 2: Offline Verdict Prediction (`generate_submission.py`)**:
   - Reads pre-fetched evidence from disk cache at **zero latency** and retrieves statutory laws via RRF Hybrid Search.
   - Delegates 100% of outcome reasoning to the LLM (`systems prompt` with legal anchors) without any brittle heuristic overrides.
   - **Strict Cache Enforcement**: Throws a `RuntimeError` immediately if any case is missing from local cache, ensuring live API calls are never triggered during label generation.

---

## 📂 Project Structure

```
ALQUAC/                                 ← Root directory
├── .env.example                        ← Template for environment variables
├── requirements.txt                    ← Python package dependencies
├── README.md                           ← Project documentation
├── configs/                            
│   └── config.py                       ← Centralized hyperparameters & truncation limits
├── data/                               
│   ├── corpus/corpus_law_pub.json      ← Vietnamese legal corpus
│   ├── cache/case_evidence_cache.json  ← Disk cache for case evidence chunks
│   ├── test/ALQAC2026_public_test.json ← Benchmark test set
│   └── output/                         ← Parquet embeddings, FAISS, and BM25 indexes
├── rag_runner/                         ← Offline indexing & embedding pipeline
│   ├── build_index.py                  ← CLI script to build FAISS & BM25 indexes
│   ├── corpus_loader.py                ← Parses JSON corpus & applies windowed chunking
│   ├── embedder.py                     ← Computes BGE-M3 embeddings & Parquet caching
│   └── indexer.py                      ← Builds FAISS & BM25Okapi indexes
└── rag_retrieval/                      ← Prediction pipeline & API clients
    ├── prefetch_cache.py               ← Step 1: Pre-fetcher to crawl & cache case evidence
    ├── generate_submission.py          ← Step 2: Automated prediction & submission builder
    ├── hybrid_retriever.py             ← Hybrid search combining FAISS + BM25 via RRF
    ├── evidence_api_client.py          ← Retrieval API client with strict offline cache check
    ├── llm_client.py                   ← OpenAI-compatible LLM wrapper
    ├── test_connection.py              ← Utility to verify LLM endpoint connectivity
    └── utils.py                        ← Shared JSON extraction & text formatting utilities
```

---

## 🚀 Steps to Run

### 1. Install Dependencies & Setup Environment

Ensure you are using Python 3.10+ and install required packages:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your LLM endpoint and competition API credentials:
```ini
LLM_API_KEY="your-llm-api-key"
LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL="gemini-2.5-flash"
ALQAC_TOKEN="your-competition-api-token"
NUM_RESULTS=5
DEFAULT_ALPHA=0.5
```

### 2. Build Search Indexes (Run Once)

Execute the index builder from the **root directory** to generate `law.faiss` and `law_bm25.pkl`:

```bash
python -m rag_runner.build_index
```

### 3. Verify LLM Connection

Test your `.env` credentials and endpoint connectivity:

```bash
python -m rag_retrieval.test_connection
```

### 4. Step 1: Pre-fetch Case Evidence (MANDATORY)

You **MUST** pre-fetch and cache all case evidence chunks from the competition API before predicting labels:

```bash
python -m rag_retrieval.prefetch_cache --test-file data/test/ALQAC2026_public_test.json
```

*Note: This script safely queries the API and saves all segments to `data/cache/case_evidence_cache.json`. Use `--force` if you want to re-fetch existing entries.*

### 5. Step 2: Generate Submission (STRICT OFFLINE MODE)

Run the automated prediction pipeline across the benchmark test cases:

```bash
python -m rag_retrieval.generate_submission --test-file data/test/ALQAC2026_public_test.json --output submission.json
```

> [!IMPORTANT]
> **Strict Offline Pipeline**: `generate_submission` requires all processed cases to exist inside `data/cache/case_evidence_cache.json`. If any case is missing, the program **throws a `RuntimeError` and stops immediately**, guaranteeing zero API calls and maximum execution speed during label prediction.

---

## 🏆 Competition Optimization Summary

$$\text{FinalScore} = 0.70 \times \text{OutcomeAccuracy} + 0.20 \times \text{PenalizedCaseRecall} + 0.10 \times \text{LawF1micro}$$

- **Outcome Accuracy (70%)**: Pure LLM reasoning with comprehensive case facts (`case_fact`), cached decisions (`case_evidence`), and exact laws (`rel_laws`), free of biased rule overrides.
- **Penalized Case Recall (20%)**: Multi-query extraction ($c_i \le 4$) guarantees full credit ($E_i = 1.0$) while adhering to strict `5.2s` API rate limits during pre-fetching.
- **Micro Law F1 (10%)**: RRF Hybrid Search (`alpha=0.5`) combining dense embeddings (`BAAI/bge-m3`) and lexical matches (`BM25`) accurately isolates relevant Civil/Commercial Code articles.
