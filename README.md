# ALQUAC 2026 – Vietnamese Legal RAG & Verdict Prediction Pipeline

A professional **Retrieval-Augmented Generation (RAG)** and **Court Verdict Prediction** system designed for Vietnamese legal texts and civil/commercial case analysis. 

The pipeline integrates **Hybrid Search** (combining **FAISS** semantic vector similarity with **BM25** lexical keyword matching via **Reciprocal Rank Fusion - RRF**) to retrieve relevant legal articles and court case evidence, powering an OpenAI-compatible LLM to predict court case outcomes with high precision.

---

## 🏛️ System Architecture

1. **Centralized Configuration (`configs/config.py`)**: A single source of truth for all system hyperparameters, file paths, BM25 scoring bonuses, truncation limits, and LLM retry backoff strategies.
2. **Offline Corpus & Indexing (`rag_runner/`)**:
   - Loads Vietnamese legal corpora (`data/corpus/corpus_law_pub.json`).
   - Splits long articles into overlapping semantic windows (`MAX_CHUNK_TOKENS`, `CHUNK_STRIDE`).
   - Computes dense vector embeddings using `BAAI/bge-m3` and stores them in high-performance Parquet format (`data/output/law_embeddings.parquet`).
   - Builds both dense **FAISS** (`law.faiss`) and lexical **BM25** (`law_bm25.pkl`) search indexes.
3. **Online Retrieval & Prediction (`rag_retrieval/`)**:
   - **Hybrid Retriever**: Merges FAISS and BM25 results using Reciprocal Rank Fusion (RRF) with adjustable semantic weight (`alpha`).
   - **Evidence Retrieval**: Retrieves case evidence chunks from local disk cache (`data/cache/case_evidence_cache.json`) with zero latency penalty, applying BM25 ranking and structural verdict keyword bonuses.
   - **Verdict Prediction Pipeline**: Analyzes plaintiff claims, evidence, and laws to predict 4-class court verdicts (`A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`), backed by deterministic legal keyword rule overrides and robust negation filtering.

---

## 📂 Project Structure

```
ALQUAC/                                 ← Root directory
├── .env.example                        ← Template for environment variables
├── requirements.txt                    ← Python package dependencies
├── README.md                           ← Project documentation
├── configs/                            
│   └── config.py                       ← Centralized system configurations & hyperparameters
├── data/                               
│   ├── corpus/
│   │   └── corpus_law_pub.json         ← Vietnamese legal corpus
│   ├── cache/
│   │   └── case_evidence_cache.json    ← Disk cache for case evidence chunks
│   ├── test/
│   │   └── ALQAC2026_public_test.json  ← Public benchmark evaluation set
│   └── output/                         ← Built Parquet embeddings, FAISS, and BM25 indexes
├── rag_runner/                         ← Offline indexing & embedding pipeline
│   ├── build_index.py                  ← CLI script to build Parquet, FAISS, and BM25 indexes
│   ├── corpus_loader.py                ← Parses JSON corpus & applies windowed chunking
│   ├── embedder.py                     ← Computes BGE-M3 embeddings & manages Parquet caching
│   └── indexer.py                      ← Builds FAISS IndexFlatIP & BM25Okapi indexes
└── rag_retrieval/                      ← Online RAG, hybrid search & prediction pipeline
    ├── generate_submission.py          ← Automated ALQAC 2026 prediction & submission builder
    ├── prefetch_cache.py               ← Automated pre-fetcher to crawl and cache case evidence
    ├── rag.py                          ← Main RAG pipeline & interactive query solver
    ├── hybrid_retriever.py             ← Hybrid search combining FAISS + BM25 via RRF
    ├── case_api.py                     ← Cache-only case evidence BM25 retriever
    ├── evidence_api_client.py          ← API client with exponential backoff & rate-limiting
    ├── llm_client.py                   ← OpenAI-compatible LLM wrapper (Google AI Studio/Ollama)
    ├── query.py                        ← Interactive CLI to query legal articles
    ├── test_connection.py              ← Utility to verify LLM endpoint connectivity
    └── utils.py                        ← Shared JSON extraction, text & whitespace utilities
```

---

## 🚀 Quick Start

### 1. Install Dependencies

Ensure you are using Python 3.10+ and install the required packages from the project root:

```bash
pip install -r requirements.txt
```

> **Vietnamese NLP Note**: For optimal lexical segmentation in BM25 indexing, installing `underthesea` is recommended. The embedding pipeline uses `BAAI/bge-m3` by default, which excels at Vietnamese semantic representation.

### 2. Environment Configuration

Copy the example environment file and configure your LLM endpoint and competition API credentials:

```bash
cp .env.example .env
```

Edit `.env` to specify your API credentials and preferences:
```ini
LLM_API_KEY="your-llm-api-key"
LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL="gemini-2.5-flash"
ALQAC_TOKEN="your-competition-api-token"
NUM_RESULTS=5
DEFAULT_ALPHA=0.5
```

### 3. Build Search Indexes (Run Once)

Execute the index builder from the **root directory** using Python's module flag (`-m`):

```bash
python -m rag_runner.build_index
```

This command sequentially performs:
1. Loads and chunks `data/corpus/corpus_law_pub.json`.
2. Computes embeddings and caches them to `data/output/law_embeddings.parquet`.
3. Builds the FAISS vector similarity index (`data/output/law.faiss`).
4. Builds the BM25 lexical keyword index (`data/output/law_bm25.pkl`).

*Note: If underlying Parquet data changes, re-running automatically synchronizes and rebuilds the indexes. Use `--force` to trigger a manual rebuild.*

### 4. Verify LLM Connection

Test your `.env` credentials and endpoint connectivity:

```bash
python -m rag_retrieval.test_connection
```

### 5. Interactive Legal Querying

Ask individual legal questions or start an interactive chat session:

```bash
# Ask a single legal question
python -m rag_retrieval.query "Điều kiện để thành lập tổ chức tín dụng là gì?"

# Customizing retrieved results (k) and semantic weight (alpha: 1.0 = FAISS only, 0.0 = BM25 only)
python -m rag_retrieval.query "Quy định về hợp đồng lao động" --k 10 --alpha 0.7

# Start interactive loop
python -m rag_retrieval.query
```

### 6. Pre-fetch Case Evidence (Pro-Tip #1)

To avoid hitting API rate limits (1 request / 5s) or experiencing timeouts during LLM inference, pre-fetch and cache all case evidence chunks from the competition API ahead of time:

```bash
python -m rag_retrieval.prefetch_cache --test-file data/test/ALQAC2026_public_test.json
```

This script runs asynchronously, complies strictly with the 5.2s rate-limit delay (`RETRY_DELAY_NORMAL`), and saves rich evidence chunks to `data/cache/case_evidence_cache.json`. Use `--force` to re-query the API and refresh existing cache entries.

### 7. Generate ALQAC 2026 Submission

Run the automated RAG prediction pipeline across benchmark test cases:

```bash
python -m rag_retrieval.generate_submission --test-file data/test/ALQAC2026_public_test.json --output submission.json
```

The script processes cases incrementally, reading pre-fetched evidence from disk cache at zero latency and saving progress after each case so execution can be safely resumed without duplicate work.

---

## 🏆 ALQAC 2026 Competition Metric Optimizations

The pipeline is architected specifically around the competition's evaluation formula:
$$\text{FinalScore} = 0.70 \times \text{OutcomeAccuracy} + 0.20 \times \text{PenalizedCaseRecall} + 0.10 \times \text{LawF1micro}$$

1. **Maximized Outcome Accuracy (70% Weight)**:
   - **Few-Shot Classification Prompt**: The LLM system prompt includes structured Vietnamese legal reasoning examples guiding classification across `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, and `B_WIN`.
   - **Deterministic Rule Override (`rule_override`)**: Court decisions containing unambiguous verdict phrasing (e.g., *"chấp nhận một phần yêu cầu khởi kiện"*, *"không có căn cứ để chấp nhận"*) bypass LLM inference entirely, guaranteeing 100% precision on explicit rulings.
2. **Maximized Penalized Case Recall (20% Weight) & API Efficiency**:
   - **Multi-Query Dispute Extraction**: The pipeline analyzes `case_fact` to extract specific plaintiff claims (*"nguyên đơn trình bày"*), defendant counterclaims (*"bị đơn trình bày"*), and court reasoning (*"tòa án nhận định"*), generating up to 4 targeted search queries per case.
   - **Zero Penalty Guarantee**: By capping queries at $\le 4$ per case, total API calls remain within the safe threshold ($c_i \le 2n_i$), ensuring the API efficiency factor $E_i = 1.0$ (no penalty).
   - **Rate Limit Compliance**: Enforces a strict `5.2s` interval (`RETRY_DELAY_NORMAL`) between API requests and exponential backoff (`6.0s+` for `429 Too Many Requests`) to prevent server blocks.
3. **Maximized Micro Law F1 (10% Weight)**:
   - **Extended Fact Context Window**: Increases `MAX_FACT_LEN_FOR_SEARCH` to `2500` characters, ensuring the Hybrid Retriever evaluates the full legal substance of disputes rather than superficial administrative headers.
   - **Reciprocal Rank Fusion (RRF)**: Combines dense vector similarity (`BAAI/bge-m3`) with lexical keyword matching (`BM25Okapi`) to retrieve precise Civil and Commercial Code article IDs (`law_id`, `aid`).

---

## 📦 Core Packages & Technologies

| Package | Purpose in Pipeline |
| :--- | :--- |
| `torch` & `sentence-transformers` | Computes dense vector representations using local embedding models (`BAAI/bge-m3`). |
| `faiss-cpu` / `faiss-gpu` | High-speed inner-product (cosine similarity) vector indexing and search. |
| `rank_bm25` | Lexical keyword indexing and Okapi BM25 score calculation. |
| `pyarrow` & `pandas` | Type-safe, high-throughput binary caching of embeddings and metadata via Parquet. |
| `openai` | Standardized REST client wrapper communicating with Google AI Studio, OpenAI, or local LM Studio endpoints. |
| `python-dotenv` | Transparently loads system environment variables and secrets from `.env`. |

---

## ⚙️ Customization & Tuning

All key parameters can be adjusted directly in **`configs/config.py`** or overridden via **`.env`**:

- **Hybrid Weighting (`DEFAULT_ALPHA`)**: Controls the balance between semantic FAISS search and lexical BM25 search during Reciprocal Rank Fusion (`0.5` = balanced, `0.7` = favor semantics, `0.3` = favor exact keywords).
- **Chunking Strategy (`MAX_CHUNK_TOKENS`, `CHUNK_STRIDE`)**: Configures window size and overlap for splitting long legal statutes into context-preserving segments.
- **Verdict Keyword Boosting (`VERDICT_KEYWORDS`, `VERDICT_KEYWORD_BOOST`)**: Fine-tunes BM25 score multipliers for evidence chunks containing critical judgement headers (`"tuyên xử"`, `"quyết định"`, `"án phí"`).
- **Rule Overrides (`RULE_KEYWORDS_...`)**: Tailors deterministic keyword rules and negation filters (`"không "`, `"chưa "`, `"để "`, `"cứ "`) for court decision classification.
- **Rate Limit Delays (`RETRY_DELAY_NORMAL`, `RETRY_DELAY_429`)**: Adjusts sleep intervals between API requests to comply with competition server limits.
