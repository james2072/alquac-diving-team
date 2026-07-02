# Law RAG

A **Retrieval-Augmented Generation** pipeline for querying Vietnamese legal texts,
adapted from the [simple-local-rag notebook](00_simple_local_rag.ipynb).

Instead of a PDF, this project reads from `corpus_law_pub.json` where each entry
follows the schema:

```json
[
  {
    "id": 33,
    "law_id": "66/2014/QH13",
    "content": [
      {"aid": 819, "content_Article": "...article text..."},
      ...
    ]
  }
]
```

---

## Project structure

```
ALQUAC/                         ← root directory
├── .env.example                ← copy to .env and fill in your API keys
├── .gitignore
├── requirements.txt            ← pip requirements
├── README.md                   ← this documentation
├── corpus_law_pub.json         ← the law corpus JSON
├── rag_runner/                 ← Offline embedding logic
│   ├── config.py               ← reads all settings from root .env
│   ├── corpus_loader.py        ← loads & cleans the JSON corpus
│   ├── embedder.py             ← embeds articles, caches Parquet, retrieves top-k
│   └── build_index.py          ← builds the cached embedding index
└── rag_retrieval/              ← Online retrieval & Generation logic
    ├── llm_client.py           ← OpenAI-compatible completions API client
    ├── rag.py                  ← RAG pipeline (retrieve → prompt → generate)
    ├── test_connection.py      ← connection playground to test Gemini
    └── query.py                ← interactive CLI to run questions
```

---

## Quick start

### 1. Install dependencies

```bash
# In the root directory (ALQUAC)
pip install -r requirements.txt
```

> **Vietnamese text note**: the default embedding model is `BAAI/bge-m3` which is highly recommended for Vietnamese text. You can customize the model and device inside `.env`.

### 2. Configure

```bash
cp .env.example .env
# Edit .env: set your API key (e.g. Google AI Studio key), base URL, and model name.
```

### 3. Build the embedding index (run once)

```bash
cd rag_runner
python build_index.py
```

This reads `../corpus_law_pub.json`, embeds every article, and saves the binary representation to `../data/law_embeddings.parquet`. Re-running skips the rebuild unless you pass the `--force` flag.

### 4. Query

From inside the `rag_retrieval` directory:

```bash
cd ../rag_retrieval

# Optional: test connection to Google AI Studio first
python test_connection.py

# Ask a single question
python query.py "Điều kiện để thành lập tổ chức tín dụng là gì?"

# Or start the interactive chat loop
python query.py
```

---

## Required packages — explained

| Package | Why |
|---------|-----|
| `torch` | Tensor operations for embedding search (dot-product) |
| `sentence-transformers` | Runs the embedding model locally |
| `pyarrow` | Memory-efficient Parquet read/write |
| `pandas` / `numpy` | DataFrames for index caching |
| `python-dotenv` | Loads `.env` configuration from root |
| `openai` | Chat API client for Google AI Studio / custom LLM endpoints |

---

## Customisation tips

- **Chunking strategy**: currently each `content_Article` is one chunk.
  If articles are very long you can split them further in `corpus_loader.py`.
- **Number of retrieved articles**: set `NUM_RESULTS` in `.env` (default 5).
- **Min chunk tokens**: set `CHUNK_MIN_TOKENS` (default 30) to filter noise.
- **Prompt language**: edit `SYSTEM_PROMPT` and `PROMPT_TEMPLATE` in `rag.py`.

