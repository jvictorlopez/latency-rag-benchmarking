## RAG PDF QA (Weaviate + Streamlit)

Production-ready, dockerized RAG system implementing the "Machine Learning Engineering – LLM" challenge:

- Upload PDFs → extract → token-aware chunk → embed via local Flask inference → store in Weaviate
- Ask questions with 5 retrieval modes: semantic, semantic_rerank, bm25, hybrid, no_rag
- Clean REST API with FastAPI, optional Streamlit UI, and Weaviate backed by FlagEmbedding models

### Quickstart

```bash
cp .env.example .env   # provide OPENAI_API_KEY (or USE_OLLAMA=true)
make up                 # builds and starts Weaviate, local-inference, API, UI
# UI:  http://localhost:8501
# API: http://localhost:8000 (docs at /docs)
```

### Pull the LLM once (inside Docker)

```bash
docker compose run --rm ollama ollama pull llama3.2:3b-instruct
```

### Services

- `weaviate`: Vector DB with modules `text2vec-transformers` and `reranker-transformers`, pointing to the local inference service
- `local-inference`: Flask server exposing `/vectors` and `/rerank` using FlagEmbedding
- `api`: FastAPI orchestrating ingestion and QA
- `ui`: Streamlit app for upload and Q&A

### Environment

Copy `.env.example` to `.env` and adjust:

```
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
USE_OLLAMA=false
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_TOKENS=450
CHUNK_OVERLAP=60
```

### Endpoints

`POST /documents` (multipart/form-data)
- Field: `files` (one or more PDFs)
- Response: `{ "message": "Documents processed successfully", "documents_indexed": 2, "total_chunks": 128 }`

`POST /question` (application/json)

```json
{
  "question": "What is the power consumption of the motor?",
  "mode": "hybrid",               
  "top_k": 5,
  "alpha": 0.5,
  "rerank_property": "chunk"
}
```

Response:

```json
{
  "answer": "The motor's power consumption is 2.3 kW.",
  "references": ["<title> (p.7)", "..."],
  "contexts": [{ "title": "...", "page": 7, "chunk": "...", "score": null }]
}
```

### Retrieval Modes

- **semantic**: Weaviate `near_vector` using local BGE embeddings
- **semantic_rerank**: `near_vector` + `Rerank(query, prop='chunk')` (local BGE reranker)
- **bm25**: sparse keyword search
- **hybrid**: Weaviate hybrid (RRF) combining vector and BM25 with `alpha`
- **no_rag**: LLM only

### Models

- Embeddings: `BAAI/bge-small-en-v1.5` via local-inference (FlagEmbedding)
- Reranker: `BAAI/bge-reranker-base` via local-inference
- LLM: Ollama `llama3.2:3b-instruct` (default)

### Developer

```bash
make logs
make test
make down
```

Notes:
- Chunking is token-aware via `tiktoken` (defaults 450 tokens, 60 overlap)
- If context is insufficient, the model responds briefly and states limitation
- Auto docs at `http://localhost:8000/docs`


