## System Architecture

- **UI (Streamlit)** → calls API.
- **API (FastAPI)** → orchestrates ingestion & QA:
  - `/documents`: extract → chunk (token-aware) → embed (local-inference) → upsert to Weaviate.
  - `/question`: retrieval per mode (semantic / semantic+rerank / BM25 / hybrid / no_rag) → prompt → LLM.
- **Weaviate** with `text2vec-transformers` & `reranker-transformers` backed by **Local Inference** Flask service.
- **Local Inference** (Flask): `POST /vectors` & `POST /rerank` powered by FlagEmbedding (BAAI bge models).

Data model: `DocChunk{ doc_id, source, title, page, chunk_index, chunk, created_at, mime, hash, num_tokens }`.


