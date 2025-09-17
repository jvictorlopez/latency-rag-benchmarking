# RAG PDF QA – Weaviate

## Resumo Executivo

- **Upload e indexação de PDFs**: extração de texto (pypdf), chunking consciente de tokens, geração de embeddings via serviço local Flask (FlagEmbedding) e inserção no Weaviate.
- **Perguntas e respostas com RAG**: API FastAPI expõe `/documents` e `/question`; respostas incluem referências e (quando disponíveis) contextos.
- **Cinco estratégias de busca**: `semantic`, `semantic_rerank`, `bm25`, `hybrid` e `no_rag` (baseline sem recuperação). Parâmetros: `top_k`, `alpha` (para híbrido) e `rerank_property`.
- **Tela “⚡ Latency Benchmark” (UI)**: dispara as 5 estratégias em paralelo (asyncio + httpx.AsyncClient) e exibe “Latency: X ms” por modo em cartões lado a lado.
- **Arquitetura**: FastAPI (API) + Weaviate (vetores, BM25, híbrido) + serviço local de embeddings/reranker (Flask + FlagEmbedding) + Streamlit (UI).
- **Execução via Docker Compose**: sobe `weaviate` (8080/50051), `local-inference` (5001), `api` (8000) e `ui` (8501).
- **LLM para resposta final**: OpenAI Chat Completions (`OPENAI_API_BASE`/`OPENAI_API_KEY`); `no_rag` permite comparar a resposta sem recuperação.

## Matriz “Requisito do Desafio vs Entrega”

| Requisito | Como foi atendido | Endpoint/Tela | Evidência |
|---|---|---|---|
| Upload e indexação (PDF → chunks → embeddings → armazenamento) | Extração com pypdf, chunking token-aware, embeddings no serviço local e inserção no Weaviate | `POST /documents` | `api/src/main.py::upload_documents`, `api/src/rag/ingest.py::embed_texts`, `api/src/rag/weav_client.py::ensure_schema`, `docker-compose.yml` |
| Consulta e resposta com referências | Recupera contextos (quando o modo usa vetores/BM25), monta prompt e chama LLM; retorna `answer`, `references`, `contexts` | `POST /question` | `api/src/main.py::ask`, `api/src/rag/prompts.py::build_prompt`, `api/src/rag/llm.py::chat` |
| Estratégias de busca (Semantic, Semantic+Rerank, BM25, Hybrid, NoRAG) | Implementadas no módulo de retrievers; `no_rag` responde sem recuperação | Tela QA; `POST /question` | `api/src/rag/retrievers.py::semantic`, `::semantic_with_rerank`, `::bm25`, `::hybrid`, `::to_props` |
| Frontend funcional (QA e Latência) | Tela QA e tela ⚡ Latency Benchmark com 5 cartões e latência por modo | Streamlit | `ui/app.py` (`view_qa`, `view_benchmark`, asyncio/httpx) |
| Extras: latência, Docker, logs | Benchmark paralelo; Compose para todos os serviços; logs por serviço | – | `ui/app.py` (medição com `time.perf_counter()`), `Makefile`, `docker compose logs` |

## Arquitetura & Fluxo

- Upload: `POST /documents` → salva PDF temporário → `extract_pdf_text` (pypdf) → `build_chunks` (token-aware) → `embed_texts` (HTTP `POST /vectors` no serviço `local-inference`) → `col.data.insert` no Weaviate.
- Perguntas: `POST /question` (um modo de cada vez na tela QA) ou 5 chamadas paralelas na tela “⚡ Latency Benchmark” → recuperação (`semantic`/`semantic_rerank`/`bm25`/`hybrid`) → `build_prompt` → `chat` (OpenAI) → resposta com referências.
- Weaviate: classe `DocChunk` (vectorizer `none`) e propriedades de metadados; BM25/híbrido habilitados no servidor (módulos `text2vec-transformers` e `reranker-transformers` apontando para `local-inference`).
- Serviço de embeddings/reranker: Flask (`inference/app.py`) expõe `/.well-known/ready`, `/meta`, `/vectors` e `/rerank` (porta 5001) usando FlagEmbedding (BAAI).
- Robustez com cliente Weaviate: `_call_near_vector` adapta diferenças de assinatura (`near_vector` vs `vector`) para compatibilidade entre versões do cliente.

## Endpoints

| Endpoint | Propósito | Payload | Exemplo (cURL) |
|---|---|---|---|
| `GET /.well-known/ready` | Readiness simples | – | `curl -s http://localhost:8000/.well-known/ready` |
| `GET /meta` | Metadados de configuração | – | `curl -s http://localhost:8000/meta` |
| `POST /documents` | Upload/ingestão de PDFs | multipart `files[]` | ```bash
curl -s -F "files=@docs/produto_2.pdf;type=application/pdf" \
     -F "files=@docs/1756-in043_-en-p.pdf;type=application/pdf" \
     http://localhost:8000/documents
``` |
| `POST /question` | Pergunta + modo de recuperação | JSON `{question, mode, top_k, alpha, rerank_property}` | ```bash
curl -s -X POST http://localhost:8000/question \
  -H 'Content-Type: application/json' \
  -d '{"question":"motor power rating","mode":"hybrid","top_k":5,"alpha":0.5,"rerank_property":"chunk"}'
``` |

- Resposta típica: `{ "answer": str, "references": [str], "contexts": [{title,page,chunk,...}] }`.
- Benchmark: a UI dispara 5 requisições ao mesmo `POST /question` (modos fixos) em paralelo e mede a latência por modo; não há endpoint extra.

## Frontend (Streamlit)

- **Tela QA**: inputs — pergunta, `top_k`, `alpha`, `rerank_property`, seletor de `mode` e botão “Get Answer”.
- **Tela ⚡ Latency Benchmark**: navegação no topo e na sidebar; controles em uma única linha (sem seletor de modo); ao clicar “Get Answers”, roda as 5 estratégias em paralelo com `asyncio.gather` + `httpx.AsyncClient` e renderiza 5 colunas (título do modo, `Latency: X ms`, “Answer”, “References”).
- Configuração: `st.set_page_config(page_title="RAG PDF QA – Weaviate", layout="wide")`.

## Como Rodar

### Docker (recomendado)

```bash
# Subir todos os serviços
docker compose up -d --build
# ou
make up

# Endereços
# UI:  http://localhost:8501
# API: http://localhost:8000 (docs em /docs)
# Weaviate: http://localhost:8080
```

### Local (venv)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r api/requirements.txt
pip install -r ui/requirements.txt

# Variáveis mínimas
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"

# API (rode dentro do diretório api/)
cd api
uvicorn src.main:app --host 0.0.0.0 --port 8000

# UI (em outro terminal na raiz do repo)
streamlit run ui/app.py
```

### Scripts úteis

```bash
# Diagnóstico da assinatura near_vector (compatibilidade de cliente)
python scripts/diag_semantic_sig.py

# Ingestão end-to-end fora da UI (aguarda serviços, extrai, embeda e insere)
python scripts/index_debug.py --pdf docs/produto_2.pdf --limit 50
```

## Variáveis de Ambiente

- **API (FastAPI)**
  - `WEAVIATE_HTTP_HOST` (default: `weaviate` no Docker)
  - `WEAVIATE_HTTP_PORT` (default: `8080`)
  - `WEAVIATE_GRPC_HOST` (default: `weaviate`)
  - `WEAVIATE_GRPC_PORT` (default: `50051`)
  - `OPENAI_API_BASE` (default: `https://api.openai.com/v1`)
  - `OPENAI_API_KEY` (obrigatória para respostas com LLM)
  - `LLM_MODEL` (default: `gpt-4o-mini`)
  - `CHUNK_TOKENS` (default: `450`), `CHUNK_OVERLAP` (default: `60`)
  - `INFER_BASE` (default: `http://local-inference:5001`) — base do serviço de embeddings
- **Serviço de embeddings (Flask)**
  - `EMBEDDING_MODEL` (default: `BAAI/bge-small-en-v1.5`)
  - `RERANK_MODEL` (default: `BAAI/bge-reranker-base`)
- **UI (Streamlit)**
  - `API_BASE_URL` (default: `http://localhost:8000` fora de Docker; no Compose: `http://api:8000`)

> Nunca comite segredos. Use placeholders (ex.: `OPENAI_API_KEY="sk-..."`).

## Testes & Smoke

```bash
# Testes (dentro dos contêineres)
make test

# Smoke manual
curl -s http://localhost:8000/.well-known/ready
curl -s -F "files=@docs/produto_2.pdf;type=application/pdf" http://localhost:8000/documents
curl -s -X POST http://localhost:8000/question \
  -H 'Content-Type: application/json' \
  -d '{"question":"motor power rating","mode":"bm25","top_k":5,"alpha":0.5,"rerank_property":"chunk"}'
```

## Observabilidade & Latência

- **Logs**: `docker compose logs -f api`, `docker compose logs -f ui`, `docker compose logs -f weaviate`, `docker compose logs -f local-inference`.
- **Medição de latência por modo (UI)**: calculada com `time.perf_counter()` em cada requisição paralela; exibida como “Latency: X ms” em cada cartão da tela de benchmark.

## Limitações & Próximos Passos

- Suporte a provedores alternativos de LLM/embeddings; cache de vetores; avaliação automática de qualidade.
- Autenticação/controle de acesso; tracing distribuído; dashboards de métricas.
- Enriquecimento de contexto (citations com trechos destacados) e monitoramento de custo/latência fim-a-fim.

## Decisões de Design

- **Cinco estratégias lado a lado**: comparação objetiva de qualidade/latência e alinhamento aos requisitos.
- **Weaviate**: BM25, híbrido e API de vetores maduras; schema `vectorizer: none` para usar embeddings locais de forma explícita.
- **Serviço local de embeddings**: isolamento operacional, controle de modelos (`/vectors`, `/rerank`) e previsibilidade de latência.
- **Compatibilidade do cliente**: `_call_near_vector` abstrai diferenças de assinatura entre versões do cliente Weaviate.
- **Baseline `no_rag`**: controle experimental para comparar com respostas puras de LLM.

## Licença

MIT.

---

### Screenshots

- UI – QA e Benchmark: ver `docs/screenshot.png`.


