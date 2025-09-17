from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List, Optional
import os, tempfile, uuid, time

from .settings import CHUNK_TOKENS, CHUNK_OVERLAP
from .rag.weav_client import get_client, ensure_schema, get_collection
from weaviate.exceptions import WeaviateBaseError
from .rag.ingest import extract_pdf_text, build_chunks, embed_texts
from .rag.retrievers import semantic, semantic_with_rerank, bm25, hybrid, to_props
from .rag.prompts import build_prompt
from .rag.llm import chat
from .rag.types import QuestionRequest, AnswerResponse, DocRef
import logging, traceback

app = FastAPI(title="RAG PDF QA", version="1.0.0")

@app.get("/.well-known/ready")
def ready():
    return PlainTextResponse("Ready", 200)

@app.get("/meta")
def meta():
    return {"status": "Ready", "chunk_tokens": CHUNK_TOKENS, "overlap": CHUNK_OVERLAP}

@app.on_event("startup")
def init():
    # retry loop so the container doesn't crash before weaviate is ready
    for attempt in range(30):
        try:
            client = get_client()
            ensure_schema(client)
            client.close()
            return
        except Exception as e:
            try:
                if isinstance(e, WeaviateBaseError):
                    pass
            except Exception:
                pass
            time.sleep(2)
    # final attempt (raise if still failing)
    client = get_client()
    ensure_schema(client)
    client.close()

@app.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    import traceback, logging
    logger = logging.getLogger("uvicorn.error")
    try:
        client = get_client()
        col = get_collection(client)
        total_chunks = 0
        try:
            for f in files:
                if not f.filename.lower().endswith(".pdf"):
                    return JSONResponse(status_code=400, content={"error": f"Only PDF supported. Got {f.filename}"})
                blob = await f.read()
                import tempfile, os, uuid
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fp:
                    fp.write(blob)
                    path = fp.name

                pages = extract_pdf_text(path)
                doc_id = str(uuid.uuid4())
                items = build_chunks(doc_id, f.filename, f.filename, pages, CHUNK_TOKENS, CHUNK_OVERLAP)
                if not items:
                    continue
                vectors = embed_texts([x["chunk"] for x in items])
                for i, x in enumerate(items):
                    col.data.insert(properties=x, vector=vectors[i])
                total_chunks += len(items)
                os.remove(path)
            return {"message": "Documents processed successfully", "documents_indexed": len(files), "total_chunks": total_chunks}
        finally:
            client.close()
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})

@app.post("/question", response_model=AnswerResponse)
def ask(body: QuestionRequest):
    logger = logging.getLogger("uvicorn.error")
    client = get_client()
    col = get_collection(client)
    try:
        logger.info(f"/question mode={body.mode} top_k={body.top_k} alpha={body.alpha} rerank_prop={body.rerank_property}")
        if body.mode == "semantic":
            res = semantic(col, body.question, body.top_k)
        elif body.mode == "semantic_rerank":
            res = semantic_with_rerank(col, body.question, body.top_k, body.rerank_property)
        elif body.mode == "bm25":
            res = bm25(col, body.question, body.top_k)
        elif body.mode == "hybrid":
            res = hybrid(col, body.question, body.top_k, body.alpha)
        elif body.mode == "no_rag":
            res = None
        else:
            return JSONResponse(status_code=400, content={"error": "Unknown mode"})

        contexts = []
        if res is not None:
            props = to_props(res)
            contexts = props
        logger.info(f"retrieved_contexts={len(contexts)}")

        if body.mode == "no_rag" or not contexts:
            try:
                answer = chat(body.question, f"Answer the question: {body.question}")
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                return JSONResponse(status_code=500, content={"error": "LLM call failed", "traceback": tb})
            return AnswerResponse(answer=answer, references=[], contexts=[])

        prompt = build_prompt(body.question, contexts)
        try:
            answer = chat(body.question, prompt)
        except Exception:
            tb = traceback.format_exc()
            logger.error(tb)
            return JSONResponse(status_code=500, content={"error": "LLM call failed", "traceback": tb})
        refs = []
        ctx_objs = []
        for c in contexts:
            ref = f"{c.get('title','')} (p.{c.get('page','?')})"
            refs.append(ref)
            ctx_objs.append(DocRef(title=c.get("title"), page=c.get("page"), score=None, link=None, chunk=c.get("chunk")))
        return AnswerResponse(answer=answer, references=refs, contexts=ctx_objs)
    except Exception:
        tb = traceback.format_exc()
        logger.error(tb)
        return JSONResponse(status_code=500, content={"error": "/question failed", "traceback": tb})
    finally:
        client.close()


