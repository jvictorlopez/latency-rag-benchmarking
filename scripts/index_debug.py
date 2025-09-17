#!/usr/bin/env python3
"""
Standalone ingest debugger for PDF -> chunks -> embeddings -> Weaviate insert.

Usage:
  python scripts/index_debug.py --pdf docs/produto_2.pdf --limit 999
"""

import argparse, os, sys, time, uuid, traceback
import httpx
from dotenv import load_dotenv

# Local repo imports (reuse the same code paths the API uses)
sys.path.append(os.path.abspath("api"))
from src.rag.weav_client import get_client, ensure_schema, get_collection
from src.rag.ingest import extract_pdf_text, build_chunks, embed_texts
from src.rag.utils import now_iso


def wait_ready(url: str, name: str, tries=60, sleep=2):
    for i in range(tries):
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                print(f"[ok] {name} ready: {url}")
                return
        except Exception:
            pass
        time.sleep(sleep)
    raise RuntimeError(f"[fail] {name} not ready: {url}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--limit", type=int, default=999, help="limit chunks to insert")
    args = ap.parse_args()

    # 0) Sanity
    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    print(f"[info] Using PDF: {args.pdf}")

    # 1) Wait for dependencies
    wait_ready("http://localhost:5001/.well-known/ready", "local-inference")
    wait_ready("http://localhost:8080/v1/.well-known/ready", "weaviate")

    # 2) Ensure schema
    client = get_client()
    try:
        ensure_schema(client)
        col = get_collection(client)
        print("[ok] Weaviate schema ensured")
    finally:
        client.close()

    # 3) Extract PDF text
    pages = extract_pdf_text(args.pdf)
    total_chars = sum(len(p.get("text") or "") for p in pages)
    print(f"[info] pages={len(pages)} total_chars={total_chars}")
    if total_chars == 0:
        raise RuntimeError("PDF extraction returned 0 characters. Check PDF or poppler utils.")

    # 4) Chunk
    CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "450"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
    fname = os.path.basename(args.pdf)
    doc_id = str(uuid.uuid4())
    items = build_chunks(doc_id, fname, fname, pages, CHUNK_TOKENS, CHUNK_OVERLAP)
    if not items:
        raise RuntimeError("No chunks produced from PDF.")
    print(f"[info] chunks_built={len(items)} (first chunk chars={len(items[0]['chunk'])})")

    # 5) Embed (via local-inference /vectors)
    texts = [x["chunk"] for x in items[:args.limit]]
    try:
        vecs = embed_texts(texts)
    except Exception:
        print("[error] Embedding failed. Full traceback:")
        traceback.print_exc()
        print(f"[debug] Example payload text len={len(texts[0])} chars")
        sys.exit(2)
    print(f"[ok] embeddings_received={len(vecs)} dim={len(vecs[0]) if vecs else 'n/a'}")

    # 6) Insert back into Weaviate
    client = get_client()
    try:
        col = get_collection(client)
        inserted = 0
        for i, x in enumerate(items[:args.limit]):
            try:
                col.data.insert(properties=x, vector=vecs[i])
                inserted += 1
            except Exception:
                print(f"[error] Insert failed at chunk {i} (page={x['page']}, idx={x['chunk_index']}). Traceback:")
                traceback.print_exc()
                print(f"[debug] chunk_chars={len(x['chunk'])}")
                sys.exit(3)
        print(f"[ok] inserted={inserted}/{min(args.limit,len(items))}")
    finally:
        client.close()

    print("[SUCCESS] Indexing path (PDF -> chunks -> embed -> insert) is working.")


if __name__ == "__main__":
    main()


