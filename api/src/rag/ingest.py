from typing import List, Dict, Any
from pypdf import PdfReader
from .utils import chunk_text, tokenize_len, now_iso, sha1_bytes
import httpx, os

# Allow overriding the inference base from the host (e.g., http://localhost:5001)
INFER_BASE = os.getenv("INFER_BASE", "http://local-inference:5001")

def extract_pdf_text(file_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append({"page": i+1, "text": txt})
    return pages

def embed_texts(texts: List[str]) -> List[List[float]]:
    with httpx.Client(timeout=60) as c:
        resp = c.post(f"{INFER_BASE}/vectors", json={"text": texts})
        resp.raise_for_status()
    return resp.json()["vector"]

def build_chunks(doc_id: str, source: str, title: str, pages, max_tokens: int, overlap: int):
    items = []
    for p in pages:
        segments = chunk_text(p["text"], max_tokens, overlap) if p["text"] else []
        for j, seg in enumerate(segments):
            items.append({
                "doc_id": doc_id,
                "source": source,
                "title": title or source,
                "page": p["page"],
                "chunk_index": j,
                "chunk": seg,
                "created_at": now_iso(),
                "mime": "application/pdf",
                "hash": sha1_bytes(seg.encode("utf-8")),
                "num_tokens": tokenize_len(seg),
            })
    return items


