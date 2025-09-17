import hashlib, datetime
from typing import List, Dict
import tiktoken

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def tokenize_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]
        chunks.append(enc.decode(window))
        if i + max_tokens >= len(toks): break
        i += max_tokens - overlap
    return chunks


