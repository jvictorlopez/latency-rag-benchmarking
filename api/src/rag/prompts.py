SYSTEM = (
  "You are a helpful assistant. Use ONLY the provided context to answer. "
  "If the context is insufficient, say you don't know. Be concise and cite references."
)

def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts):
        title = c.get("title", "")
        page = c.get("page", "")
        chunk = c.get("chunk", "")
        ctx_lines.append(f"[{i+1}] Title: {title} (p.{page})\n{chunk}")
    ctx_blob = "\n\n".join(ctx_lines)
    return f"{SYSTEM}\n\nQuestion: {question}\n\nContext:\n{ctx_blob}\n\nAnswer:"


