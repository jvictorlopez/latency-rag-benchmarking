import os, httpx, logging
from httpx import HTTPStatusError

# Normalize env
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
_ALLOW = {"gpt-4o-mini", "gpt-4o"}
if LLM_MODEL not in _ALLOW:
    logging.getLogger("uvicorn.error").warning("LLM_MODEL '%s' not in %s; falling back to gpt-4o-mini", LLM_MODEL, sorted(_ALLOW))
    LLM_MODEL = "gpt-4o-mini"

def chat(question: str, prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    with httpx.Client(timeout=120) as c:
        r = c.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            logging.getLogger("uvicorn.error").error("OpenAI error %s: %s", r.status_code, r.text)
            r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


