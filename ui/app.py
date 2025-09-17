import os
import asyncio
import time
from typing import Any, Dict, List, Tuple

import httpx
import streamlit as st

# ----- Page setup -----
st.set_page_config(page_title="RAG PDF QA – Weaviate", layout="wide")

# Keep existing env wiring as-is
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ----- Sidebar: keep existing upload/index flow -----
with st.sidebar:
    st.header("Upload PDFs")
    # Preserve original uploader + indexing logic exactly
    up_files = st.file_uploader("Select one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if up_files and st.button("Index documents"):
        with st.spinner("Uploading & indexing..."):
            try:
                files = [("files", (f.name, f.read(), "application/pdf")) for f in up_files]
                r = httpx.post(f"{API_BASE}/documents", files=files, timeout=300)
                if r.status_code == 200:
                    st.success(r.json())
                else:
                    st.error(f"{r.status_code} - {r.text}")
            except httpx.ConnectError:
                st.error("API is not reachable. Wait a few seconds and retry. "
                         "If you run UI outside Docker, set API_BASE_URL=http://localhost:8000")
            except Exception as e:
                st.error(str(e))

    st.divider()
    # View session state
    if "view" not in st.session_state:
        st.session_state["view"] = "qa"
    # Sidebar navigation affordance
    if st.session_state["view"] != "qa":
        if st.button("← Back to QA", use_container_width=True):
            st.session_state["view"] = "qa"
            st.rerun()
    else:
        if st.button("⚡ Latency Benchmark", type="primary", use_container_width=True):
            st.session_state["view"] = "bench"
            st.rerun()

# ----- Helpers -----
MODES: List[Tuple[str, str]] = [
    ("semantic", "Semantic Search"),
    ("semantic_rerank", "Semantic Reranking Search"),
    ("bm25", "BM25 Syntactic Search"),
    ("hybrid", "Hybrid Search"),
    ("no_rag", "No RAG"),
]

def extract_answer_and_refs(payload: Dict[str, Any]) -> Tuple[str, List[str]]:
    ans = payload.get("answer") or payload.get("result") or ""
    refs = payload.get("references") or payload.get("sources") or payload.get("docs") or []
    out: List[str] = []
    if isinstance(refs, list):
        for r in refs:
            if isinstance(r, dict):
                fname = r.get("filename") or r.get("source") or r.get("doc") or r.get("title") or str(r)
                page = r.get("page") or r.get("page_number") or r.get("p")
                out.append(f"{fname} (p.{page})" if page is not None else str(fname))
            else:
                out.append(str(r))
    else:
        out.append(str(refs))
    return ans, out

async def _call_one(client: httpx.AsyncClient, mode: str, question: str, top_k: int, alpha: float, rr_prop: str):
    url = f"{API_BASE}/question"
    payload = {
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "alpha": alpha,
        "rerank_property": rr_prop,
    }
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload, timeout=60)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    resp.raise_for_status()
    return {"mode": mode, "latency_ms": dt_ms, "data": resp.json()}

async def run_benchmark(question: str, top_k: int, alpha: float, rr_prop: str):
    async with httpx.AsyncClient() as client:
        tasks = [_call_one(client, mode, question, top_k, alpha, rr_prop) for mode, _ in MODES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    packed = []
    for (mode, label), res in zip(MODES, results):
        if isinstance(res, Exception):
            packed.append({"mode": mode, "label": label, "error": str(res)})
        else:
            packed.append({"mode": mode, "label": label, **res})
    return packed

def top_bar_nav():
    # Right-aligned CTA to switch views
    left, right = st.columns([5, 1])
    with right:
        if st.session_state.get("view", "qa") == "qa":
            if st.button("⚡ Latency Benchmark", key="nav_top_bench"):
                st.session_state["view"] = "bench"
                st.rerun()
        else:
            if st.button("← Back to QA", key="nav_top_qa"):
                st.session_state["view"] = "qa"
                st.rerun()

# ----- Views -----
def view_qa():
    st.title("📄 RAG PDF QA – Weaviate")
    top_bar_nav()
    st.subheader("Ask a question")
    # Preserve existing QA controls and logic exactly
    q = st.text_input("Your question")
    mode = st.selectbox("Mode", ["semantic","semantic_rerank","bm25","hybrid","no_rag"], index=3)
    top_k = st.slider("Top K", 1, 10, 5)
    alpha = st.slider("Alpha (hybrid)", 0.0, 1.0, 0.5)
    rerank_prop = st.selectbox("Rerank property", ["chunk","title"], index=0)

    if st.button("Get Answer"):
        payload = {"question": q, "mode": mode, "top_k": top_k, "alpha": alpha, "rerank_property": rerank_prop}
        with st.spinner("Thinking..."):
            r = httpx.post(f"{API_BASE}/question", json=payload, timeout=120)
            if r.status_code != 200:
                st.error(r.text)
            else:
                data = r.json()
                st.markdown("### Answer")
                st.write(data.get("answer", ""))
                if data.get("references"):
                    st.markdown("### References")
                    for i, ref in enumerate(data["references"], 1):
                        st.write(f"{i}. {ref}")
                if data.get("contexts"):
                    with st.expander("Show retrieved context"):
                        for c in data["contexts"]:
                            st.markdown(f"**{c.get('title','')} (p.{c.get('page','?')})**")
                            st.write(c.get("chunk", "")[:1500])
                            st.write("---")

def view_benchmark():
    st.title("⚡ Latency Benchmark")
    top_bar_nav()
    st.caption("Runs all 5 retrieval strategies in parallel and shows per-mode latency with answers and references.")

    # Horizontal control row
    q_col, k_col, a_col, rr_col, go_col = st.columns([4, 1, 1, 2, 1])
    with q_col:
        question = st.text_input("Your question", placeholder="Type a question to benchmark…")
    with k_col:
        top_k = st.slider("Top K", 1, 10, 5)
    with a_col:
        alpha = st.slider("Alpha (hybrid)", 0.0, 1.0, 0.50, step=0.01)
    with rr_col:
        rerank_property = st.selectbox("Rerank property", ["title", "text", "chunk", "content"], index=0)
    with go_col:
        run = st.button("Get Answers", type="primary", use_container_width=True)

    st.divider()

    if run:
        if not question.strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Running all strategies…"):
            results = asyncio.run(run_benchmark(question, top_k, alpha, rerank_property))

        cols = st.columns(5)
        for (mode, label), col in zip(MODES, cols):
            with col:
                box = st.container(border=True)
                with box:
                    st.subheader(label, anchor=False)
                    r = next((x for x in results if x["mode"] == mode), None)
                    if not r:
                        st.error("No result.")
                        continue
                    if "error" in r:
                        st.error(r["error"])
                        st.caption("Latency: —")
                        continue
                    latency_ms = r.get("latency_ms")
                    st.caption(f"Latency: {latency_ms} ms" if latency_ms is not None else "Latency: —")

                    data = r.get("data", {}) or {}
                    answer, refs = extract_answer_and_refs(data)
                    st.markdown("**Answer**")
                    st.write(answer or "—")
                    if refs:
                        st.markdown("**References**")
                        for i, ref in enumerate(refs, 1):
                            st.markdown(f"{i}. {ref}")

# ----- Router -----
if "view" not in st.session_state:
    st.session_state["view"] = "qa"

if st.session_state["view"] == "bench":
    view_benchmark()
else:
    view_qa()

