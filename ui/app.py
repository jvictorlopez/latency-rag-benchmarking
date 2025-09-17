import streamlit as st
import httpx
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG PDF QA", layout="centered")
st.title("📄 RAG PDF QA – Weaviate")

with st.sidebar:
    st.header("Upload PDFs")
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

st.subheader("Ask a question")
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
            st.write(data["answer"])
            if data.get("references"):
                st.markdown("### References")
                for i, ref in enumerate(data["references"], 1):
                    st.write(f"{i}. {ref}")
            if data.get("contexts"):
                with st.expander("Show retrieved context"):
                    for c in data["contexts"]:
                        st.markdown(f"**{c.get('title','')} (p.{c.get('page','?')})**")
                        st.write(c.get("chunk","")[:1500])
                        st.write("---")


