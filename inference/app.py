from flask import Flask, request, jsonify
import json
import os
from FlagEmbedding import FlagModel, FlagReranker
import numpy as np

# Read model names from env to match notebook settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Init models (FlagEmbedding downloads to default cache)
emb_model = FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)

app = Flask(__name__)

@app.get("/.well-known/ready")
def ready():
    return "Ready", 200

@app.get("/meta")
def meta():
    return jsonify({"status": "Ready", "embedding_model": EMBEDDING_MODEL_NAME, "reranker": RERANK_MODEL_NAME}), 200

@app.post("/vectors")
def vectors():
    try:
        body = request.json
        if isinstance(body, dict) and "text" in body:
            texts = body["text"]
        else:
            texts = json.loads(request.data.decode("utf-8"))
            if isinstance(texts, dict) and "text" in texts:
                texts = texts["text"]
        if isinstance(texts, str):
            texts = [texts]

        # FlagEmbedding's encode does not accept normalize_embeddings. Normalize manually.
        vecs = emb_model.encode(texts, batch_size=32)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        return jsonify({"vector": vecs.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/rerank")
def rerank_route():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            payload = json.loads(request.data.decode("utf-8"))
        if not isinstance(payload, dict) or "query" not in payload or "documents" not in payload:
            return jsonify({"error": "Expected {'query': str, 'documents': [str,...]}"}), 400
        query = payload["query"]
        docs = payload["documents"] or []
        if not docs:
            return jsonify({"scores": []}), 200

        pairs = [(query, d) for d in docs]
        scores = reranker.compute_score(pairs)
        out = [{"document": docs[i], "score": float(scores[i])} for i in range(len(docs))]
        return jsonify({"scores": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)


