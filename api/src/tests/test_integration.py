from fastapi.testclient import TestClient
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import json

from src.main import app


def _make_pdf_bytes(text: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, text)
    c.save()
    buf.seek(0)
    return buf.read()


class FakeCollection:
    def __init__(self):
        self._items = []

    class _Data:
        def __init__(self, outer):
            self.outer = outer
        def insert(self, properties, vectors=None):
            self.outer._items.append(properties)

    class _Query:
        def __init__(self, outer):
            self.outer = outer
        class _Res:
            def __init__(self, objs):
                class _Obj:
                    def __init__(self, props):
                        self.properties = props
                self.objects = [_Obj(p) for p in objs]
        def bm25(self, query: str, limit: int):
            # naive: return first N
            return FakeCollection._Query._Res(self.outer._items[:limit])
        def near_text(self, query: str, limit: int, rerank=None):
            return FakeCollection._Query._Res(self.outer._items[:limit])
        def hybrid(self, query: str, limit: int, alpha: float):
            return FakeCollection._Query._Res(self.outer._items[:limit])

    @property
    def data(self):
        return FakeCollection._Data(self)

    @property
    def query(self):
        return FakeCollection._Query(self)


def test_documents_and_question_bm25(monkeypatch):
    # Speed up test by mocking embeddings and LLM
    monkeypatch.setattr("src.rag.ingest.embed_texts", lambda texts: [[0.0] * 384 for _ in texts])
    monkeypatch.setattr("src.rag.llm.chat", lambda q, p: "ok")
    # Stub weaviate client usage inside app
    fake_collection = FakeCollection()
    monkeypatch.setattr("src.rag.weav_client.get_client", lambda: object())
    monkeypatch.setattr("src.rag.weav_client.ensure_schema", lambda client: None)
    monkeypatch.setattr("src.rag.weav_client.get_collection", lambda client: fake_collection)

    client = TestClient(app)

    # Upload a small 1-page PDF
    pdf_bytes = _make_pdf_bytes("The flux capacitor enables time travel in this test document.")
    files = {"files": ("test.pdf", pdf_bytes, "application/pdf")}
    r = client.post("/documents", files=files)
    assert r.status_code == 200

    # Ask a BM25 question (no vectorization required)
    payload = {
        "question": "What enables time travel?",
        "mode": "bm25",
        "top_k": 3,
        "alpha": 0.5,
        "rerank_property": "chunk",
    }
    r2 = client.post("/question", content=json.dumps(payload))
    assert r2.status_code == 200
    data = r2.json()
    assert "answer" in data
    assert isinstance(data.get("contexts", []), list)


