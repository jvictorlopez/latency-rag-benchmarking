from src.rag.utils import chunk_text

def test_chunker_basic():
    text = "hello " * 1000
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 5


