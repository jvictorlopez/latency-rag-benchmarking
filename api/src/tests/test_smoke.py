from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_meta():
    r = client.get("/meta")
    assert r.status_code == 200


