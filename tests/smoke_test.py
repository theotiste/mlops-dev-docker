import os, requests

API_URL = os.getenv("API_URL", "http://localhost:5001")

def test_health():
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json().get("ok") is True









