# tests/smoke_test.py
import os
import time
import requests

API_URL = os.getenv("API_URL", "http://localhost:5001").rstrip("/")

def wait_until_ready(url: str, timeout_s: int = 25) -> None:
    """Attend que /health réponde OK (avec retries)."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"Service not ready at {url}/health") from last_err

def test_health():
    wait_until_ready(API_URL, timeout_s=25)
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.ok, f"/health not OK: {r.status_code} {r.text!r}"
