import json, urllib.request

def test_health():
    with urllib.request.urlopen("http://localhost:5001/health") as r:
        data = json.loads(r.read().decode())
        assert data.get("ok") is True
