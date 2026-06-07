"""keep_alive pinning: embed + generate must send keep_alive so the GPU host keeps
the hot-path models (bge-m3 + qwen3.5-mayring:2b) resident instead of evicting them."""
import httpx

from mayring_core import ollama_client as oc


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_embed_batch_sends_keep_alive(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, timeout=None, verify=None):
        captured["json"] = json
        return _Resp({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr(httpx, "post", _fake_post)
    out = oc.embed_batch("http://gpu:11434", "bge-m3", ["hi"])
    assert out == [[0.1, 0.2]]
    assert "keep_alive" in captured["json"]
    assert captured["json"]["keep_alive"] == oc._KEEP_ALIVE


def test_embed_single_sends_keep_alive(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, timeout=None, verify=None):
        captured["json"] = json
        return _Resp({"embedding": [0.3]})

    monkeypatch.setattr(httpx, "post", _fake_post)
    out = oc.embed_single("http://gpu:11434", "bge-m3", "hi")
    assert out == [0.3]
    assert captured["json"].get("keep_alive") == oc._KEEP_ALIVE


def test_default_keep_alive_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "-1")
    assert oc._default_keep_alive() == -1
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    assert oc._default_keep_alive() == "30m"
