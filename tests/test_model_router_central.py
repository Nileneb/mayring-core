"""Zentrale Modell-Konfig (app.linn.games /api/mcp/model-config) im ModelRouter."""
import mayring_core.model_router as mr
from mayring_core.model_router import ModelRouter


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class _FakeHttpx:
    def __init__(self, resp=None, exc=None):
        self._resp, self._exc = resp, exc
        self.calls = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        if self._exc:
            raise self._exc
        return self._resp


def _enable_central(monkeypatch, fake):
    mr._CENTRAL_CACHE.clear()
    monkeypatch.setenv("MAYRING_MODEL_CONFIG_URL", "http://web")
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "svc")
    monkeypatch.setattr(mr, "_httpx", fake)


def test_central_config_wins_over_text_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "text_model.txt").write_text("qwen3.5-mayring:2b", encoding="utf-8")
    fake = _FakeHttpx(_FakeResp({"config": {"text": {"model": "gemma4:e4b", "options": {"think": False}}}}))
    _enable_central(monkeypatch, fake)

    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") == "gemma4:e4b"          # zentral schlägt text_model.txt
    assert r.options_for("text") == {"think": False}


def test_no_central_url_preserves_legacy_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("MAYRING_MODEL_CONFIG_URL", raising=False)
    (tmp_path / "text_model.txt").write_text("qwen3.5-mayring:2b", encoding="utf-8")
    mr._CENTRAL_CACHE.clear()

    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") == "qwen3.5-mayring:2b"  # null Verhaltensänderung ohne env


def test_central_fail_soft_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "text_model.txt").write_text("qwen3.5-mayring:2b", encoding="utf-8")
    _enable_central(monkeypatch, _FakeHttpx(exc=RuntimeError("backend down")))

    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") == "qwen3.5-mayring:2b"  # Fehler → Fallback, kein Crash


def test_central_only_overrides_configured_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    fake = _FakeHttpx(_FakeResp({"config": {"text": {"model": "gemma4:e4b"}}}))
    _enable_central(monkeypatch, fake)

    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") == "gemma4:e4b"
    assert r.resolve("embedding") != "gemma4:e4b"     # kein zentraler embedding-Eintrag → yaml
    assert r.options_for("embedding") == {}
