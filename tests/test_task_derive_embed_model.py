"""_embed_text must use the configured embedding model (bge-m3), not hardcoded nomic —
it runs on every /memory/search (derive_research_question_fast) and a dead embed model
cold-loaded the GPU and stalled the hook."""
from mayring_core.memory import task_derivation as td


def test_embed_text_uses_configured_model(monkeypatch):
    captured = {}

    def _fake_embed_single(url, model, text, *, timeout=240.0, **kw):
        captured["model"] = model
        captured["url"] = url
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("mayring_core.ollama_client.embed_single", _fake_embed_single)
    monkeypatch.setattr("mayring_core.config.EMBEDDING_MODEL", "bge-m3", raising=False)
    out = td._embed_text("probe", "http://gpu:11434")
    assert out == [0.1, 0.2, 0.3]
    assert captured["model"] == "bge-m3"  # NOT nomic-embed-text


def test_embed_text_failsoft_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr("mayring_core.ollama_client.embed_single", _boom)
    assert td._embed_text("x", "http://gpu:11434") is None
