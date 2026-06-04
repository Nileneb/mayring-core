"""#330: deductive chunk→codebook linking must run even without an LLM model.

The bug: ingest() gated category linking on `model` (it only ran the full LLM
mixed-method categorize_chunks). A categorize-off / no-model ingest therefore
wrote 0 chunk_categories → reranker-v3 cat_match silently lost coverage for that
chunk (the production `ingest_links_categories` smoke went red). The cheap,
deterministic link_chunks_deductive needs no model — it must run on the
already-computed embeddings when do_link is on but no model is available.
"""
from typing import Any

from mayring_core.memory.ingestion.core import ingest
from mayring_core.memory.schema import Source
from mayring_core.memory.store import init_memory_db


class _FakeCollection:
    def __init__(self):
        self._n = 0

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        self._n += 1

    def count(self) -> int:
        return self._n


def test_deductive_link_runs_without_llm_model(monkeypatch, tmp_path):
    fake_emb = [0.1] * 384
    import mayring_core.providers as _providers
    monkeypatch.setattr(_providers, "embed_texts", lambda texts, url: [fake_emb] * len(texts))

    import mayring_core.memory.ingestion.conversation_filter as _cf
    monkeypatch.setattr(_cf, "should_skip_chunk", lambda text, stype: (False, ""))

    captured: dict[str, Any] = {}
    import mayring_core.memory.ingestion.mayring_process as _mp

    def _fake_link(conn, chroma, chunk_embs, *, project_id=None, **kw):
        captured["chunk_embs"] = chunk_embs
        return len(chunk_embs)

    monkeypatch.setattr(_mp, "link_chunks_deductive", _fake_link)
    monkeypatch.setattr(_mp, "derive_labels_from_categories", lambda conn, ids: {})
    import mayring_core.memory.store as _store
    monkeypatch.setattr(_store, "get_chroma_collection", lambda name=None: object())

    db_conn = init_memory_db(tmp_path / "memory.db")
    source = Source(
        source_id="test:owner/repo:auth.py",
        source_type="repo_file",
        repo="owner/repo",
        path="auth.py",
        visibility="private",
        user_id="u-7",
    )

    result = ingest(
        source=source,
        content="User authentication and login flow with JWT and password hashing.",
        conn=db_conn,
        chroma_collection=_FakeCollection(),
        ollama_url="http://localhost:11434",
        model="",  # NO LLM model — must still link via the deductive path
        opts={"categorize": False, "link_categories": True},
    )

    assert "chunk_embs" in captured, (
        "link_chunks_deductive must run when do_link is on and no model is available"
    )
    assert all(e for _, e in captured["chunk_embs"]), "embeddings must be passed through"
    assert result["category_links"] >= 1, "category_links must reflect the deductive links"


def test_categorize_false_uses_cheap_deductive_not_llm(monkeypatch, tmp_path):
    """#330 flake fix: categorize=False with a model PRESENT must use the cheap
    deterministic deductive link, NOT the ~40s LLM categorize_chunks (gated on
    do_categorize now, not do_link+model)."""
    fake_emb = [0.1] * 384
    import mayring_core.providers as _providers
    monkeypatch.setattr(_providers, "embed_texts", lambda texts, url: [fake_emb] * len(texts))
    import mayring_core.memory.ingestion.conversation_filter as _cf
    monkeypatch.setattr(_cf, "should_skip_chunk", lambda text, stype: (False, ""))

    flags = {"deductive": False, "llm": False}
    import mayring_core.memory.ingestion.mayring_process as _mp

    def _fake_link(conn, chroma, chunk_embs, *, project_id=None, **kw):
        flags["deductive"] = True
        return len(chunk_embs)

    def _fake_categorize(*a, **k):
        flags["llm"] = True
        return []

    monkeypatch.setattr(_mp, "link_chunks_deductive", _fake_link)
    monkeypatch.setattr(_mp, "categorize_chunks", _fake_categorize)
    monkeypatch.setattr(_mp, "derive_labels_from_categories", lambda conn, ids: {})
    import mayring_core.memory.store as _store
    monkeypatch.setattr(_store, "get_chroma_collection", lambda name=None: object())

    db_conn = init_memory_db(tmp_path / "memory.db")
    source = Source(source_id="test:o/r:a.py", source_type="repo_file",
                    repo="o/r", path="a.py", visibility="private", user_id="u-7")
    ingest(
        source=source, content="auth login jwt flow", conn=db_conn,
        chroma_collection=_FakeCollection(), ollama_url="http://localhost:11434",
        model="real-text-model",  # model PRESENT, but categorize is OFF
        opts={"categorize": False, "link_categories": True},
    )
    assert flags["deductive"] is True, "categorize=False must use the cheap deductive link"
    assert flags["llm"] is False, "categorize=False must NOT invoke the LLM categorize_chunks"


def test_no_link_when_link_categories_disabled(monkeypatch, tmp_path):
    """do_link=False → neither path runs → 0 links (no accidental linking)."""
    fake_emb = [0.1] * 384
    import mayring_core.providers as _providers
    monkeypatch.setattr(_providers, "embed_texts", lambda texts, url: [fake_emb] * len(texts))
    import mayring_core.memory.ingestion.conversation_filter as _cf
    monkeypatch.setattr(_cf, "should_skip_chunk", lambda text, stype: (False, ""))

    ran = {"called": False}
    import mayring_core.memory.ingestion.mayring_process as _mp

    def _fake_link(*a, **k):
        ran["called"] = True
        return 0

    monkeypatch.setattr(_mp, "link_chunks_deductive", _fake_link)

    db_conn = init_memory_db(tmp_path / "memory.db")
    source = Source(source_id="test:o/r:x.py", source_type="repo_file",
                    repo="o/r", path="x.py", visibility="private", user_id="u-7")
    result = ingest(
        source=source, content="some content here", conn=db_conn,
        chroma_collection=_FakeCollection(), ollama_url="http://localhost:11434",
        model="", opts={"categorize": False, "link_categories": False},
    )
    assert ran["called"] is False
    assert result["category_links"] == 0
