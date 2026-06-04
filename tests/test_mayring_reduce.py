import json

from mayring_core.memory.ingestion.mayring_process import (
    ReduceResult,
    _structured_reduce_prompt,
    mayring_reduce,
    reduce_prompt,
)


def test_reduce_prompt_is_bare_label():
    p = reduce_prompt("auth code", "security")
    assert "NUR mit dem finalen Kategorie-Label" in p
    # KEIN Modus mehr — die Methode ist immer mixed
    assert "deduktiv" not in p.lower() and "induktiv" not in p.lower()


def test_structured_reduce_prompt_asks_json():
    p = _structured_reduce_prompt("auth code", "security", ["auth", "api"])
    assert "paraphrase" in p and "generalization" in p and "label" in p
    assert "JSON" in p
    assert "auth, api" in p  # Granularitäts-Beispiele


class _Chroma:
    def __init__(self, vecs):
        self._v = vecs

    def get(self, ids, include):
        out_ids, out = [], []
        for i in ids:
            if i in self._v:
                out_ids.append(i)
                out.append(self._v[i])
        return {"ids": out_ids, "embeddings": out}

    def upsert(self, ids, embeddings, documents):
        for i, e in zip(ids, embeddings):
            self._v[i] = e


def _conn_with_one_active(tmp_path):
    from mayring_core.memory.store import init_memory_db
    conn = init_memory_db(tmp_path / "m.db")
    conn.execute(
        "INSERT INTO codebooks(slug,description,auto_promote_threshold,created_at,updated_at)"
        " VALUES ('code','',3,'2026-01-01T00:00:00','2026-01-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO codebook_categories(id,codebook_id,name,status,embedding_id,"
        "evidence_count,project_id) VALUES (1,1,'auth','active','cb:1',5,NULL)"
    )
    conn.commit()
    return conn


def test_mayring_reduce_deductive_half_hits_existing(tmp_path):
    """Treffer >=0.70 → bestehende Kategorie (deduktive Hälfte der EINEN Methode)."""
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    res = mayring_reduce(
        "JWT login flow", theme="security", conn=conn,
        chroma_categories=chroma,
        embed_fn=lambda s: [1.0, 0.0, 0.0],
        llm_fn=lambda p: json.dumps(
            {"paraphrase": "uses jwt", "generalization": "authn", "label": "auth_login"}),
    )
    assert isinstance(res, ReduceResult)
    assert res.paraphrase == "uses jwt"
    assert res.generalization == "authn"
    assert res.candidates[0].match == "deductive"
    assert res.candidates[0].label == "auth"          # bestehende, nicht das Roh-Label
    assert res.candidates[0].score >= 0.70


def test_mayring_reduce_always_structured_even_if_llm_returns_bare(tmp_path):
    """Fail-soft: liefert das LLM doch nur ein bare Label, bleibt die Methode nutzbar
    (paraphrase/generalization leer, Kategorie trotzdem zugeordnet)."""
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    res = mayring_reduce(
        "JWT login flow", theme="security", conn=conn,
        chroma_categories=chroma, embed_fn=lambda s: [1.0, 0.0, 0.0],
        llm_fn=lambda p: "auth_login",
    )
    assert res.paraphrase == "" and res.generalization == ""
    assert res.candidates[0].label == "auth"
    assert res.candidates[0].match == "deductive"


def test_mayring_reduce_inductive_half_creates_when_no_match(tmp_path):
    """Kein cosine-Treffer → induktive Hälfte bildet neu (proposed) im Write-Target."""
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    # Kandidat-Embedding orthogonal zur Bestandskategorie → kein Treffer → neu
    res = mayring_reduce(
        "completely unrelated topic", theme="ux", conn=conn,
        chroma_categories=chroma, embed_fn=lambda s: [0.0, 1.0, 0.0],
        llm_fn=lambda p: json.dumps(
            {"paraphrase": "p", "generalization": "g", "label": "ux_flow"}),
        chunk_id="chk1",
    )
    assert res.candidates[0].label == "ux_flow"
    assert res.candidates[0].match in ("inductive", "dedup")
    row = conn.execute(
        "SELECT name FROM codebook_categories WHERE name='ux_flow'").fetchone()
    assert row is not None


def test_mayring_reduce_fail_closed_on_empty(tmp_path):
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    import pytest
    with pytest.raises(ValueError):
        mayring_reduce("", theme="security", conn=conn,
                       chroma_categories=chroma, embed_fn=lambda s: [1.0],
                       llm_fn=lambda p: "x")
    with pytest.raises(ValueError):
        mayring_reduce("text", theme="", conn=conn,
                       chroma_categories=chroma, embed_fn=lambda s: [1.0],
                       llm_fn=lambda p: "x")
