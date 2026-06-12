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
        "INSERT INTO categories(id,name,status,embedding_id,evidence_count,project_id)"
        " VALUES (1,'auth','active','cb:1',5,NULL)"
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


def test_mayring_reduce_inductive_half_creates_when_no_match(tmp_path, monkeypatch):
    """Kein cosine-Treffer → induktive Hälfte bildet neu (proposed) im Write-Target."""
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})

    # Monkeypatch record_proposal mit der NEUEN Signatur (ohne codebook_id).
    # Der Controller in MayringCoder/src/api/routes/codebooks.py passt die Definition an.
    # WHY(core-standalone 2026-06-08): dieser Test prüft die CONSUMER-Integration
    # (MayringCoder src/). Im core-eigenen CI existiert `src` nicht → sauber skippen
    # statt hart zu failen; im Consumer (vendored) läuft er normal.
    import pytest as _pytest
    _codebooks_mod = _pytest.importorskip(
        "src.api.routes.codebooks",
        reason="consumer-only (MayringCoder src/) — skipped in core-standalone",
    )

    def _fake_record_proposal(conn, candidate_label, *, paraphrase="",
                              parent_hint_id=None, pi_job_id="",
                              chunk_id=None, project_id=None):
        conn.execute(
            "INSERT OR IGNORE INTO categories(name, status, source, evidence_count) "
            "VALUES (?, 'proposed', 'induced', 1)",
            (candidate_label,),
        )
        row = conn.execute("SELECT id FROM categories WHERE name=?",
                           (candidate_label,)).fetchone()
        return row[0]

    monkeypatch.setattr(_codebooks_mod, "record_proposal", _fake_record_proposal)

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
        "SELECT name FROM categories WHERE name='ux_flow'").fetchone()
    assert row is not None


def test_mayring_reduce_dry_run_never_writes(tmp_path):
    """dry_run: liefert das induktive Label OHNE eine Kategorie anzulegen (Modell-Duelle
    dürfen das eine Codebook nicht verschmutzen)."""
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    before = conn.execute("SELECT count(*) FROM categories").fetchone()[0]
    res = mayring_reduce(
        "completely unrelated topic", theme="ux", conn=conn,
        chroma_categories=chroma, embed_fn=lambda s: [0.0, 1.0, 0.0],
        llm_fn=lambda p: json.dumps(
            {"paraphrase": "p", "generalization": "g", "label": "ux_flow"}),
        dry_run=True,
    )
    assert res.candidates[0].label == "ux_flow"
    assert res.candidates[0].match == "inductive"
    after = conn.execute("SELECT count(*) FROM categories").fetchone()[0]
    assert after == before          # KEIN Write
    assert conn.execute("SELECT id FROM categories WHERE name='ux_flow'").fetchone() is None


def test_mayring_reduce_dry_run_deductive_hit_read_only(tmp_path):
    """dry_run mit cosine-Treffer: meldet die bestehende Kategorie (deductive), schreibt nichts."""
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    res = mayring_reduce(
        "JWT login flow", theme="security", conn=conn, chroma_categories=chroma,
        embed_fn=lambda s: [1.0, 0.0, 0.0],
        llm_fn=lambda p: json.dumps(
            {"paraphrase": "x", "generalization": "y", "label": "auth_login"}),
        dry_run=True,
    )
    assert res.candidates[0].match == "deductive"
    assert res.candidates[0].label == "auth"


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


def test_parse_structured_handles_markdown_fence_and_thinking():
    """Regression 2026-06-05: LLMs wrappen JSON in ```json-Fences / <think>-Tokens →
    json.loads schlug fehl → Label wurde "json". Muss das echte Label extrahieren."""
    from mayring_core.memory.ingestion.mayring_process import _parse_structured
    fenced = '```json\n{"paraphrase":"p","generalization":"g","label":"auth"}\n```'
    assert _parse_structured(fenced) == ("p", "g", "auth")
    thinking = '<think>hmm let me reduce</think>\n{"paraphrase":"x","generalization":"y","label":"data_access"}'
    assert _parse_structured(thinking)[2] == "data_access"
    prose = 'Here is the result: {"paraphrase":"a","generalization":"b","label":"config"} done.'
    assert _parse_structured(prose)[2] == "config"
    # bare JSON weiterhin ok
    assert _parse_structured('{"label":"api"}')[2] == "api"
    # echtes Nicht-JSON → fail-soft auf clean label, NICHT "json"
    assert _parse_structured("auth_flow") == ("", "", "auth_flow")
