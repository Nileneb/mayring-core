import json
from mayring_core.memory.ingestion.mayring_process import reduce_prompt, mayring_reduce, ReduceResult


def test_reduce_prompt_mode_deductive_forbids_new():
    p = reduce_prompt("auth code", "security", ["auth", "api"], mode="deductive")
    assert "security" in p
    assert "auth, api" in p
    assert "keine neue" in p.lower() or "nur aus" in p.lower()


def test_reduce_prompt_mode_inductive_no_anchor_pressure():
    p = reduce_prompt("auth code", "security", ["auth"], mode="inductive")
    assert "frei" in p.lower() or "neu" in p.lower()


def test_reduce_prompt_default_is_bare_label():
    p = reduce_prompt("auth code", "security")
    assert "NUR mit dem finalen Kategorie-Label" in p


def test_reduce_prompt_structured_asks_json():
    p = reduce_prompt("auth code", "security", structured=True)
    assert "paraphrase" in p and "generalization" in p and "label" in p
    assert "JSON" in p


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


def test_mayring_reduce_deductive_hit(tmp_path):
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    res = mayring_reduce(
        "JWT login flow", theme="security", codebook_id=1, conn=conn,
        chroma_categories=chroma,
        embed_fn=lambda s: [1.0, 0.0, 0.0],
        llm_fn=lambda p: "auth_login",
        reduce=True, mode="hybrid",
    )
    assert isinstance(res, ReduceResult)
    assert res.candidates[0].match == "deductive"
    assert res.candidates[0].label == "auth"
    assert res.candidates[0].score >= 0.70


def test_mayring_reduce_structured_fields(tmp_path):
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    res = mayring_reduce(
        "JWT login flow", theme="security", codebook_id=1, conn=conn,
        chroma_categories=chroma, embed_fn=lambda s: [1.0, 0.0, 0.0],
        llm_fn=lambda p: json.dumps({"paraphrase": "uses jwt",
                                     "generalization": "authn", "label": "auth_login"}),
        reduce=True, structured=True,
    )
    assert res.paraphrase == "uses jwt"
    assert res.generalization == "authn"
    assert res.candidates[0].match == "deductive"


def test_reduce_false_never_calls_llm(tmp_path):
    conn = _conn_with_one_active(tmp_path)
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0]})
    calls = {"llm": 0}

    def _boom(_p):
        calls["llm"] += 1
        raise AssertionError("LLM must NOT be called when reduce=False (#330/§5.1)")

    res = mayring_reduce(
        "JWT login flow", theme="security", codebook_id=1, conn=conn,
        chroma_categories=chroma, embed_fn=lambda s: [1.0, 0.0, 0.0],
        llm_fn=_boom, reduce=False, chunk_id="chk1",
    )
    assert calls["llm"] == 0
    assert res.candidates and res.candidates[0].match == "deductive"
    n = conn.execute("SELECT COUNT(*) FROM chunk_categories WHERE chunk_id='chk1'").fetchone()[0]
    assert n == 1


def test_reduce_false_multilabel_top_n(tmp_path):
    conn = _conn_with_one_active(tmp_path)
    conn.execute(
        "INSERT INTO codebook_categories(id,codebook_id,name,status,embedding_id,"
        "evidence_count,project_id) VALUES (2,1,'api','active','cb:2',5,NULL)"
    )
    conn.commit()
    chroma = _Chroma({"cb:1": [1.0, 0.0, 0.0], "cb:2": [0.9, 0.1, 0.0]})
    res = mayring_reduce(
        "auth and routing", theme="backend", codebook_id=1, conn=conn,
        chroma_categories=chroma, embed_fn=lambda s: [0.97, 0.05, 0.0],
        llm_fn=lambda p: "x", reduce=False, top_n=2,
    )
    labels = {c.label for c in res.candidates}
    assert labels == {"auth", "api"}
