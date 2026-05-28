"""Phase 3: mixed-method, fail-closed categorization (`mayring_process`).

Server-side orchestration (User-Entscheid 2026-05-24: zentral in MayringCoder,
da der Pi-Agent keinen Direktzugriff auf die codebook_categories-Embeddings hat).
Deduktiv (Cosine) → Merge nach Score → induktiv (LLM + Pflicht-parent_hint) →
Embedding-Dedup → Proposal/chunk_categories-Write. Reine, injizierbare Funktionen;
die FastAPI-Route in src/api/routes/codebooks.py verdrahtet die echten Provider.

See docs/superpowers/specs/2026-05-24-phase3-mayring-process.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

# WHY(#270): Schwellen sind das Herz der Methodik — deduktiv schlägt induktiv nur,
# wenn die Cosine wirklich hoch ist; dazwischen wird beides als Evidenz gewertet.
_DEDUCTIVE_MIN = 0.78
_HYBRID_MIN = 0.55
_DEDUP_MIN = 0.92

EmbedFn = Callable[[str], list[float]]
LlmFn = Callable[[str], str]


@dataclass
class ProcessResult:
    category_id: int | None
    category_name: str | None
    decision: str  # deductive | hybrid | inductive | inductive-dedup
    confidence: float
    igio_axis: str | None
    proposed: bool


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _load_categories(conn: Any, codebook_id: int, statuses: tuple[str, ...],
                     project_id: str | None = None) -> list[dict]:
    """Active category scope (project-scoped codebook, Phase 3.2): the shared
    profile base (project_id IS NULL) ∪ the active project's own induced
    categories (project_id = active). With no active project → base only, so a
    session never sees another project's private categories."""
    placeholders = ",".join("?" for _ in statuses)
    if project_id:
        scope, extra = "AND (project_id IS NULL OR project_id = ?)", (project_id,)
    else:
        scope, extra = "AND project_id IS NULL", ()
    rows = conn.execute(
        "SELECT id, name, igio_axis, parent_id, embedding_id, status, evidence_count "
        f"FROM codebook_categories WHERE codebook_id=? AND status IN ({placeholders}) "
        f"{scope} ORDER BY id",
        (codebook_id, *statuses, *extra),
    ).fetchall()
    return [{"id": r[0], "name": r[1], "igio_axis": r[2], "parent_id": r[3],
             "embedding_id": r[4], "status": r[5], "evidence_count": r[6]} for r in rows]


def _category_embeddings(chroma: Any, cats: list[dict]) -> list[tuple[dict, list[float]]]:
    """Fetch each category's vector from Chroma, keyed by embedding_id.

    Categories without an embedding_id (endpoint-created proposals) are skipped —
    they simply can't participate in cosine matching until embedded.
    """
    ids = [c["embedding_id"] for c in cats if c["embedding_id"]]
    if chroma is None or not ids:
        return []
    data = chroma.get(ids=ids, include=["embeddings"])
    got_ids = list(data.get("ids") or [])
    embs = data.get("embeddings")
    # ChromaDB returns embeddings as numpy arrays → `x or []` raises ValueError.
    embs = [] if embs is None else embs
    by_id = {gid: embs[i] for i, gid in enumerate(got_ids)
             if i < len(embs) and embs[i] is not None and len(embs[i])}
    out: list[tuple[dict, list[float]]] = []
    for c in cats:
        emb = by_id.get(c["embedding_id"])
        if emb is not None and len(emb):
            out.append((c, list(emb)))
    return out


def _best_match(query_emb: list[float],
                pairs: list[tuple[dict, list[float]]]) -> tuple[dict | None, float]:
    best_cat: dict | None = None
    best_score = 0.0
    for cat, emb in pairs:
        s = _cosine(query_emb, emb)
        if best_cat is None or s > best_score:
            best_cat, best_score = cat, s
    return best_cat, best_score


def _infer_igio_axis(name: str) -> str | None:
    """Best-effort IGIO axis from the label until the importer backfills it
    (Phase 1 gap: igio_axis is NULL for all imported categories)."""
    low = name.lower()
    if any(k in low for k in ("ergebnis", "result", "finding", "outcome")):
        return "O"
    if any(k in low for k in ("limitation", "issue", "problem", "risk", "input")):
        return "I"
    if any(k in low for k in ("goal", "ziel", "objective", "aim")):
        return "G"
    if any(k in low for k in ("vorgehen", "method", "process", "approach")):
        return "V"
    return None


def _clean_label(raw: str) -> str:
    line = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    line = line.strip().strip("\"'`").strip()
    # collapse to a single snake_case-ish token group, max 60 chars
    return line[:60]


def _link_chunk(conn: Any, chunk_id: str, category_id: int, *,
                version: int, confidence: float, source: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO chunk_categories(chunk_id, category_id, "
        "codebook_version, confidence, source) VALUES (?,?,?,?,?)",
        (chunk_id, category_id, version, confidence, source))


def link_chunks_deductive(
    conn: Any, chroma_categories: Any,
    chunk_embeddings: list[tuple[str, list[float]]], *,
    project_id: str | None = None,
    min_score: float = _HYBRID_MIN, codebook_version: int = 1,
) -> int:
    """Cheap, LLM-free deductive linking for the bulk ingestion path.

    Matches each chunk embedding against ALL active codebook categories
    (cross-codebook — the embeddings route a chunk to its domain themselves, so
    no fragile source_type→codebook_id mapping is needed) and writes a
    chunk_categories row for the single best match >= min_score. NO proposals,
    NO new categories, NO LLM — this keeps a 500-file populate fast and free of
    proposal spam. The interactive /process path keeps the full mixed-method
    flow. Returns the number of links written. Fail-soft: empty category set or
    missing chroma → 0 (logs nothing, ingestion proceeds)."""
    if project_id:
        scope, extra = "AND (project_id IS NULL OR project_id = ?)", (project_id,)
    else:
        scope, extra = "AND project_id IS NULL", ()
    rows = conn.execute(
        "SELECT id, name, igio_axis, parent_id, embedding_id FROM codebook_categories "
        f"WHERE status='active' AND embedding_id != '' {scope} ORDER BY id", extra).fetchall()
    cats = [{"id": r[0], "name": r[1], "igio_axis": r[2], "parent_id": r[3],
             "embedding_id": r[4]} for r in rows]
    pairs = _category_embeddings(chroma_categories, cats)
    if not pairs:
        return 0
    linked = 0
    for chunk_id, emb in chunk_embeddings:
        if emb is None or not len(emb):
            continue
        cat, score = _best_match(list(emb), pairs)
        if cat is not None and score >= min_score:
            _link_chunk(conn, chunk_id, cat["id"], version=codebook_version,
                        confidence=score, source="deductive")
            linked += 1
    return linked


# Query-side category derivation (Reranker-v3 cat_match). Cache the active
# category embeddings so the hot search path never fetches them from Chroma per
# query — categories change only when the codebook is (re-)processed.
_CAT_EMB_CACHE: dict[str, tuple[float, list]] = {}
_CAT_EMB_TTL = 300.0


def _active_category_pairs(conn: Any, chroma_categories: Any,
                           project_id: str | None) -> list[tuple[dict, list[float]]]:
    import time as _t
    key = project_id or "__base__"
    now = _t.monotonic()
    hit = _CAT_EMB_CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    if project_id:
        scope, extra = "AND (project_id IS NULL OR project_id = ?)", (project_id,)
    else:
        scope, extra = "AND project_id IS NULL", ()
    rows = conn.execute(
        "SELECT id, name, igio_axis, parent_id, embedding_id FROM codebook_categories "
        f"WHERE status='active' AND embedding_id != '' {scope} ORDER BY id", extra).fetchall()
    cats = [{"id": r[0], "name": r[1], "igio_axis": r[2], "parent_id": r[3],
             "embedding_id": r[4]} for r in rows]
    pairs = _category_embeddings(chroma_categories, cats)
    # WHY(2026-05-28): do NOT negative-cache an empty result. The chroma
    # codebook_categories collection is empty right after a deploy cutover and
    # only repopulated by the post-deploy reembed; caching pairs=[] for the full
    # TTL kept query→category derivation (→ reranker-v3 cat_match) DEAD for up to
    # 300s AFTER the reembed → recurring red cat_match smoke (a reembed had no
    # immediate effect). Only cache a non-empty result; an empty one is re-tried
    # on the next call so a reembed takes effect immediately.
    if pairs:
        _CAT_EMB_CACHE[key] = (now + _CAT_EMB_TTL, pairs)
    return pairs


def derive_query_category_ids(
    conn: Any, chroma_categories: Any, query_emb: list[float], *,
    project_id: str | None = None, min_score: float = _HYBRID_MIN, top_n: int = 3,
) -> set[int]:
    """Query-side mirror of link_chunks_deductive: map the query embedding to the
    codebook category_ids it most likely belongs to (cosine >= min_score, up to
    top_n). The reranker's cat_match needs the QUERY's categories too — without
    this only callers that pass category_hint get cat_match, so the session-hook
    path (which doesn't) left reranker-v3 inert even after chunk_categories was
    populated by the deductive ingest link. Cheap: 1 query emb x N cached category
    embeddings. Fail-soft → empty set (no Chroma / no categories)."""
    if not query_emb:
        return set()
    pairs = _active_category_pairs(conn, chroma_categories, project_id)
    if not pairs:
        return set()
    scored = sorted(
        ((_cosine(query_emb, emb), cat["id"]) for cat, emb in pairs),
        key=lambda t: t[0], reverse=True,
    )
    return {cid for s, cid in scored[:top_n] if s >= min_score}


def mayring_process(
    text: str, task: str, codebook_id: int, *,
    conn: Any, chroma_categories: Any, embed_fn: EmbedFn, llm_fn: LlmFn,
    chunk_id: str | None = None, pi_job_id: str = "", codebook_version: int = 1,
    active_project_id: str | None = None,
) -> ProcessResult:
    """Mixed-method, fail-closed categorization. Raises ValueError (→ HTTP 400)
    when text/task are empty or the codebook has no active categories — no silent
    'uncategorized' fallback (that is exactly the #270 anti-pattern)."""
    from src.api.routes.codebooks import record_proposal  # late: avoid web import cycle

    if not (text or "").strip():
        raise ValueError("mayring_process: 'text' required (fail-closed)")
    if not (task or "").strip():
        raise ValueError("mayring_process: 'task' required (fail-closed)")

    active = _load_categories(conn, codebook_id, ("active",), active_project_id)
    if not active:
        raise ValueError(
            f"mayring_process: codebook {codebook_id} has no active categories (fail-closed)")

    text_emb = embed_fn(text)
    if not text_emb:
        raise ValueError("mayring_process: embedding failed (fail-closed)")

    active_pairs = _category_embeddings(chroma_categories, active)
    top_cat, score = _best_match(text_emb, active_pairs)

    # 3c-1 deduktiv: harte Zuordnung, kein LLM
    if top_cat is not None and score >= _DEDUCTIVE_MIN:
        if chunk_id:
            _link_chunk(conn, chunk_id, top_cat["id"], version=codebook_version,
                        confidence=score, source="deductive")
            conn.commit()
        return ProcessResult(top_cat["id"], top_cat["name"], "deductive",
                             round(score, 4), top_cat["igio_axis"], proposed=False)

    # 3c-2 hybrid: zuordnen UND Proposal (Evidenz auf dieselbe Kategorie)
    if top_cat is not None and score >= _HYBRID_MIN:
        record_proposal(conn, codebook_id, top_cat["name"], paraphrase=text[:200],
                        parent_hint_id=top_cat["parent_id"], igio_axis=top_cat["igio_axis"],
                        pi_job_id=pi_job_id, chunk_id=chunk_id, project_id=active_project_id)
        if chunk_id:
            _link_chunk(conn, chunk_id, top_cat["id"], version=codebook_version,
                        confidence=score, source="hybrid-merge")
        conn.commit()
        return ProcessResult(top_cat["id"], top_cat["name"], "hybrid",
                             round(score, 4), top_cat["igio_axis"], proposed=True)

    # 3b induktiv: LLM leitet Label ab; parent_hint = nächste deduktive Kategorie (PFLICHT)
    parent_hint_id = top_cat["id"] if top_cat is not None else active[0]["id"]
    prompt = (
        "Du klassifizierst Text nach qualitativer Inhaltsanalyse (Mayring).\n"
        f"Aufgabe/Kontext: {task[:200]}\n"
        f"Naheliegendste bestehende Kategorie: {top_cat['name'] if top_cat else '—'}\n"
        f"Text:\n{text[:1200]}\n\n"
        "Leite EINE neue, prägnante Kategorie (snake_case, max 4 Wörter) ab, die diesen "
        "Text als Unterkategorie der genannten Kategorie beschreibt. Antworte NUR mit dem Label."
    )
    label = _clean_label(llm_fn(prompt))
    if not label:
        raise ValueError("mayring_process: inductive label derivation failed (fail-closed)")

    igio = _infer_igio_axis(label)
    label_emb = embed_fn(label)

    # Embedding-Dedup: cosine > 0.92 zu existierender (aktiv ODER proposed) → Evidenz statt Neu
    all_cats = _load_categories(conn, codebook_id, ("active", "proposed"), active_project_id)
    dedup_cat, dedup_score = _best_match(label_emb, _category_embeddings(chroma_categories, all_cats))
    if dedup_cat is not None and dedup_score > _DEDUP_MIN:
        conn.execute("UPDATE codebook_categories SET evidence_count = evidence_count + 1 "
                     "WHERE id=?", (dedup_cat["id"],))
        if chunk_id:
            _link_chunk(conn, chunk_id, dedup_cat["id"], version=codebook_version,
                        confidence=dedup_score, source="inductive")
        conn.commit()
        return ProcessResult(dedup_cat["id"], dedup_cat["name"], "inductive-dedup",
                             round(dedup_score, 4), dedup_cat["igio_axis"], proposed=False)

    cat_id = record_proposal(conn, codebook_id, label, paraphrase=text[:200],
                             parent_hint_id=parent_hint_id, igio_axis=igio,
                             pi_job_id=pi_job_id, chunk_id=chunk_id,
                             project_id=active_project_id)
    # Embedding für künftige Dedup-Runden hinterlegen (Dedup-fix gegenüber leerem embedding_id)
    emb_id = f"cb:proposed:{cat_id}"
    if chroma_categories is not None and label_emb:
        chroma_categories.upsert(ids=[emb_id], embeddings=[label_emb], documents=[label])
        conn.execute("UPDATE codebook_categories SET embedding_id=? WHERE id=?", (emb_id, cat_id))
    if chunk_id:
        _link_chunk(conn, chunk_id, cat_id, version=codebook_version,
                    confidence=score, source="inductive")
    conn.commit()
    return ProcessResult(cat_id, label, "inductive", round(score, 4), igio, proposed=True)
