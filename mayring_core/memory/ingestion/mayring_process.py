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
from datetime import datetime, timezone
from typing import Any, Callable

# WHY(canonical Mayring method): die EINE Schwelle für die Zuordnungs-Entscheidung —
# matcht die ABGELEITETE Kategorie (Paraphrase→Generalisierung→Reduktion) eine
# vorhandene ziel-bezogene Kategorie mit cosine >= 0.75 → deduktiv zuordnen; sonst
# induktiv neu bilden. Siehe [[feedback-mayring-canonical-method]].
_MATCH_MIN = 0.75
# Embedding-Dedup beim induktiven Neu-Bilden: zwei Labels >0.92 = dieselbe Kategorie
# unter anderem Namen → Evidenz statt Fragment (live: 3× "auth" durch unkontrolliertes Neu).
_DEDUP_MIN = 0.92
# Query-/Backfill-seitige Cosine-Plumbing (reranker-v3 cat_match-Coverage, NICHT die
# Kategorisierungs-Entscheidung): bewusst großzügiger, sonst sinkt cat_match-Coverage.
_HYBRID_MIN = 0.55

EmbedFn = Callable[[str], list[float]]
LlmFn = Callable[[str], str]
BatchEmbedFn = Callable[[list[str]], list[list[float]]]
BatchReduceFn = Callable[[list[tuple[str, str]]], list[str]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load_categories(conn: Any, codebook_id: int | None, statuses: tuple[str, ...],
                     project_id: str | None = None) -> list[dict]:
    """Category scope. codebook_id=None → CROSS-codebook (alle Codebooks) — der
    Bulk-Pfad matcht so domänen-übergreifend (das Embedding routet den Chunk selbst,
    kein fragiles source_type→codebook-Mapping); ein konkretes codebook_id scoped auf
    eines (interaktiv: der gewählte Codebook). project_id: shared base (NULL) ∪ die
    eigenen induzierten Kategorien des aktiven Projekts."""
    placeholders = ",".join("?" for _ in statuses)
    if project_id:
        scope, extra = "AND (project_id IS NULL OR project_id = ?)", (project_id,)
    else:
        scope, extra = "AND project_id IS NULL", ()
    cb_clause, cb_extra = ("AND codebook_id=?", (codebook_id,)) if codebook_id is not None else ("", ())
    rows = conn.execute(
        "SELECT id, name, igio_axis, parent_id, embedding_id, status, evidence_count "
        f"FROM codebook_categories WHERE status IN ({placeholders}) {cb_clause} "
        f"{scope} ORDER BY id",
        (*statuses, *cb_extra, *extra),
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


def _best_matches(query_emb: list[float],
                  pairs: list[tuple[dict, list[float]]], *,
                  top_n: int, min_score: float) -> list[tuple[dict, float]]:
    """Top-N categories by cosine, each >= min_score. Multi-label variant of
    _best_match for the category-consolidation (one cosine pass → several FK
    links, replacing the separate LLM multi-label categorize)."""
    scored = sorted(
        ((_cosine(query_emb, emb), cat) for cat, emb in pairs),
        key=lambda t: t[0], reverse=True,
    )
    return [(cat, s) for s, cat in scored[:top_n] if s >= min_score]


def derive_labels_from_categories(
    conn: Any, chunk_ids: list[str],
) -> dict[str, list[str]]:
    """Derive the legacy free-string category_labels from the structured
    chunk_categories FK (highest-confidence first). Lets consumers that read
    chunks.category_labels keep working after the LLM mayring_categorize pass is
    retired — one cosine SoT (chunk_categories), labels are a derived view."""
    if not chunk_ids:
        return {}
    ph = ",".join("?" for _ in chunk_ids)
    out: dict[str, list[str]] = {}
    for cid, name in conn.execute(
        "SELECT cc.chunk_id, cat.name FROM chunk_categories cc "
        "JOIN codebook_categories cat ON cat.id = cc.category_id "
        f"WHERE cc.chunk_id IN ({ph}) ORDER BY cc.confidence DESC",
        tuple(chunk_ids),
    ).fetchall():
        labels = out.setdefault(cid, [])
        if name and name not in labels:
            labels.append(name)
    return out


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
    top_n: int = 1,
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
        if top_n <= 1:
            cat, score = _best_match(list(emb), pairs)
            matches = [(cat, score)] if (cat is not None and score >= min_score) else []
        else:
            matches = _best_matches(list(emb), pairs, top_n=top_n, min_score=min_score)
        for cat, score in matches:
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


def reduce_prompt(text: str, task: str) -> str:
    """Mayrings Reduktion als Prompt: aus Rohtext, GEBUNDEN AN DAS ZIEL, über
    Paraphrase→Generalisierung→Reduktion EINE Kategorie ableiten. Der Zielbezug ist
    obligatorisch (sonst random Kategorien). Siehe [[feedback-mayring-canonical-method]]."""
    return (
        "Du bildest eine Kategorie nach qualitativer Inhaltsanalyse (Mayring).\n"
        f"ZIEL/Aufgabe (obligatorischer Bezug): {task[:300]}\n"
        f"Textstelle:\n{text[:1200]}\n\n"
        "Gehe in drei Schritten vor:\n"
        "1. PARAPHRASE: gib den inhaltstragenden Kern der Stelle knapp wieder.\n"
        "2. GENERALISIERUNG: hebe ihn auf das Abstraktionsniveau des Ziels.\n"
        "3. REDUKTION: verdichte zu EINER prägnanten Kategorie (snake_case, max 4 Wörter), "
        "die den Bezug zum Ziel wahrt.\n"
        "Antworte NUR mit dem finalen Kategorie-Label (snake_case), nichts anderes."
    )


def _promote_threshold(conn: Any, codebook_id: int) -> int:
    row = conn.execute(
        "SELECT auto_promote_threshold FROM codebooks WHERE id=?", (codebook_id,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 3


def _bump_evidence(conn: Any, cat_id: int) -> int:
    """Increment evidence_count and return the NEW value (for promotion check)."""
    conn.execute("UPDATE codebook_categories SET evidence_count = evidence_count + 1 "
                 "WHERE id=?", (cat_id,))
    row = conn.execute("SELECT evidence_count FROM codebook_categories WHERE id=?",
                       (cat_id,)).fetchone()
    return int(row[0]) if row else 0


def _maybe_promote(conn: Any, cat_id: int, status: str, evidence_count: int,
                   threshold: int) -> bool:
    """proposed→active sobald genug Evidenz: erst dann sieht der Reranker (cat_match,
    active-only) die induktiv gebildete Kategorie. Ohne diesen Schritt bleibt induktiver
    Output inert (Defekt vor canonical-Konsolidierung)."""
    if status != "active" and evidence_count >= threshold:
        conn.execute("UPDATE codebook_categories SET status='active', promoted_at=? WHERE id=?",
                     (_now_iso(), cat_id))
        conn.execute("UPDATE codebook_proposals SET decision='promote', reviewed_by='auto' "
                     "WHERE category_id=? AND decision IS NULL", (cat_id,))
        return True
    return False


def _assign_or_create(
    conn: Any, chroma_categories: Any, target_codebook_id: int, chunk_id: str | None,
    candidate_label: str, candidate_emb: list[float],
    active_pairs: list[tuple[dict, list[float]]],
    dedup_pairs: list[tuple[dict, list[float]]], *,
    task: str, codebook_version: int, project_id: str | None,
    promote_threshold: int, pi_job_id: str = "",
) -> ProcessResult:
    """Die geteilte Entscheidung beider Pfade (Einzel + Bulk) — EINE Logik, kein Duplikat.
    Erwartet die ABGELEITETE Kandidat-Kategorie (Label + Embedding) und die geladenen
    Kategorie-Mengen: `active_pairs` für den deduktiven Match (vom Caller gescoped:
    cross-codebook im Bulk, ein Codebook interaktiv), `dedup_pairs` (active+proposed) fürs
    induktive Dedup. `target_codebook_id` = wohin eine echte Neu-Kategorie geschrieben wird.
    `dedup_pairs` wird bei einer Neu-Kategorie IN-PLACE erweitert → nachfolgende Chunks
    desselben Batches dedupen darauf (intra-batch, ohne Reload). Schritte 3-5 des Ablaufs."""
    # 3+4 deduktiv: abgeleitete Kategorie matcht eine vorhandene >= 0.75 → vorhandene nutzen
    top_cat, score = _best_match(candidate_emb, active_pairs)
    if top_cat is not None and score >= _MATCH_MIN:
        if chunk_id:
            _link_chunk(conn, chunk_id, top_cat["id"], version=codebook_version,
                        confidence=score, source="deductive")
        return ProcessResult(top_cat["id"], top_cat["name"], "deductive",
                             round(score, 4), top_cat.get("igio_axis"), proposed=False)

    # 5 induktiv: keine vorhandene passt. Erst HART dedupen (active ODER proposed), sonst
    # entsteht ein zweites Fragment derselben Kategorie.
    dedup_cat, dedup_score = _best_match(candidate_emb, dedup_pairs)
    if dedup_cat is not None and dedup_score > _DEDUP_MIN:
        new_count = _bump_evidence(conn, dedup_cat["id"])
        if chunk_id:
            _link_chunk(conn, chunk_id, dedup_cat["id"], version=codebook_version,
                        confidence=dedup_score, source="inductive")
        promoted = _maybe_promote(conn, dedup_cat["id"], dedup_cat.get("status", "proposed"),
                                  new_count, promote_threshold)
        return ProcessResult(dedup_cat["id"], dedup_cat["name"], "inductive-dedup",
                             round(dedup_score, 4), dedup_cat.get("igio_axis"),
                             proposed=(dedup_cat.get("status") != "active" and not promoted))

    # echte Neu-Kategorie (ziel-gebunden gebildet) als 'proposed' im Ziel-Codebook; Embedding
    # hinterlegen, damit sie ab Promotion für cat_match matchbar ist + Folge-Runden dedupen.
    from src.api.routes.codebooks import record_proposal  # late: web import cycle
    parent_hint_id = top_cat["id"] if top_cat is not None else None
    igio = _infer_igio_axis(candidate_label)
    cat_id = record_proposal(conn, target_codebook_id, candidate_label, paraphrase=task[:200],
                             parent_hint_id=parent_hint_id, igio_axis=igio,
                             pi_job_id=pi_job_id, chunk_id=chunk_id, project_id=project_id)
    emb_id = f"cb:proposed:{cat_id}"
    if chroma_categories is not None and candidate_emb:
        chroma_categories.upsert(ids=[emb_id], embeddings=[candidate_emb],
                                 documents=[candidate_label])
        conn.execute("UPDATE codebook_categories SET embedding_id=? WHERE id=?",
                     (emb_id, cat_id))
    if chunk_id:
        _link_chunk(conn, chunk_id, cat_id, version=codebook_version,
                    confidence=score, source="inductive")
    promoted = _maybe_promote(conn, cat_id, "proposed", 1, promote_threshold)  # frisch=evidence 1
    new_cat = {"id": cat_id, "name": candidate_label, "igio_axis": igio,
               "parent_id": parent_hint_id, "embedding_id": emb_id,
               "status": "active" if promoted else "proposed", "evidence_count": 1}
    dedup_pairs.append((new_cat, candidate_emb))          # intra-batch Dedup
    if promoted:
        active_pairs.append((new_cat, candidate_emb))     # ab jetzt auch deduktiv matchbar
    return ProcessResult(cat_id, candidate_label, "inductive", round(score, 4), igio,
                         proposed=not promoted)


def mayring_process(
    text: str, task: str, codebook_id: int, *,
    conn: Any, chroma_categories: Any, embed_fn: EmbedFn, llm_fn: LlmFn,
    chunk_id: str | None = None, pi_job_id: str = "", codebook_version: int = 1,
    active_project_id: str | None = None,
) -> ProcessResult:
    """Die EINE Mayring-Methode (Einzel-Pfad: interaktiv + Advisor). IMMER mixed,
    IMMER ziel-gebunden. Ablauf: Ziel(=task) → REDUKTION ZUERST (Paraphrase→
    Generalisierung→Reduktion) → cosine 0.75 gegen vorhandene (im gewählten Codebook)
    → Treffer=deduktiv, sonst induktiv neu. Raises ValueError (→ HTTP 400) bei leerem
    text/task. Siehe [[feedback-mayring-canonical-method]]."""
    if not (text or "").strip():
        raise ValueError("mayring_process: 'text' required (fail-closed)")
    if not (task or "").strip():
        raise ValueError("mayring_process: 'task' required — Zielbezug ist obligatorisch")

    # Schritt 1+2: aus Text GEBUNDEN AN DAS ZIEL die Kandidat-Kategorie ableiten.
    candidate = _clean_label(llm_fn(reduce_prompt(text, task)))
    if not candidate:
        raise ValueError("mayring_process: Reduktion lieferte kein Label (fail-closed)")
    candidate_emb = embed_fn(candidate)
    if not candidate_emb:
        raise ValueError("mayring_process: embedding der Kategorie fehlgeschlagen (fail-closed)")

    # Interaktiv: Match + Create im gewählten Codebook.
    active_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, codebook_id, ("active",), active_project_id))
    dedup_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, codebook_id, ("active", "proposed"), active_project_id))
    res = _assign_or_create(
        conn, chroma_categories, codebook_id, chunk_id, candidate, candidate_emb,
        active_pairs, dedup_pairs, task=task, codebook_version=codebook_version,
        project_id=active_project_id, promote_threshold=_promote_threshold(conn, codebook_id),
        pi_job_id=pi_job_id)
    conn.commit()
    return res


def categorize_chunks(
    items: list[tuple[str, str]], task: str, target_codebook_id: int, *,
    conn: Any, chroma_categories: Any,
    batch_embed_fn: BatchEmbedFn, batch_reduce_fn: BatchReduceFn,
    match_codebook_id: int | None = None,
    codebook_version: int = 1, project_id: str | None = None,
) -> list[ProcessResult]:
    """Bulk-Pfad der EINEN Methode: dieselbe `_assign_or_create`-Logik wie mayring_process,
    aber die teure Reduktion + das Kandidaten-Embedding laufen GEBATCHT (eine LLM-Reduktion
    + ein Embed-Call pro Batch, über die zentrale PiQueue/cloud-split) statt pro Chunk.
    Der deduktive Match ist CROSS-codebook (`match_codebook_id=None`) — das Embedding routet
    den Chunk in seine Domäne, kein fragiles source_type→codebook-Mapping. Echte Neu-Kategorien
    landen im `target_codebook_id` (aus der Domäne der Quelle aufgelöst). items = [(chunk_id, text)].
    KEIN separater cosine-only-Pfad mehr — Bulk macht jetzt das volle goal-anchored Mayring."""
    if not items:
        return []
    if not (task or "").strip():
        raise ValueError("categorize_chunks: 'task' required — Zielbezug ist obligatorisch")
    labels = batch_reduce_fn([(text, task) for _, text in items])
    if len(labels) != len(items):
        raise ValueError(
            f"categorize_chunks: Reduktion lieferte {len(labels)} Labels für {len(items)} Chunks")
    candidates = [_clean_label(lbl) for lbl in labels]
    embs = batch_embed_fn(candidates)
    if len(embs) != len(candidates):
        raise ValueError("categorize_chunks: Embedding-Anzahl != Kandidaten-Anzahl")

    threshold = _promote_threshold(conn, target_codebook_id)
    # active (deduktiver Match) + dedup (active+proposed) EINMAL pro Batch laden; neue
    # Kategorien werden von _assign_or_create in-place angehängt → intra-batch sichtbar.
    active_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, match_codebook_id, ("active",), project_id))
    dedup_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, match_codebook_id, ("active", "proposed"), project_id))
    out: list[ProcessResult] = []
    for (chunk_id, _text), candidate, emb in zip(items, candidates, embs):
        if not candidate or not emb:
            continue
        out.append(_assign_or_create(
            conn, chroma_categories, target_codebook_id, chunk_id, candidate, list(emb),
            active_pairs, dedup_pairs, task=task, codebook_version=codebook_version,
            project_id=project_id, promote_threshold=threshold))
    conn.commit()
    return out
