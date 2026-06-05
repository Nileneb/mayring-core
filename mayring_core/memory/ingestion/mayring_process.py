"""Phase 3: mixed-method, fail-closed categorization (`mayring_process`).

Server-side orchestration (User-Entscheid 2026-05-24: zentral in MayringCoder,
da der Pi-Agent keinen Direktzugriff auf die codebook_categories-Embeddings hat).
Deduktiv (Cosine) → Merge nach Score → induktiv (LLM + Pflicht-parent_hint) →
Embedding-Dedup → Proposal/chunk_categories-Write. Reine, injizierbare Funktionen;
die FastAPI-Route in src/api/routes/codebooks.py verdrahtet die echten Provider.

See docs/superpowers/specs/2026-05-24-phase3-mayring-process.md.
"""
from __future__ import annotations

import json as _json_mod
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

# WHY(canonical Mayring method): die EINE Schwelle für die Zuordnungs-Entscheidung —
# matcht die ABGELEITETE Kategorie (Paraphrase→Generalisierung→Reduktion) eine
# vorhandene ziel-bezogene Kategorie mit cosine >= 0.70 → deduktiv zuordnen; sonst
# induktiv neu bilden. 0.70 (vorher 0.75) = mehr Wiederverwendung statt Fragmente, da
# reduzierte Labels spezifischer sind als die breiten Bestands-Kategorien (User 2026-05-29:
# „mehr mergen") — zusammen mit der Granularitäts-Kalibrierung im reduce_prompt.
# Siehe [[feedback-mayring-canonical-method]].
_MATCH_MIN = 0.70
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


def _load_categories(conn: Any, statuses: tuple[str, ...],
                     project_id: str | None = None) -> list[dict]:
    """Kategorien aus der flachen `categories`-Tabelle (v19, kein codebook_id mehr).
    project_id: shared base (NULL) ∪ die eigenen induzierten Kategorien des aktiven Projekts."""
    placeholders = ",".join("?" for _ in statuses)
    if project_id:
        scope, extra = "AND (project_id IS NULL OR project_id = ?)", (project_id,)
    else:
        scope, extra = "AND project_id IS NULL", ()
    rows = conn.execute(
        "SELECT id, name, igio_axis, parent_id, embedding_id, status, evidence_count "
        f"FROM categories WHERE status IN ({placeholders}) "
        f"{scope} ORDER BY id",
        (*statuses, *extra),
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
        "JOIN categories cat ON cat.id = cc.category_id "
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
    """Säubert die LLM-Antwort zu EINEM Kategorie-Label. Wenn das Modell statt eines bloßen
    snake_case-Labels JSON geleakt hat (Objekt/Array — passiert ohne JSON-Mode bzw. wenn der
    Batch-Parse aufs per-item-Fallback fällt), erst die VALUES extrahieren; bleibt es JSON-ish,
    verwerfen (→ ''), statt einen kaputten Namen wie {"x":""} als Kategorie zu speichern
    (Root-Cause der JSON-Namen-Junk in Prod). '' → Caller überspringt/fail-closed."""
    line = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    line = line.strip().strip("\"'`").strip()
    if any(ch in line for ch in '{}[]"'):
        # Versuch, das eigentliche Label aus geleaktem JSON zu bergen (erster String-Value),
        # sonst verwerfen — nie den rohen JSON-Blob als Namen behalten.
        import json
        import re
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                vals = [v for v in obj.values() if isinstance(v, str) and v.strip()]
                line = vals[0] if vals else (next(iter(obj), "") if obj else "")
            elif isinstance(obj, list) and obj:
                line = str(obj[0])
            else:
                line = ""
        except (json.JSONDecodeError, TypeError):
            m = re.search(r'"\s*:\s*"([^"]+)"', line) or re.search(r'([A-Za-z][\w-]{2,})', line)
            line = m.group(1) if m else ""
        line = line.strip().strip("\"'`").strip()
        if any(ch in line for ch in '{}[]"'):
            return ""
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
        "SELECT id, name, igio_axis, parent_id, embedding_id FROM categories "
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
        "SELECT id, name, igio_axis, parent_id, embedding_id FROM categories "
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


def _granularity_hint(example_categories: list[str] | None) -> str:
    """Kalibriert die Generalisierung auf das Abstraktionsniveau der BESTEHENDEN Kategorien
    (Beispiele), damit das Label breit & wiederverwendbar wird statt hyper-spezifisch — so
    matcht der nachgelagerte cosine-Schritt öfter eine vorhandene Kategorie (mehr Merging).
    Zeigt nur das GRANULARITÄTSNIVEAU, KEINE Anweisung daraus zu wählen (die Zuordnung macht
    weiterhin der cosine-Schritt, nicht das LLM)."""
    if not example_categories:
        return ""
    sample = ", ".join(example_categories[:25])
    return (
        f"\nBestehende Kategorien haben dieses GRANULARITÄTSNIVEAU (Beispiele): {sample}.\n"
        "Bilde dein Label auf GENAU DIESEM Abstraktionsniveau — breit & wiederverwendbar, "
        "NICHT hyper-spezifisch. (Nicht aus der Liste wählen; nur die Granularität treffen.)\n"
    )


def _reduce_head(text: str, task: str,
                 example_categories: list[str] | None = None) -> str:
    """Der gemeinsame Reduktions-Kopf der EINEN Methode: Ziel-Bezug + die drei
    Mayring-Schritte (Paraphrase→Generalisierung→Reduktion) + Granularitäts-
    Kalibrierung. KEIN Modus — die Methode ist IMMER mixed (deduktiv treffen,
    sonst induktiv neu); welche Hälfte greift, entscheidet der cosine-Schritt
    im Code (_assign_or_create), nicht das LLM. Siehe [[feedback-mayring-canonical-method]]."""
    return (
        "Du bildest eine Kategorie nach qualitativer Inhaltsanalyse (Mayring).\n"
        f"ZIEL/Aufgabe (obligatorischer Bezug): {task[:300]}\n"
        f"Textstelle:\n{text[:1200]}\n\n"
        "Gehe in drei Schritten vor:\n"
        "1. PARAPHRASE: gib den inhaltstragenden Kern der Stelle knapp wieder.\n"
        "2. GENERALISIERUNG: hebe ihn auf das Abstraktionsniveau bestehender Kategorien "
        "(breiter Konzept-Typ, nicht der Einzelfall).\n"
        "3. REDUKTION: verdichte zu EINER prägnanten Kategorie (snake_case, max 4 Wörter), "
        "die den Bezug zum Ziel wahrt.\n"
        f"{_granularity_hint(example_categories)}"
    )


def reduce_prompt(text: str, task: str,
                  example_categories: list[str] | None = None) -> str:
    """Reduktion → bare Label (für den Batch-/Einzelpfad mayring_process/categorize_chunks,
    die das Label selbst embedden + via _assign_or_create mixed zuordnen)."""
    return _reduce_head(text, task, example_categories) + (
        "Antworte NUR mit dem finalen Kategorie-Label (snake_case), nichts anderes.")


# WHY(v19-drop-codebook): codebooks-Tabelle weg, auto_promote_threshold war IMMER 3
# (kein Caller hat jemals ein abweichendes Codebook mit anderem Threshold angelegt).
_AUTO_PROMOTE_THRESHOLD = 3


def _bump_evidence(conn: Any, cat_id: int) -> int:
    """Increment evidence_count and return the NEW value (for promotion check)."""
    conn.execute("UPDATE categories SET evidence_count = evidence_count + 1 "
                 "WHERE id=?", (cat_id,))
    row = conn.execute("SELECT evidence_count FROM categories WHERE id=?",
                       (cat_id,)).fetchone()
    return int(row[0]) if row else 0


def _maybe_promote(conn: Any, cat_id: int, status: str, evidence_count: int,
                   threshold: int) -> bool:
    """proposed→active sobald genug Evidenz: erst dann sieht der Reranker (cat_match,
    active-only) die induktiv gebildete Kategorie. Ohne diesen Schritt bleibt induktiver
    Output inert (Defekt vor canonical-Konsolidierung)."""
    if status != "active" and evidence_count >= threshold:
        conn.execute("UPDATE categories SET status='active', promoted_at=? WHERE id=?",
                     (_now_iso(), cat_id))
        conn.execute("UPDATE codebook_proposals SET decision='promote', reviewed_by='auto' "
                     "WHERE category_id=? AND decision IS NULL", (cat_id,))
        return True
    return False


def _assign_or_create(
    conn: Any, chroma_categories: Any, chunk_id: str | None,
    candidate_label: str, candidate_emb: list[float],
    active_pairs: list[tuple[dict, list[float]]],
    dedup_pairs: list[tuple[dict, list[float]]], *,
    task: str, codebook_version: int, project_id: str | None,
    promote_threshold: int, pi_job_id: str = "",
) -> ProcessResult:
    """Die geteilte Entscheidung beider Pfade (Einzel + Bulk) — EINE Logik, kein Duplikat.
    Erwartet die ABGELEITETE Kandidat-Kategorie (Label + Embedding) und die geladenen
    Kategorie-Mengen: `active_pairs` für den deduktiven Match, `dedup_pairs` (active+proposed)
    fürs induktive Dedup. `dedup_pairs` wird bei einer Neu-Kategorie IN-PLACE erweitert →
    nachfolgende Chunks desselben Batches dedupen darauf (intra-batch, ohne Reload).
    Schritte 3-5 des Ablaufs. (v19: kein target_codebook_id mehr — eine flache categories-Tabelle.)"""
    # 3+4 deduktiv: abgeleitete Kategorie matcht eine vorhandene >= 0.70 → vorhandene nutzen
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

    # echte Neu-Kategorie (ziel-gebunden gebildet) als 'proposed'; Embedding hinterlegen,
    # damit sie ab Promotion für cat_match matchbar ist + Folge-Runden dedupen.
    from src.api.routes.codebooks import record_proposal  # late: web import cycle
    parent_hint_id = top_cat["id"] if top_cat is not None else None
    igio = _infer_igio_axis(candidate_label)
    # WHY(v19-drop-codebook): record_proposal-Signatur OHNE codebook_id (Controller passt
    # die Definition in MayringCoder/src/api/routes/codebooks.py an).
    cat_id = record_proposal(conn, candidate_label, paraphrase=task[:200],
                             parent_hint_id=parent_hint_id, igio_axis=igio,
                             pi_job_id=pi_job_id, chunk_id=chunk_id, project_id=project_id)
    emb_id = f"cb:proposed:{cat_id}"
    if chroma_categories is not None and candidate_emb:
        chroma_categories.upsert(ids=[emb_id], embeddings=[candidate_emb],
                                 documents=[candidate_label])
        conn.execute("UPDATE categories SET embedding_id=? WHERE id=?",
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
    text: str, task: str, *,
    conn: Any, chroma_categories: Any, embed_fn: EmbedFn, llm_fn: LlmFn,
    chunk_id: str | None = None, pi_job_id: str = "", codebook_version: int = 1,
    active_project_id: str | None = None,
) -> ProcessResult:
    """Die EINE Mayring-Methode (Einzel-Pfad: interaktiv + Advisor). IMMER mixed,
    IMMER ziel-gebunden, domänenunabhängig. Ablauf: Ziel(=task) → REDUKTION ZUERST
    (Paraphrase→Generalisierung→Reduktion, kalibriert aufs Granularitätsniveau der
    vorhandenen Kategorien) → cosine 0.70 gegen ALLE vorhandenen → Treffer=deduktiv, sonst
    induktiv neu im EINEN Codebook. Kein codebook_id mehr (es gibt nur eins).
    Raises ValueError (→ HTTP 400) bei leerem text/task. Siehe [[feedback-mayring-canonical-method]]."""
    if not (text or "").strip():
        raise ValueError("mayring_process: 'text' required (fail-closed)")
    if not (task or "").strip():
        raise ValueError("mayring_process: 'task' required — Zielbezug ist obligatorisch")

    # active EINMAL laden — Granularitäts-Beispiele für die Reduktion + Match-Menge.
    active = _load_categories(conn, ("active",), active_project_id)
    # Schritt 1+2: aus Text GEBUNDEN AN DAS ZIEL die Kandidat-Kategorie ableiten.
    candidate = _clean_label(llm_fn(reduce_prompt(text, task, [c["name"] for c in active])))
    if not candidate:
        raise ValueError("mayring_process: Reduktion lieferte kein Label (fail-closed)")
    candidate_emb = embed_fn(candidate)
    if not candidate_emb:
        raise ValueError("mayring_process: embedding der Kategorie fehlgeschlagen (fail-closed)")

    active_pairs = _category_embeddings(chroma_categories, active)
    dedup_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, ("active", "proposed"), active_project_id))
    try:
        res = _assign_or_create(
            conn, chroma_categories, chunk_id, candidate, candidate_emb,
            active_pairs, dedup_pairs, task=task, codebook_version=codebook_version,
            project_id=active_project_id, promote_threshold=_AUTO_PROMOTE_THRESHOLD,
            pi_job_id=pi_job_id)
        conn.commit()
    except Exception:
        conn.rollback()  # offene Transaktion NIE leaken → kein persistenter DB-Lock
        raise
    return res


def categorize_chunks(
    items: list[tuple[str, str]], task: str, *,
    conn: Any, chroma_categories: Any,
    batch_embed_fn: BatchEmbedFn, batch_reduce_fn: BatchReduceFn,
    codebook_version: int = 1, project_id: str | None = None,
) -> list[ProcessResult]:
    """Bulk-Pfad der EINEN Methode: dieselbe `_assign_or_create`-Logik wie mayring_process,
    aber die teure Reduktion + das Kandidaten-Embedding laufen GEBATCHT (eine LLM-Reduktion
    + ein Embed-Call pro Batch, über die zentrale PiQueue/cloud-split) statt pro Chunk.
    Der deduktive Match ist CROSS-codebook — das Embedding routet den Chunk selbst, KEIN
    source_type→codebook-Routing. Echte Neu-Kategorien landen im EINEN Codebook
    (_the_codebook_id). items = [(chunk_id, text)]."""
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

    # active (deduktiver Match) + dedup (active+proposed) EINMAL pro Batch laden; neue
    # Kategorien werden von _assign_or_create in-place angehängt → intra-batch sichtbar.
    active_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, ("active",), project_id))
    dedup_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, ("active", "proposed"), project_id))
    out: list[ProcessResult] = []
    # Per-Chunk committen: hält den SQLite-Write-Lock NICHT über den ganzen Batch + alle
    # Chroma-Upserts; rollback bei Fehler, damit NIE eine offene Transaktion leakt (sonst
    # persistenter "database is locked" für alle anderen Writer — Incident 2026-05-29).
    try:
        for (chunk_id, _text), candidate, emb in zip(items, candidates, embs):
            if not candidate or not emb:
                continue
            out.append(_assign_or_create(
                conn, chroma_categories, chunk_id, candidate, list(emb),
                active_pairs, dedup_pairs, task=task, codebook_version=codebook_version,
                project_id=project_id, promote_threshold=_AUTO_PROMOTE_THRESHOLD))
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    return out


@dataclass
class Candidate:
    label: str
    match: str          # deductive | dedup | inductive
    score: float
    category_id: int | None = None


@dataclass
class ReduceResult:
    paraphrase: str
    generalization: str
    candidates: list[Candidate]


_DECISION_TO_MATCH = {
    "deductive": "deductive",
    "inductive-dedup": "dedup",
    "inductive": "inductive",
}


def _parse_structured(raw: str) -> tuple[str, str, str]:
    """JSON {paraphrase,generalization,label} → (paraphrase, generalization, clean label).
    Robust gegen reale LLM-Outputs: <think>…</think> (thinking-Modelle), Markdown-Fences
    (```json …```) und Prosa um das Objekt. WHY(2026-06-05): ohne Fence-Stripping schlug
    json.loads fehl → _clean_label griff "json" aus der ```json-Zeile → jede Kategorie
    wurde "json" (pi_categorize in Prod kaputt). Fail-soft erst wenn KEIN {…} parsebar."""
    import re as _re
    s = _re.sub(r"<think>.*?</think>", "", raw or "", flags=_re.S).strip()
    candidates = [s]
    fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=_re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    obj_match = _re.search(r"\{.*\}", s, flags=_re.S)   # erstes/größtes JSON-Objekt
    if obj_match:
        candidates.append(obj_match.group(0))
    for cand in candidates:
        try:
            obj = _json_mod.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and str(obj.get("label", "")).strip():
            return (str(obj.get("paraphrase", "")).strip(),
                    str(obj.get("generalization", "")).strip(),
                    _clean_label(str(obj.get("label", ""))))
    return "", "", _clean_label(s)


def _structured_reduce_prompt(text: str, theme: str,
                              example_categories: list[str] | None) -> str:
    """Reduktion → JSON {paraphrase, generalization, label}. Dieselben drei Mayring-
    Schritte wie reduce_prompt, nur dass Paraphrase + Generalisierung mit zurückkommen
    (sie SIND die Methode), nicht nur das Endlabel."""
    return _reduce_head(text, theme, example_categories) + (
        'Antworte NUR mit JSON: {"paraphrase":"...","generalization":"...",'
        '"label":"<snake_case>"} — kein Markdown, keine Prosa.')


def mayring_reduce(
    text: str, theme: str, *,
    conn: Any, chroma_categories: Any, embed_fn: EmbedFn, llm_fn: LlmFn,
    chunk_id: str | None = None, codebook_version: int = 1,
    project_id: str | None = None,
) -> ReduceResult:
    """Die EINE Mayring-Methode (mixed, immer, domänenunabhängig): Ziel → Paraphrase →
    Generalisierung → Reduktion → cosine gegen ALLE aktiven Kategorien. Treffer >=0.70:
    bestehende nutzen (deduktive Hälfte); sonst Dedup >0.92, sonst neu bilden (induktive
    Hälfte) — die Entscheidung trifft der Code (_assign_or_create), nicht das LLM. Es gibt
    KEINEN Modus (deduktiv/induktiv = zwei Hälften derselben Methode) und KEIN codebook_id:
    es gibt nur DAS eine Codebook (_the_codebook_id), neue Kategorien landen dort.
    Liefert immer paraphrase + generalization + candidates[0] (die EINE Kategorie).
    Raises ValueError bei leerem text/theme bzw. fehlgeschlagener Reduktion (fail-closed).
    Bulk-Verlinkung OHNE LLM ist NICHT diese Funktion — das ist link_chunks_deductive
    (hängt Chunks an bereits gebildete Kategorien, erstellt keine; #330). Siehe
    [[feedback-mayring-canonical-method]]."""
    if not (text or "").strip():
        raise ValueError("mayring_reduce: 'text' required (fail-closed)")
    if not (theme or "").strip():
        raise ValueError("mayring_reduce: 'theme' required — Zielbezug obligatorisch")

    active = _load_categories(conn, ("active",), project_id)
    paraphrase, generalization, candidate = _parse_structured(
        llm_fn(_structured_reduce_prompt(text, theme, [c["name"] for c in active])))
    if not candidate:
        raise ValueError("mayring_reduce: Reduktion lieferte kein Label (fail-closed)")
    candidate_emb = embed_fn(candidate)
    if not candidate_emb:
        raise ValueError("mayring_reduce: embedding fehlgeschlagen (fail-closed)")

    active_pairs = _category_embeddings(chroma_categories, active)
    dedup_pairs = _category_embeddings(
        chroma_categories, _load_categories(conn, ("active", "proposed"), project_id))
    try:
        res = _assign_or_create(
            conn, chroma_categories, chunk_id, candidate, candidate_emb,
            active_pairs, dedup_pairs, task=theme, codebook_version=codebook_version,
            project_id=project_id, promote_threshold=_AUTO_PROMOTE_THRESHOLD)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return ReduceResult(paraphrase, generalization, [Candidate(
        res.category_name or candidate,
        _DECISION_TO_MATCH.get(res.decision, "inductive"),
        res.confidence, res.category_id)])
