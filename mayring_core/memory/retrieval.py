"""Memory Retrieval Policy — 4-stage hybrid search.

Stage 1: SQLite scope filter (repo, category, source_type, active)
Stage 2: Symbolic matching (token overlap + path bonus)
Stage 3: Vector retrieval (ChromaDB similarity)
Stage 4: Re-ranking (weighted combination)
Stage 5: compress_for_prompt() — format results for Claude
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.schema import Chunk, RetrievalRecord
from mayring_core.memory.store import get_chunk, get_chunks_bulk, kv_get, get_feedback_score

# ---------------------------------------------------------------------------
# Query-Cache (in-process, invalidated on any memory mutation)
# ---------------------------------------------------------------------------

_QUERY_CACHE: dict[str, list["RetrievalRecord"]] = {}  # cache_key → ranked records (text stripped)


def _cache_key(query: str, opts: dict, session_compacted: bool) -> str:
    """Stable hash key for query + retrieval-relevant opts."""
    relevant = {k: v for k, v in opts.items() if k != "include_text"}
    payload = _json.dumps({"q": query, "opts": relevant, "sc": session_compacted}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def invalidate_query_cache() -> None:
    """Clear the entire query cache. Call after any memory mutation (put/invalidate/reindex)."""
    _QUERY_CACHE.clear()

try:
    from mayring_core.providers import embed_texts as _embed_texts
    _HAS_EMBED = True
except ImportError:
    _HAS_EMBED = False

_WEIGHTS = {
    "vector":          0.30,
    "symbolic":        0.20,
    "recency":         0.10,
    "source_affinity": 0.10,
    "feedback":        0.15,   # additive ±0.15, plus asymmetric -0.10 penalty when sf<0.3
    "llm_advisor":     0.15,   # PI-advisor semantic relevance (0.5 = neutral when missing)
}

_RECENCY_DECAY_DAYS = 30.0


# ---------------------------------------------------------------------------
# Stage 1: Scope filter
# ---------------------------------------------------------------------------

def _scope_filter(
    conn: DBAdapter,
    repo: str | None = None,
    categories: list[str] | None = None,
    source_type: str | None = None,
    workspace_id: str | None = None,
    org_id: str | None = None,
    org_ids: tuple[str, ...] | list[str] | None = None,
    user_id: str | None = None,
    scope_key: str | None = None,
    project_id: str | None = None,
) -> list[str]:
    """Return chunk_ids of active chunks matching hard scope filters.

    Visibility model — a chunk is visible to the caller when ANY of:
      - visibility='public'                                  (everyone)
      - visibility='org'     AND s.org_id   IN caller_orgs   (any of caller's orgs)
      - visibility='user'    AND s.user_id  = caller_sub     (same human user,
                                                              any workspace —
                                                              claude.ai vs CLI)
      - visibility='private' AND c.workspace_id = caller_ws  (legacy default)

    org_ids supersedes the legacy single org_id param (V2 multi-org JWT).
    Both are accepted: callers passing org_id (singular) are folded into the
    list so every existing call-site keeps working.
    """
    query = """
        SELECT c.chunk_id, c.category_labels
        FROM chunks c
        JOIN sources s ON c.source_id = s.source_id
        WHERE c.is_active = 1
    """
    params: list = []

    # Normalise org input: org_ids (V2) wins; otherwise fold legacy org_id.
    effective_org_ids: tuple[str, ...]
    if org_ids:
        effective_org_ids = tuple(o for o in org_ids if o)
    elif org_id:
        effective_org_ids = (org_id,)
    else:
        effective_org_ids = ()

    if workspace_id:
        if effective_org_ids:
            placeholders = ",".join("?" * len(effective_org_ids))
            org_clause = f" OR (s.visibility = 'org' AND s.org_id IN ({placeholders}))"
        else:
            org_clause = ""
        query += (
            " AND (s.visibility = 'public'"
            " OR (s.visibility = 'user' AND s.user_id = ?)"
            f"{org_clause}"
            " OR (s.visibility = 'private' AND c.workspace_id = ?))"
        )
        params.append(user_id)
        params.extend(effective_org_ids)
        params.append(workspace_id)
    if repo:
        query += " AND s.repo = ?"
        params.append(repo)
    if source_type:
        query += " AND s.source_type = ?"
        params.append(source_type)
    if scope_key:
        # #252: restrict to one logical sub-bucket of the workspace (e.g.
        # "project:<id>") — a Recherche search only sees that project's
        # papers, never another project's chunks in the same workspace.
        query += " AND s.scope_key = ?"
        params.append(scope_key)
    if project_id:
        # #workspace-uuid-sot (v7): Project-ID-Dimension (User-Diagramm). Scopt
        # die Suche auf EIN Projekt innerhalb des Workspace. NOTE: überlappt
        # konzeptionell mit scope_key="project:<id>" (#252) — Konsolidierung auf
        # EINE Achse (project_id kanonisch, scope_key→migrieren/retire) ist ein
        # eigener Follow-up; nicht im selben Schritt überstürzen.
        query += " AND c.project_id = ?"
        params.append(project_id)

    rows = conn.execute(query, params).fetchall()

    if not categories:
        return [row[0] for row in rows]

    # Category filter: post-query (comma-split stored labels)
    cat_set = {c.lower() for c in categories}
    result = []
    for chunk_id, labels_str in rows:
        chunk_cats = {l.strip().lower() for l in (labels_str or "").split(",") if l.strip()}
        if chunk_cats & cat_set:
            result.append(chunk_id)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Symbolic scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Split text into lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _symbolic_score(chunk: Chunk, query_terms: set[str]) -> float:
    """0.0–1.0 score based on token overlap + path/category bonuses."""
    if not query_terms:
        return 0.0

    chunk_terms = _tokenize(chunk.text) | _tokenize(chunk.summary)
    overlap = len(query_terms & chunk_terms) / len(query_terms)

    # Bonus: query term appears in source path (file relevance)
    path_bonus = 0.0
    source_path = chunk.source_id.lower()
    for term in query_terms:
        if term in source_path:
            path_bonus = 0.2
            break

    # Bonus: query term matches a category label
    cat_bonus = 0.0
    cat_terms = _tokenize(",".join(chunk.category_labels))
    if query_terms & cat_terms:
        cat_bonus = 0.1

    return min(1.0, overlap + path_bonus + cat_bonus)


# ---------------------------------------------------------------------------
# Stage 3 helper: normalize ChromaDB distances to scores
# ---------------------------------------------------------------------------

def _normalize_vector_scores(
    chroma_ids: list[str],
    chroma_distances: list[float],
    candidate_ids: set[str],
) -> dict[str, float]:
    """Convert ChromaDB distances to similarity scores (0.0–1.0), filtered to candidates."""
    scores: dict[str, float] = {}
    for cid, dist in zip(chroma_ids, chroma_distances):
        if cid not in candidate_ids:
            continue
        # ChromaDB cosine distance: score = 1 - distance (clamped)
        scores[cid] = max(0.0, min(1.0, 1.0 - dist))
    return scores


# ---------------------------------------------------------------------------
# IGIO-intent detection — outcome/issue/goal/intervention boost
# ---------------------------------------------------------------------------

# WHY(2026-05-10 IGIO-utilisation): outcome ist die wichtigste axis für
# "was kam dabei raus / wirkung / konsequenz"-fragen, wurde aber im v1-
# rerank ignoriert. Dieser intent-detector ist KEYWORD-basiert (kein LLM)
# damit jede search-call ihn aufrufen kann ohne extra-latenz. Treffer-
# patterns kalibriert auf DE+EN, da user häufig deutsch fragt.
_IGIO_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "outcome": (
        # de
        "ergebnis", "ergebnisse", "konsequenz", "konsequenzen", "wirkung",
        "auswirkung", "auswirkungen", "resultat", "resultate", "was kam dabei raus",
        "was kam raus", "was passierte", "was hat sich geändert", "fazit",
        "outcome", "outcomes",
        # en
        "result", "results", "consequence", "consequences", "effect", "effects",
        "what happened", "what changed", "impact",
    ),
    "issue": (
        # de
        "problem", "probleme", "bug", "fehler", "issue", "warum", "wieso",
        "weshalb", "ursache", "ursachen",
        # en
        "issues", "bugs", "error", "errors", "why does", "why is",
        "root cause", "broken",
    ),
    "goal": (
        # de
        "ziel", "ziele", "vorhaben", "soll", "möchte", "möchten", "plan",
        "planung", "intention",
        # en
        "goal", "goals", "objective", "objectives", "target", "we want",
        "intend", "intent", "plan to",
    ),
    "intervention": (
        # de
        "wie umsetzen", "wie implementieren", "wie löse", "wie löst",
        "wie fixe", "wie repariere", "wie baue", "schritte", "anleitung",
        "how to fix", "how to implement",
        # en
        "implementation", "implement", "how do i", "how can i", "fix it",
        "steps", "approach",
    ),
}


def detect_igio_intent(query: str) -> str | None:
    """Erkenne welche IGIO-axis die query primär anspricht.

    Returns "outcome" | "issue" | "goal" | "intervention" oder None wenn
    kein klares signal. Bei mehrfach-treffern wins die axis mit der
    höchsten match-zahl. Bei gleichstand wins outcome > issue > goal >
    intervention (ranking nach informationsgehalt — outcome ist am
    spezifischsten).
    """
    if not query:
        return None
    q = query.lower()
    counts: dict[str, int] = {}
    for axis, patterns in _IGIO_INTENT_PATTERNS.items():
        n = sum(1 for p in patterns if p in q)
        if n > 0:
            counts[axis] = n
    if not counts:
        return None
    # Wins-priority bei tie: outcome > issue > goal > intervention
    priority = ("outcome", "issue", "goal", "intervention")
    max_count = max(counts.values())
    for axis in priority:
        if counts.get(axis, 0) == max_count:
            return axis
    return None


# ---------------------------------------------------------------------------
# Stage 3b: Optional LLM pre-filter
# ---------------------------------------------------------------------------

def _llm_relevance_scores(
    query: str,
    candidates: list[Chunk],
    ollama_url: str,
    model: str = "qwen2.5-coder:7b",
    timeout: float = 10.0,
    task_context: str | None = None,
) -> dict[str, float]:
    """Best-effort: LLM rates relevance of each candidate for query.
    Returns chunk_id → [0.0, 1.0]. Falls back to 0.5 on any error.

    If task_context is provided (e.g. plan/todo content), the PI-advisor sees
    the broader task description and can judge relevance more accurately than
    from the query alone.
    """
    from mayring_core.ollama_client import generate as _oll_gen
    scores: dict[str, float] = {}
    task_block = (
        f"Task context (what the user is working on):\n{task_context[:600]}\n\n"
        if task_context else ""
    )
    for chunk in candidates[:20]:
        prompt = (
            f"{task_block}"
            f"Query: {query}\n\n"
            f"Memory chunk: {chunk.text[:300]}\n\n"
            "How relevant is this chunk for the task above? "
            "Rate 0.0–1.0. Answer ONLY with the number."
        )
        try:
            raw = _oll_gen(
                ollama_url, model, prompt,
                stream=False, timeout=timeout, num_predict=8,
            ).strip()
            scores[chunk.chunk_id] = max(0.0, min(1.0, float(raw)))
        except (ConnectionError, TimeoutError, OSError, ValueError):
            # WHY(v2-stufe2.1): LLM-Advisor ist Best-Effort — fallback 0.5
            # bedeutet "keine starke Aussage". Konkrete Fehlerklassen
            # benannt; alles andere (z.B. AssertionError aus mock) muss laut.
            scores[chunk.chunk_id] = 0.5
    return scores


# ---------------------------------------------------------------------------
# Stage 4: Recency and affinity
# ---------------------------------------------------------------------------

def _recency_score(chunk: Chunk) -> float:
    """Decay: 1.0 for brand-new chunks, 0.0 for chunks >30 days old."""
    # WHY(v2-stufe2.1): nur konkrete Parser-Fehler schlucken — sonst
    # legacy created_at-Format würde alle Chunks scoren mit 0.0 und
    # fail wäre unsichtbar.
    try:
        created = datetime.fromisoformat(chunk.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_old = (now - created).total_seconds() / 86400.0
        return max(0.0, 1.0 - days_old / _RECENCY_DECAY_DAYS)
    except (ValueError, TypeError):
        return 0.0


def _source_affinity_score(chunk: Chunk, affinity_source_id: str | None) -> float:
    """1.0 if chunk belongs to the affinity source, else 0.0."""
    if affinity_source_id and chunk.source_id == affinity_source_id:
        return 1.0
    return 0.0


def _has_table(conn: DBAdapter, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _category_id_match(query_cat_ids: set[int], chunk_cat_ids: set[int]) -> float:
    """Reranker-v3 (#270): 1.0 wenn ein Kandidat über chunk_categories an eine
    Kategorie geknüpft ist, die auch zur Query-Kategorie gehört — strukturierter
    category_id-Abgleich statt fragilem Free-String-Match auf category_labels."""
    return 1.0 if (query_cat_ids and chunk_cat_ids and (query_cat_ids & chunk_cat_ids)) else 0.0


# ---------------------------------------------------------------------------
# Stage 4: Re-ranking
# ---------------------------------------------------------------------------

def _rerank(
    candidates: list[Chunk],
    vector_scores: dict[str, float],
    symbolic_scores: dict[str, float],
    top_k: int,
    conn: DBAdapter,
    affinity_source_id: str | None = None,
    source_type_map: dict[str, str] | None = None,
    session_compacted: bool = False,
    llm_scores: dict[str, float] | None = None,
    reranker_query_hint: str | None = None,
    reranker_override: str | None = None,
    predicted_topics: set[str] | None = None,
    category_hint: list[str] | None = None,
    igio_intent: str | None = None,
    task_id: str | None = None,
) -> list[RetrievalRecord]:
    """Combine scores and return top_k RetrievalRecords sorted by score_final DESC.

    llm_scores: optional PI-advisor relevance ratings (chunk_id → [0,1]). Missing
    chunks default to 0.5 (neutral) so the LLM weight contributes equally and
    doesn't bias the ranking when scores are unavailable.

    reranker_query_hint / reranker_override: control which weighting
    formula is used per call.
      * v1 (default): the hand-tuned ``_WEIGHTS`` dict + asymmetric
        feedback penalty + compaction boost.
      * v2: learned weights from ``cache/rerank_v2.json``. Computes the
        v1 score first (so v2 can use ``f=score_v1`` as a feature), then
        replaces the final score with the sigmoid output of the linear
        model. Falls back silently to v1 if no model file exists.

    See src/memory/reranker_v2.py for selection precedence.
    """
    from mayring_core.memory.reranker_v2 import get_active_reranker, score_v2
    records: list[RetrievalRecord] = []
    llm_scores = llm_scores or {}
    rr_version, rr_model = get_active_reranker(
        query_hint=reranker_query_hint, explicit_override=reranker_override,
    )

    # WHY(2026-05-11, task-categorization): wenn der prompt zu einer
    # bekannten task_category gehört, lookup chunks die bei dieser
    # task schon positive feedback hatten → boost im rerank.
    # Verstärkt Mayring-konformes retrieval ("was hat MIR bei DIESER
    # frage geholfen") ggü reinem tech-label-clustering.
    task_boost_map: dict[str, float] = {}
    if task_id and candidates:
        try:
            from mayring_core.memory.task_derivation import get_research_question_boost
            task_boost_map = get_research_question_boost(
                conn, task_id, [c.chunk_id for c in candidates],
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("task-boost lookup failed: %s", e)

    # Stretch vector scores so the best Chroma hit in *this* query gets the
    # full vector weight. Cosine distances from nomic-embed-text typically
    # land in [0.5, 1.0] → raw scores in [0, 0.5], i.e. half the weight even
    # for the best possible match. Without stretching, symbolic (which easily
    # hits 1.0 on token overlap) wins every ranking decision and the vector
    # pipeline doesn't actually influence top_k. Per-query stretch is
    # scale-free across embedding models / domains and doesn't penalise
    # queries with weak corpus support — it just stops the asymmetry from
    # silently disabling vector search.
    if vector_scores:
        _max_v = max(vector_scores.values())
        if _max_v > 0:
            stretched_vec = {cid: v / _max_v for cid, v in vector_scores.items()}
        else:
            stretched_vec = vector_scores
    else:
        stretched_vec = {}

    predicted_topics = predicted_topics or set()
    # Predicted-topic-Boost (Issue #184): chunk.category_labels OR chunk.text
    # gegen die Markov-Chain-Vorhersagen abgleichen. Bewusst kleines Gewicht
    # (0.05), damit es Recall erweitert ohne den Mainstream-Score zu kapern.
    _PRED_BOOST = 0.05

    # WHY(2026-05-10 multi-cat prompt-decomp): wenn der hook prompt-
    # kategorien mitsendet ("auth + caching + deployment"), boosten wir
    # chunks deren category_labels mit DIESEN kategorien überlappen.
    # Bewusst klein (0.08) — bias soll recall verbessern, nicht ein
    # einzelnes label dominieren. Normalisieren auf set für O(1)-overlap.
    _CAT_HINT_BOOST = 0.08
    cat_hint_set: set[str] = {
        c.lower() for c in (category_hint or []) if c
    }

    # WHY(2026-05-10 IGIO-intent-boost): wir klassifizieren chunks auf
    # issue/goal/intervention/outcome-axis (igio_classifier.py), nutzen die
    # axis aber im aktiven v1-rerank GAR NICHT (nur disabled v2 hat sie
    # als one-hot-feature). Konsequenz: wenn user nach "konsequenzen",
    # "ergebnissen", "was kam dabei raus" fragt, werden outcome-chunks
    # NICHT priorisiert — outcome ist die wichtigste axis für
    # nach-fakten-fragen, geht aber im scoring unter. Fix: chunks die zu
    # der detected intent-axis matchen kriegen +0.10 boost.
    _IGIO_INTENT_BOOST = 0.10
    igio_intent_axis = (igio_intent or "").lower().strip()

    # Reranker-v3 (#270): strukturierter category_id-Match. Die category_hint-Strings
    # → category_ids auflösen, dann gegen die chunk_categories-FKs der Kandidaten
    # matchen. Kleiner deterministischer Boost im v1-Pfad (sofort + sicher, kein
    # Modell-Risiko) PLUS als 'cat_match'-Feature für die gelernte v2-Reranking
    # geloggt. Coverage wächst mit chunk_categories (deduktiver Link-Pass).
    _CAT_MATCH_BOOST = 0.08
    query_cat_ids: set[int] = set()
    chunk_cat_map: dict[str, set[int]] = {}
    # Feature ist optional: ohne die Codebook-Tabellen (Minimal-/Alt-DB) bleibt es
    # inaktiv (explizite Precondition, kein stilles except).
    if (cat_hint_set and candidates
            and _has_table(conn, "codebook_categories")
            and _has_table(conn, "chunk_categories")):
        ph = ",".join("?" for _ in cat_hint_set)
        query_cat_ids = {
            r[0] for r in conn.execute(
                f"SELECT id FROM codebook_categories WHERE lower(name) IN ({ph})",
                tuple(cat_hint_set)).fetchall()
        }
        if query_cat_ids:
            ids = [c.chunk_id for c in candidates]
            qp = ",".join("?" for _ in ids)
            for cid, cat_id in conn.execute(
                    f"SELECT chunk_id, category_id FROM chunk_categories "
                    f"WHERE chunk_id IN ({qp})", tuple(ids)).fetchall():
                chunk_cat_map.setdefault(cid, set()).add(cat_id)

    for chunk in candidates:
        sv_raw = vector_scores.get(chunk.chunk_id, 0.0)         # for the record
        sv_eff = stretched_vec.get(chunk.chunk_id, 0.0)         # for ranking
        ss = symbolic_scores.get(chunk.chunk_id, 0.0)
        sr = _recency_score(chunk)
        sa = _source_affinity_score(chunk, affinity_source_id)
        sf = get_feedback_score(conn, chunk.chunk_id)
        sl = llm_scores.get(chunk.chunk_id, 0.5)  # neutral default

        # score_predicted_topic: 1.0 wenn ANY chunk-tag (category_label oder
        # ein lowercased Wort im Text der ersten 200 Zeichen) zu einem
        # vorhergesagten Folgethema matcht. Lowercase auf beiden Seiten
        # damit "MCP" vs "mcp" matcht.
        sp = 0.0
        if predicted_topics and (chunk.category_labels or chunk.text):
            chunk_tags = {lbl.lower() for lbl in (chunk.category_labels or [])}
            text_words = set((chunk.text or "")[:200].lower().split())
            if (predicted_topics & chunk_tags) or (predicted_topics & text_words):
                sp = 1.0

        # Category-hint overlap: 1.0 wenn ANY chunk-label im hint-set,
        # sonst 0.0. Auf labels-side bereits lowercase normalize.
        sc = 0.0
        if cat_hint_set and chunk.category_labels:
            chunk_cats = {lbl.lower().strip() for lbl in chunk.category_labels}
            if cat_hint_set & chunk_cats:
                sc = 1.0

        # IGIO-intent boost: 1.0 wenn chunk.igio_axis zur intent passt.
        # Beispiel: query="was kam dabei raus" → intent="outcome",
        # chunk mit igio_axis="outcome" bekommt +0.10.
        si = 0.0
        if igio_intent_axis and (chunk.igio_axis or "").lower() == igio_intent_axis:
            si = 1.0

        # Reranker-v3 structured category_id match (s. Setup oben).
        scat_id = _category_id_match(query_cat_ids, chunk_cat_map.get(chunk.chunk_id, set()))

        # WHY(2026-05-11 user-feedback): "Reasons sollten HOCH gewertet
        # werden, wenn sie in der suche auftauchen — hier sind ZUSAMMEN-
        # HÄNGE beschrieben". Konvergenz-bonus: wenn ≥2 unabhängige
        # signale gleichzeitig feuern (vector + symbolic, oder
        # symbolic + recency + affinity etc.), ist die confidence höher
        # als die summe der einzelnen scores suggeriert.
        signal_count = (
            (1 if sv_eff > 0.5 else 0)
            + (1 if ss > 0.3 else 0)
            + (1 if sr > 0.8 else 0)
            + (1 if sa > 0 else 0)
            + (1 if sl > 0.7 else 0)
        )
        convergence_bonus = 0.0
        if signal_count >= 3:
            convergence_bonus = 0.12   # 3+ signale = sehr stark
        elif signal_count >= 2:
            convergence_bonus = 0.06   # 2 signale = klarer zusammenhang

        score_v1 = (
            _WEIGHTS["vector"] * sv_eff
            + _WEIGHTS["symbolic"] * ss
            + _WEIGHTS["recency"] * sr
            + _WEIGHTS["source_affinity"] * sa
            + _WEIGHTS["feedback"] * (sf * 2.0 - 1.0)   # [0,1]→[-1,1] × 0.15 → ±0.15
            + _WEIGHTS["llm_advisor"] * sl
            + _PRED_BOOST * sp
            + _CAT_HINT_BOOST * sc
            + _IGIO_INTENT_BOOST * si
            + _CAT_MATCH_BOOST * scat_id
            + convergence_bonus
        )

        # Asymmetric negative penalty: a chunk users repeatedly mark as
        # irrelevant must drop out of top_k entirely, not just rank lower.
        # Triggers when ≥70% of feedback is negative.
        if sf < 0.3:
            score_v1 -= 0.10

        # Compaction boost: prefer conversation_summary section chunks
        if session_compacted and source_type_map:
            chunk_source_type = source_type_map.get(chunk.source_id, "")
            if chunk_source_type == "conversation_summary" and chunk.chunk_level == "section":
                score_v1 = min(1.0, score_v1 + 0.10)

        if rr_version == "v2" and rr_model is not None:
            # v3-Feature-Set: retrieval stages + IGIO one-hot. sf/sl
            # raus wegen Target-Leakage (siehe Issue #180). Für
            # Backward-Compat zu älteren v2-Modellfiles bleibt das
            # weights-dict 'forgiving' — fehlende keys → weight=0 in
            # score_v2.
            axis = (chunk.igio_axis or "unknown").lower()
            stage = {"v": sv_eff, "s": ss, "r": sr, "a": sa,
                     # legacy keys für alte rerank_v2.json
                     "sf": sf, "sl": sl}
            for a_label in ("issue", "goal", "intervention", "outcome", "unknown"):
                stage[f"igio_{a_label}"] = 1.0 if axis == a_label else 0.0
            stage["cat_match"] = scat_id   # reranker-v3 (#270): gelernt sobald geloggt
            score_final = score_v2(stage, rr_model)
        else:
            score_final = score_v1

        # Task-Boost: chunks die bei semantisch ähnlicher task schon
        # positive feedback hatten kriegen einen kleinen aber spürbaren
        # bonus. Max +0.20 bei relevance_score=5.0 (clipped); 0.04 pro
        # feedback-event. Bewusst klein gehalten — der task-signal soll
        # andere stages nicht dominieren, aber tie-breaker sein.
        task_boost = task_boost_map.get(chunk.chunk_id, 0.0)
        if task_boost > 0:
            score_final = min(1.0, score_final + task_boost * 0.04)

        reasons: list[str] = []
        # Reasons fire on the *normalised* vector score so a query with weak
        # global similarity but a clear in-query winner still surfaces
        # "embedding_similarity" — otherwise this label was effectively dead
        # for nomic-embed-text where raw scores rarely cross 0.5.
        if sv_eff > 0.5:
            reasons.append("embedding_similarity")
        if ss > 0.3:
            reasons.append("token_overlap")
        if sr > 0.8:
            reasons.append("recent_chunk")
        if sa > 0:
            reasons.append("source_affinity_match")
        if sl > 0.7:
            reasons.append("llm_advisor_high")

        records.append(
            RetrievalRecord(
                chunk_id=chunk.chunk_id,
                score_vector=sv_raw,
                score_symbolic=ss,
                score_recency=sr,
                score_source_affinity=sa,
                score_feedback=sf,
                score_llm=sl,
                score_predicted_topic=sp,
                score_final=score_final,
                reasons=reasons,
                source_id=chunk.source_id,
                text=chunk.text,
                summary=chunk.summary,
                category_labels=chunk.category_labels,
            )
        )

    records.sort(key=lambda r: r.score_final, reverse=True)
    ranked = records[:top_k]

    # Issue #185/#182 follow-up: enrich each top-K record with rationale-edges
    # from wiki_v2.db (attached as 'wikidb' in init_memory_db). Join key:
    # chunk's source.path → wiki_edges.source. Silent skip when not attached.
    if getattr(conn, "_wiki_attached", False) and ranked:
        chunk_by_id = {c.chunk_id: c for c in candidates}
        for record in ranked:
            chunk = chunk_by_id.get(record.chunk_id)
            if chunk is None:
                continue
            src_row = conn.execute(
                "SELECT path FROM sources WHERE source_id = ?",
                (chunk.source_id,),
            ).fetchone()
            if not src_row or not src_row[0]:
                continue
            try:
                rows = conn.execute(
                    "SELECT rationale, target, context FROM wikidb.wiki_edges "
                    "WHERE source = ? AND type = 'rationale' "
                    "AND rationale != ''",
                    (src_row[0],),
                ).fetchall()
            except sqlite3.OperationalError:
                # wikidb-schema mismatch oder ATTACH gone — silent skip,
                # rationale_edges bleibt []. Typed catch fängt Tippfehler
                # in der Query lautstark, statt sie zu maskieren.
                continue
            # Named row access via sqlite3.Row (DBAdapter setzt row_factory).
            # Reihenfolge-unabhängig — falls SELECT je um eine Spalte erweitert
            # wird, bleibt das Mapping korrekt.
            record.rationale_edges = [
                {
                    "target": row["target"],
                    "context": row["context"],
                    "why": row["rationale"],
                }
                for row in rows
            ]

    return ranked


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

def search(
    query: str,
    conn: DBAdapter,
    chroma_collection: Any,
    ollama_url: str,
    opts: dict | None = None,
    session_compacted: bool = False,
) -> list[RetrievalRecord]:
    """4-stage hybrid memory search.

    opts:
        repo (str): filter by repo
        categories (list[str]): filter by category labels
        source_type (str): filter by source_type
        top_k (int, default 8): max results
        include_text (bool, default True): populate .text on results
        source_affinity (str): source_id to boost
    """
    opts = opts or {}
    top_k: int = int(opts.get("top_k", 8))
    repo: str | None = opts.get("repo")
    categories: list[str] | None = opts.get("categories")
    source_type: str | None = opts.get("source_type")
    affinity_source_id: str | None = opts.get("source_affinity")
    include_text: bool = bool(opts.get("include_text", True))
    workspace_id: str | None = opts.get("workspace_id")
    org_id: str | None = opts.get("org_id")
    org_ids_opt = opts.get("org_ids")
    org_ids: tuple[str, ...] | None
    if org_ids_opt:
        org_ids = tuple(org_ids_opt)
    else:
        org_ids = None
    user_id: str | None = opts.get("user_id")
    scope_key: str | None = opts.get("scope_key")  # #252: e.g. "project:<id>"
    project_id: str | None = opts.get("project_id")  # #workspace-uuid-sot: Project-Dimension

    # Query-Cache check — hit: clone records, repopulate text on demand.
    # Bug history: cache used to store only (chunk_id, score_final), so
    # cache-hits returned chunks with score_vector=0 and reasons=[]. The
    # smoke check_rag_function_search_finds_source then saw v=0 and the
    # check_retrieval_reasons_field saw no reasons. Fix: cache the whole
    # ranked record (text stripped to keep memory bounded), and on hit
    # re-hydrate text from SQLite only when the caller asked for it.
    _ck = _cache_key(query, opts, session_compacted)
    if _ck in _QUERY_CACHE:
        import dataclasses as _dc
        opts["_vector_diag"] = "query_cache_hit"
        hydrated: list[RetrievalRecord] = []
        for cached in _QUERY_CACHE[_ck]:
            if include_text:
                chunk = get_chunk(conn, cached.chunk_id)
                if not chunk:
                    continue
                hydrated.append(_dc.replace(cached, text=chunk.text))
            else:
                hydrated.append(_dc.replace(cached))
        return hydrated

    # Stage 1: scope filter
    candidate_ids = _scope_filter(
        conn, repo=repo, categories=categories, source_type=source_type,
        workspace_id=workspace_id, org_id=org_id, org_ids=org_ids,
        user_id=user_id, scope_key=scope_key, project_id=project_id,
    )
    if not candidate_ids:
        # Tell callers WHY the result is empty: scope filter excluded
        # everything (workspace mismatch, no public/org/user share, etc.)
        opts["_vector_diag"] = "empty_after_scope_filter"
        return []

    # Load candidate chunks: KV cache first (fast in-mem hits), then ONE bulk
    # SQLite query for the misses. WHY(#workspace-uuid-sot perf): the old per-cid
    # get_chunk() loop did 4000+ single SQLite reads on a whole-workspace candidate
    # set → ~11s/search → 9s hook timeouts. Bulk-load collapses that to one query.
    candidates: list[Chunk] = []
    missing_ids: list[str] = []
    for cid in candidate_ids:
        cached = kv_get(cid)
        if cached is not None:
            try:
                candidates.append(Chunk.from_dict(cached))
                continue
            except (KeyError, TypeError, ValueError):
                pass  # stale KV schema → reload from DB (bulk below)
        missing_ids.append(cid)
    if missing_ids:
        candidates.extend(get_chunks_bulk(conn, missing_ids))

    if not candidates:
        return []

    # Stage 2: symbolic scoring (optionally enriched with task_context tokens)
    task_context: str | None = opts.get("task_context")
    query_terms = _tokenize(query)
    if task_context:
        # Task-context tokens widen recall — used for symbolic overlap only.
        # Each token still counts equally, so this won't dilute precision.
        query_terms = query_terms | _tokenize(task_context)
    symbolic_scores: dict[str, float] = {
        c.chunk_id: _symbolic_score(c, query_terms) for c in candidates
    }

    # Stage 3: vector retrieval
    vector_scores: dict[str, float] = {}
    vector_diag: str = "ok"  # surfaces in opts so callers can see WHY 0.0 happened
    if chroma_collection is None:
        vector_diag = "no_chroma_collection"
        _log.warning("vector stage skipped: chroma_collection is None (query=%r)", query[:60])
    elif not _HAS_EMBED:
        vector_diag = "embed_module_missing"
        _log.warning("vector stage skipped: _embed_texts not importable")
    else:
        try:
            query_emb = _embed_texts([query], ollama_url)[0]
            chroma_count = chroma_collection.count()
            n_results = min(top_k * 2, chroma_count)
            if n_results == 0:
                vector_diag = f"chroma_empty(count={chroma_count})"
                _log.warning("vector stage: chroma collection has 0 entries")
            else:
                chroma_where = (
                    {"workspace_id": {"$eq": workspace_id}} if workspace_id else None
                )
                results = None
                try:
                    results = chroma_collection.query(
                        query_embeddings=[query_emb],
                        n_results=n_results,
                        where=chroma_where,
                        include=["distances"],
                    )
                except Exception as _chroma_exc:
                    vector_diag = f"chroma_query_failed:{type(_chroma_exc).__name__}"
                    _log.warning(
                        "chroma workspace filter failed for workspace=%r (%s) "
                        "— skipping vector stage to prevent cross-workspace data leak",
                        workspace_id, _chroma_exc,
                    )
                if results is None:
                    ids_list, dist_list = [], []
                else:
                    ids_list = results.get("ids", [[]])[0]
                    dist_list = results.get("distances", [[]])[0]
                candidate_set = {c.chunk_id for c in candidates}
                vector_scores = _normalize_vector_scores(ids_list, dist_list, candidate_set)
                max_score = max(vector_scores.values()) if vector_scores else 0.0
                mean_dist = (sum(dist_list) / len(dist_list)) if dist_list else 0.0
                if results is not None and ids_list and not vector_scores:
                    # Hits returned, but none in our candidate set → SQLite/Chroma drift
                    vector_diag = (
                        f"chroma_candidate_mismatch(hits={len(ids_list)}"
                        f",candidates={len(candidate_set)})"
                    )
                    _log.warning(
                        "vector stage: chroma returned %d hits but 0 in candidate set "
                        "(SQLite/Chroma drift; query=%r workspace=%r)",
                        len(ids_list), query[:60], workspace_id,
                    )
                elif not ids_list and chroma_count > 0:
                    vector_diag = "chroma_query_empty"
                elif vector_scores and max_score < 0.05:
                    # Hits exist but all distances near/above 1.0 → embeddings
                    # are not semantically aligned with the query; common when
                    # query and corpus use different vocab or the corpus is too
                    # small for vector search to discriminate.
                    vector_diag = (
                        f"all_distances_too_far(max_score={max_score:.3f}"
                        f",mean_dist={mean_dist:.3f})"
                    )
                else:
                    vector_diag = (
                        f"ok(max_score={max_score:.3f},matches={len(vector_scores)}"
                        f",mean_dist={mean_dist:.3f})"
                    )
        except Exception as exc:
            vector_diag = f"embed_failed:{type(exc).__name__}"
            _log.warning("vector retrieval failed (best-effort skip): %s", exc)
    opts["_vector_diag"] = vector_diag

    # Telemetry: persist vector-stage outcome to llm_calls_log so the
    # dashboard can plot success-rate over time. Reuses the existing
    # table (call_type='vector_search'); one INSERT per search is
    # negligible compared to the embedding/Chroma/SQLite work above. Was
    # behind an env var initially "to avoid write amplification" — but
    # that was overcautious and turned a real feature into dead UI.
    import os as _os
    try:
        from mayring_core.memory.store import log_llm_call
        log_llm_call(
            conn,
            call_type="vector_search",
            model=_os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
            prompt=query[:200],
            response=_json.dumps({"vector_stage": vector_diag}),
            tool_calls=0,
            duration_ms=0,
            workspace_id=workspace_id or "default",
        )
    except Exception:
        pass  # logging must never block retrieval

    # Stage 3b: PI-advisor LLM relevance scores — feeds both the candidate
    # filter (if oversized) AND the final _rerank() weighted formula (0.25
    # weight on llm_advisor). Auto-activates when candidates > 10 OR
    # task_context was supplied (plan-driven retrieval benefits most from
    # semantic re-ranking since the task description is rich).
    #
    # Explicit opt-out: callers that need a fast turnaround (e.g. the
    # UserPromptSubmit hook with a 9s budget against a workspace of 4k+
    # chunks) pass `llm_prefilter=False` to skip this stage entirely.
    # Without the opt-out the auto-trigger >10 candidates would always fire.
    llm_scores: dict[str, float] = {}
    _llm_pref = opts.get("llm_prefilter")
    # WHY(#workspace-uuid-sot perf): the PI-advisor is now OPT-IN (llm_prefilter=True)
    # only — it makes a CLOUD-LLM call (qwen-coder, 20% cloud-routed, 10-30s) that
    # blew the 9s hook budget. The auto-trigger ">10 candidates OR task_context"
    # fired on every real search (workspace = 4k+ chunks). Default search now relies
    # on the fast symbolic+vector+feedback rerank; deep/recherche callers opt in.
    if _llm_pref is True and ollama_url and candidates:
        # Only LLM-advise a top-N pre-ranked by the cheap symbolic+vector scores —
        # one bounded advisor call instead of scoring the whole workspace.
        ADVISOR_TOP_N = 30
        pre = sorted(
            candidates,
            key=lambda c: symbolic_scores.get(c.chunk_id, 0.0)
            + vector_scores.get(c.chunk_id, 0.0),
            reverse=True,
        )[:ADVISOR_TOP_N]
        llm_model = opts.get("llm_prefilter_model", "qwen2.5-coder:7b")
        llm_scores = _llm_relevance_scores(
            query, pre, ollama_url,
            model=llm_model,
            task_context=task_context,
        )
        candidates = sorted(
            pre, key=lambda c: llm_scores.get(c.chunk_id, 0.5), reverse=True
        )[:max(top_k * 2, 6)]

    # Build source_type lookup for session_compacted boost
    source_type_map: dict[str, str] = {}
    if session_compacted and candidates:
        source_ids = list({c.source_id for c in candidates})
        placeholders = ",".join(["?"] * len(source_ids))
        rows = conn.execute(
            f"SELECT source_id, source_type FROM sources WHERE source_id IN ({placeholders})",
            source_ids,
        ).fetchall()
        source_type_map = {r[0]: r[1] for r in rows}

    # Stage 3c: Predicted-Topics (Issue #184) — Markov-Chain auf
    # conversation_summaries fragt: nach Topic A folgt typischerweise B+C.
    # Chunks mit B/C in category_labels oder Text bekommen einen kleinen
    # Score-Boost — erweitert Recall ohne Mainstream-Score zu kapern.
    # Skip wenn kein repo_slug oder kein keyword_index → predicted_topics
    # bleibt leer und der Boost ist 0 (no-op).
    predicted_topics: set[str] = set()
    _pred_repo = opts.get("predicted_repo_slug") or repo
    if _pred_repo:
        try:
            from mayring_core.memory.predictive import predict_next_topics_for_query
            preds = predict_next_topics_for_query(
                query, conn, _pred_repo, top_k=5,
            )
            predicted_topics = {p.to_topic.lower() for p in preds}
        except (ImportError, sqlite3.OperationalError, AttributeError):
            # WHY(v2-stufe2.1): predictive-Modul Optional. konkrete Fehler
            # schlucken; anderes muss laut.
            predicted_topics = set()

    # Stage 4: re-rank — v1 (hand-tuned _WEIGHTS) by default; v2 (learned)
    # when RERANKER_VERSION=v2 (or =auto and the hash splits this way) AND
    # cache/rerank_v2.json exists. See src/memory/reranker_v2.py.
    from mayring_core.memory.reranker_v2 import get_active_reranker as _get_active
    rr_version, _ = _get_active(query_hint=query,
                                explicit_override=opts.get("reranker_version"))
    opts["_reranker_used"] = rr_version  # bubbled into diagnostics
    # IGIO-intent: query nach outcome/issue/goal/intervention scannen.
    # Override per opts["igio_intent"] möglich (z.B. explicit aus UI).
    igio_intent = opts.get("igio_intent") or detect_igio_intent(query)
    opts["_igio_intent"] = igio_intent  # für diagnostics

    ranked = _rerank(
        candidates, vector_scores, symbolic_scores, top_k, conn,
        affinity_source_id,
        source_type_map=source_type_map,
        session_compacted=session_compacted,
        llm_scores=llm_scores,
        reranker_query_hint=query,
        reranker_override=opts.get("reranker_version"),
        predicted_topics=predicted_topics,
        category_hint=opts.get("category_hint"),
        igio_intent=igio_intent,
        task_id=opts.get("task_id"),
    )

    # Enrich with cross-source refs (same text found in other sources).
    # WHY(v2-stufe2.1): cross-source-Anreicherung ist nice-to-have; konkrete
    # sqlite-Fehler schlucken; anderes muss laut.
    try:
        from mayring_core.memory.store import get_source_refs
        for r in ranked:
            all_refs = get_source_refs(conn, r.chunk_id)
            r.also_in_sources = [s for s in all_refs if s != r.source_id]
    except sqlite3.OperationalError:
        pass

    # Store full records in query cache (text stripped — re-fetched on
    # hit). Caching only (chunk_id, score_final) used to drop all stage
    # scores + reasons on hit, which is why the smoke RAG check kept
    # seeing score_vector=0 for cache-hot queries.
    import dataclasses as _dc
    _QUERY_CACHE[_ck] = [_dc.replace(r, text="") for r in ranked]
    return ranked


# ---------------------------------------------------------------------------
# Stage 5: compress_for_prompt
# ---------------------------------------------------------------------------

def compress_for_prompt(results: list[RetrievalRecord], char_budget: int) -> str:
    """Format retrieval results as a context string within char_budget.

    Strategy:
    1. Deduplicate by source_id (keep highest-scoring chunk per source)
    2. Prefer summary over text when text > 500 chars
    3. Stop adding entries when budget would be exceeded
    """
    if not results:
        return ""

    # Step 1: dedup by source_id
    seen_sources: dict[str, RetrievalRecord] = {}
    for r in results:
        if r.source_id not in seen_sources:
            seen_sources[r.source_id] = r

    deduped = list(seen_sources.values())

    lines = ["## Memory Context", ""]
    used = sum(len(l) + 1 for l in lines)

    for r in deduped:
        cats = ", ".join(r.category_labels) if r.category_labels else "?"
        body = r.summary if (r.summary and len(r.text) > 500) else r.text
        # Truncate body to remaining budget divided by remaining entries
        remaining_entries = len(deduped) - deduped.index(r)
        per_entry_budget = max(100, (char_budget - used) // max(1, remaining_entries))
        if len(body) > per_entry_budget:
            body = body[:per_entry_budget - 3] + "..."

        entry = f"- [{cats}] {r.source_id}\n  {body}"
        if r.rationale_edges:
            rationale_lines = ["\n  **Rationale:**"]
            for re_dict in r.rationale_edges:
                tgt = re_dict.get("target", "")
                why = re_dict.get("why", "")
                ctx = re_dict.get("context", "")
                ctx_suffix = f" ({ctx})" if ctx else ""
                rationale_lines.append(f"  - `{tgt}` — {why}{ctx_suffix}")
            entry = entry + "\n".join(rationale_lines)
        entry_len = len(entry) + 1

        if used + entry_len > char_budget:
            break

        lines.append(entry)
        used += entry_len

    if len(lines) <= 2:
        return ""

    return "\n".join(lines)
