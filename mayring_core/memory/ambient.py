"""Ambient Context Layer — kompakter Projekt-Snapshot für Pi-Agent."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store_batch import batch_context

@dataclass
class TriggerResult:
    """Result of trigger_scan() — context string + which triggers fired."""
    context: str
    trigger_ids: list[str]  # e.g. ["keyword:creditservice", "cluster:CreditCluster"]


@dataclass
class ContextFeedback:
    """Implicit feedback record for one Pi-Agent interaction."""
    trigger_ids: list[str]
    context_text: str
    was_referenced: bool
    led_to_retrieval: bool
    relevance_score: float   # 0.0–1.0
    captured_at: str


_SNAPSHOT_SYSTEM = (
    "Du bist ein präziser Assistent. Erstelle einen kompakten Projekt-Snapshot auf Deutsch. "
    "Keine Begrüßung, kein Nachsatz. Maximal 600 Wörter."
)

_SNAPSHOT_PROMPT = """\
## Aktuelle Conversations (letzte 5 Zusammenfassungen)
{conversation_summaries}

## Offene Issues / bekannte Probleme
{issue_summaries}

## Top-Verbindungen im Wiki (stärkste Abhängigkeiten)
{wiki_connections}

---
Erstelle einen Projekt-Snapshot im Format:

**Aktueller Stand:** <1-2 Sätze was zuletzt implementiert/geändert wurde>

**Architektur-Hotspots:** <die 3-5 wichtigsten Dateien/Module mit kurzer Rolle>

**Offene Punkte:** <max. 5 konkrete TODOs oder bekannte Probleme>

**Wichtige Zusammenhänge:** <max. 5 Datei/Modul-Paare mit Erklärung warum sie zusammengehören>
"""

_STICKY_RE = re.compile(r'\[sticky\]', re.IGNORECASE)


def _score_entry(
    entry_text: str,
    captured_at_iso: str,
    conn: DBAdapter,
    half_life_days: float = 14.0,
) -> float:
    """Score a snapshot entry on [0.0, ~1.5] range.

    Components:
    - Sticky items get minimum score 0.9 regardless of age
    - Base score 0.5 + recency_boost (first 24h: +0.3)
    - Decay by exp(-age_days / half_life)
    - Feedback boost: +0.2 per feedback_log row matching a 40-char prefix of entry
    """
    is_sticky = bool(_STICKY_RE.search(entry_text))
    base = 0.9 if is_sticky else 0.5

    age_days = 365.0
    if captured_at_iso:
        try:
            ts = datetime.fromisoformat(captured_at_iso.rstrip("Z"))
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            age_seconds = (datetime.utcnow() - ts).total_seconds()
            age_days = max(age_seconds / 86400.0, 0.0)
        except (ValueError, TypeError):
            pass

    recency_boost = 0.3 if age_days < 1.0 else 0.0
    decay = math.exp(-age_days / half_life_days)

    score = (base + recency_boost) * decay
    if is_sticky:
        score = max(score, 0.9)

    prefix = entry_text.strip()[:40]
    if conn is not None and prefix:
        # WHY(v2-stufe2.1): nur konkrete sqlite-Fehler schlucken (DB-locked,
        # Spalte fehlt nach migration). Alles andere muss laut.
        import sqlite3 as _sqlite3
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM context_feedback_log WHERE context_text LIKE ? AND was_referenced = 1",
                (prefix + "%",),
            ).fetchone()
            if row:
                score += 0.2 * min(int(row[0]), 3)
        except _sqlite3.OperationalError:
            pass

    return round(score, 4)


def _score_snapshot_entries(
    conn: DBAdapter,
    entries: list[tuple[str, str]],
) -> list[tuple[str, float]]:
    """Return entries ranked by score descending. Ties broken by original order."""
    indexed = [(i, text, _score_entry(text, ts, conn)) for i, (text, ts) in enumerate(entries)]
    indexed.sort(key=lambda x: (-x[2], x[0]))
    return [(text, score) for _, text, score in indexed]


def _load_recent_conversations(conn: DBAdapter, repo_slug: str, limit: int = 5) -> list[tuple[str, str]]:
    """Lade die letzten N Conversation-Summaries aus SQLite."""
    rows = conn.execute(
        """SELECT c.text, s.captured_at FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_type = 'conversation_summary'
             AND (s.repo = ? OR ? = '')
             AND c.is_active = 1
           ORDER BY s.captured_at DESC
           LIMIT ?""",
        (repo_slug, repo_slug, limit),
    ).fetchall()
    return [(r[0][:500], r[1] or "") for r in rows]


def _load_recent_issues(conn: DBAdapter, repo_slug: str, limit: int = 10) -> list[tuple[str, str]]:
    """Lade die letzten N Issue-Summaries aus SQLite."""
    rows = conn.execute(
        """SELECT c.text, s.captured_at FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_type = 'github_issue'
             AND (s.repo = ? OR ? = '')
             AND c.is_active = 1
           ORDER BY s.captured_at DESC
           LIMIT ?""",
        (repo_slug, repo_slug, limit),
    ).fetchall()
    return [(r[0][:300], r[1] or "") for r in rows]


def _load_wiki_top_connections(repo_slug: str, limit: int = 10) -> str:
    """Lade die stärksten Wiki-Verbindungen aus dem Wiki-Markdown."""
    wiki_path = Path("cache") / f"{repo_slug}_wiki.md"
    if not wiki_path.exists():
        return "(kein Wiki vorhanden)"
    content = wiki_path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.startswith("- →")]
    return "\n".join(lines[:limit]) or "(keine Verbindungen)"


def _cosine(a: list[float], b: list[float]) -> float:
    """Inline cosine similarity. Returns 0.0 on zero-vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _is_trigger_active(trigger_id: str, conn: DBAdapter) -> bool:
    """Returns True if trigger is active or not yet in trigger_stats."""
    row = conn.execute(
        "SELECT is_active FROM trigger_stats WHERE trigger_id = ?", (trigger_id,)
    ).fetchone()
    return row is None or bool(row[0])


def trigger_scan(
    user_message: str,
    keyword_index: dict[str, list[str]],
    cluster_embs: dict[str, list[float]],
    ollama_url: str,
    conn: DBAdapter | None = None,
    threshold: float = 0.75,
    max_tokens: int = 400,
) -> TriggerResult:
    """Cascaded trigger scan: keyword-first (< 1ms), embedding only on miss.

    Returns TriggerResult with context string (max_tokens chars) and fired trigger_ids.
    """
    msg_lower = user_message.lower()
    words = set(w.strip(".,!?;:()[]{}\"'") for w in msg_lower.split() if len(w) > 2)

    matched_clusters: list[str] = []
    fired_ids: list[str] = []

    # Stage 1: keyword match (< 1ms)
    for word in words:
        if word in keyword_index:
            trigger_id = f"keyword:{word}"
            if conn is not None and not _is_trigger_active(trigger_id, conn):
                continue
            matched_clusters.extend(keyword_index[word])
            fired_ids.append(trigger_id)

    if matched_clusters:
        top = list(dict.fromkeys(matched_clusters))[:3]
        ctx = f"[Relevante Cluster: {', '.join(top)}]"[:max_tokens]
        return TriggerResult(context=ctx, trigger_ids=fired_ids)

    # Stage 2: embedding cosine (only if no keyword hit)
    if not cluster_embs or not ollama_url:
        return TriggerResult(context="", trigger_ids=[])

    try:
        from mayring_core.providers import embed_texts as _embed_texts
        vecs = _embed_texts([user_message], ollama_url)
        if not vecs:
            return TriggerResult(context="", trigger_ids=[])
        q_vec = vecs[0]
        scores = [(name, _cosine(q_vec, emb)) for name, emb in cluster_embs.items()]
        scores.sort(key=lambda x: -x[1])
        active_pairs = [
            (name, f"cluster:{name}")
            for name, _ in [(n, s) for n, s in scores[:3] if s >= threshold]
            if conn is None or _is_trigger_active(f"cluster:{name}", conn)
        ]
        top_names = [n for n, _ in active_pairs]
        trigger_ids_emb = [t for _, t in active_pairs]
        if top_names:
            ctx = f"[Relevante Cluster: {', '.join(top_names)}]"[:max_tokens]
            return TriggerResult(context=ctx, trigger_ids=trigger_ids_emb)
    except Exception as _exc:
        import warnings
        warnings.warn(f"trigger_scan embedding failed: {_exc}", stacklevel=2)

    return TriggerResult(context="", trigger_ids=[])


def compute_feedback(
    context_text: str,
    llm_response: str,
    trigger_ids: list[str],
    led_to_retrieval: bool,
    conn: DBAdapter,
    ollama_url: str,
) -> ContextFeedback:
    """Compute implicit feedback by embedding-cosine context vs. response.

    Persists result to context_feedback_log. Returns ContextFeedback.
    """
    was_referenced = False
    relevance_score = 0.0

    if context_text and llm_response and ollama_url:
        # WHY(v2-stufe2.1): embed-call kann timeout/connection-error werfen
        # — log statt silent. Empty embed = keine relevance-score, fairer
        # default ist 0.0 + was_referenced=False.
        try:
            from mayring_core.providers import embed_texts as _embed_texts
            vecs = _embed_texts([context_text, llm_response], ollama_url)
            if len(vecs) == 2:
                relevance_score = round(_cosine(vecs[0], vecs[1]), 4)
                was_referenced = relevance_score >= 0.65
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "context-feedback embed skipped: %s", e,
            )

    captured_at = datetime.utcnow().isoformat()
    fb = ContextFeedback(
        trigger_ids=trigger_ids,
        context_text=context_text[:500],
        was_referenced=was_referenced,
        led_to_retrieval=led_to_retrieval,
        relevance_score=relevance_score,
        captured_at=captured_at,
    )

    # WHY(v2-stufe2.1): Telemetrie-Insert. sqlite-Fehler (DB-locked,
    # Spalte fehlt) schlucken; alles andere muss laut.
    import sqlite3 as _sqlite3
    try:
        with batch_context(conn):
            conn.execute(
                """INSERT INTO context_feedback_log
                   (trigger_ids, context_text, was_referenced, led_to_retrieval, relevance_score, captured_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (json.dumps(trigger_ids), context_text[:500],
                 int(was_referenced), int(led_to_retrieval),
                 relevance_score, captured_at),
            )
    except (_sqlite3.OperationalError, _sqlite3.IntegrityError) as e:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "context_feedback_log insert skipped: %s", e,
        )

    return fb


def update_trigger_stats(
    trigger_ids: list[str],
    was_referenced: bool,
    conn: DBAdapter,
    deactivate_threshold: float = 0.10,
    min_fires_for_deactivation: int = 50,
) -> None:
    """Increment fire_count + ref_count. Deactivate if below threshold after min_fires."""
    now = datetime.utcnow().isoformat()
    with batch_context(conn):
        for trigger_id in trigger_ids:
            conn.execute(
                """INSERT INTO trigger_stats (trigger_id, fire_count, ref_count, is_active, last_fired)
                   VALUES (?, 1, ?, 1, ?)
                   ON CONFLICT(trigger_id) DO UPDATE SET
                       fire_count = fire_count + 1,
                       ref_count  = ref_count + ?,
                       last_fired = ?""",
                (trigger_id, int(was_referenced), now, int(was_referenced), now),
            )
            row = conn.execute(
                "SELECT fire_count, ref_count FROM trigger_stats WHERE trigger_id = ?",
                (trigger_id,),
            ).fetchone()
            if row:
                fires, refs = row
                if fires >= min_fires_for_deactivation:
                    rate = refs / fires
                    if rate < deactivate_threshold:
                        conn.execute(
                            "UPDATE trigger_stats SET is_active = 0 WHERE trigger_id = ?",
                            (trigger_id,),
                        )


def generate_ambient_snapshot(
    conn: DBAdapter,
    ollama_url: str,
    model: str,
    repo_slug: str = "",
    workspace_id: str = "system",
) -> str | None:
    """Generiere Ambient-Snapshot via LLM und speichere in SQLite.

    Returns snapshot text on success, None if model is empty.
    """
    if not model:
        return None

    from mayring_core.providers import generate_text as _ollama_generate
    from mayring_core.memory.schema import Source
    from mayring_core.memory.ingest import ingest

    convs_tuples = _load_recent_conversations(conn, repo_slug)
    issues_tuples = _load_recent_issues(conn, repo_slug)
    convs_scored = _score_snapshot_entries(conn, convs_tuples)
    issues_scored = _score_snapshot_entries(conn, issues_tuples)
    convs = [t for t, _s in convs_scored]
    issues = [t for t, _s in issues_scored]
    wiki_top = _load_wiki_top_connections(repo_slug)

    prompt = _SNAPSHOT_PROMPT.format(
        conversation_summaries="\n".join(f"- {s}" for s in convs) or "(keine)",
        issue_summaries="\n".join(f"- {s}" for s in issues) or "(keine)",
        wiki_connections=wiki_top,
    )

    try:
        snapshot_text = _ollama_generate(
            prompt, ollama_url, model, "ambient_snapshot",
            system_prompt=_SNAPSHOT_SYSTEM,
        )
    except Exception:
        return None

    source_id = f"ambient:{repo_slug or 'global'}:snapshot"
    content_hash = "sha256:" + hashlib.sha256(snapshot_text.encode()).hexdigest()[:16]

    src = Source(
        source_id=source_id,
        source_type="ambient_snapshot",
        repo=repo_slug,
        path="ambient/snapshot",
        branch="local",
        commit="",
        content_hash=content_hash,
    )
    ingest(
        src, snapshot_text, conn, None,
        ollama_url, "",
        opts={"categorize": False, "chunk_level": "ambient_snapshot"},
        workspace_id=workspace_id,
    )
    return snapshot_text


def load_ambient_snapshot(conn: DBAdapter, repo_slug: str = "") -> str | None:
    """Lade den letzten Ambient-Snapshot aus SQLite."""
    source_id = f"ambient:{repo_slug or 'global'}:snapshot"
    row = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_id = ? AND c.is_active = 1
           ORDER BY s.captured_at DESC LIMIT 1""",
        (source_id,),
    ).fetchone()
    return row[0] if row else None


def _safe_repo_slug(repo_slug: str) -> str:
    """Return a deterministic filesystem-safe cache key for repo_slug, else empty string."""
    if not repo_slug:
        return ""
    return hashlib.sha256(repo_slug.encode("utf-8")).hexdigest()


def _safe_cache_file(cache_dir: Path, repo_slug: str, suffix: str) -> Path | None:
    """Build a cache file path under cache_dir from repo_slug, else return None."""
    safe_slug = _safe_repo_slug(repo_slug)
    if not safe_slug:
        return None

    base_dir = cache_dir.resolve()
    candidate = (base_dir / f"{safe_slug}_{suffix}").resolve()

    try:
        candidate.relative_to(cache_dir)
    except ValueError:
        return None

    return candidate


def build_context(
    task: str,
    conn: DBAdapter,
    ollama_url: str,
    repo_slug: str = "",
    _out_trigger_ids: list | None = None,
    chroma_collection: Any = None,
) -> str:
    """Orchestrator: lädt Snapshot + Trigger-Scan + Hybrid-Retrieval → kompakter Kontext-String.

    Leise skippen wenn kein Snapshot vorhanden (kein LLM-Call).
    chroma_collection=None deaktiviert den Retrieval-Block (rückwärtskompatibel).
    """
    safe_repo_slug = _safe_repo_slug(repo_slug)
    snapshot = load_ambient_snapshot(conn, safe_repo_slug)
    if not snapshot:
        return ""

    keyword_index: dict[str, list[str]] = {}
    cluster_embs: dict[str, list[float]] = {}
    cache_dir = Path("cache").resolve()
    idx_path = _safe_cache_file(cache_dir, safe_repo_slug, "wiki_index.json")
    emb_path = _safe_cache_file(cache_dir, safe_repo_slug, "wiki_clusters_emb.json")
    # WHY(v2-stufe2.1): cache-files sind optional; konkrete IO/JSON-errors
    # schlucken (corrupt cache, partial write). Anderes muss laut.
    if idx_path and idx_path.exists():
        try:
            keyword_index = json.loads(idx_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    if emb_path and emb_path.exists():
        try:
            cluster_embs = json.loads(emb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    result = trigger_scan(task, keyword_index, cluster_embs, ollama_url, conn=conn)
    if _out_trigger_ids is not None:
        _out_trigger_ids.extend(result.trigger_ids)
    trigger_hint = result.context

    # Dynamic budget — compress snapshot when triggers are rich
    trigger_len = len(trigger_hint)
    if trigger_len > 300:
        snapshot_budget = 1600
        retrieval_budget = 1200
    else:
        snapshot_budget = 3200
        retrieval_budget = 2400

    if len(snapshot) > snapshot_budget:
        snapshot = snapshot[:snapshot_budget] + "…"

    retrieval_section = ""
    if chroma_collection is not None:
        # WHY(v2-stufe2.1): retrieval-fail darf snapshot-Aufbau nicht stoppen,
        # aber konkret typisieren statt Exception schlucken.
        try:
            from mayring_core.memory.retrieval import search, compress_for_prompt
            hits = search(
                task, conn, chroma_collection, ollama_url,
                opts={"top_k": 5, "repo": safe_repo_slug or None},
            )
            retrieval_section = compress_for_prompt(hits, char_budget=retrieval_budget)
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "ambient retrieval-section skipped: %s", e,
            )

    parts = [f"## Projekt-Snapshot\n{snapshot}"]
    if trigger_hint:
        parts.append(f"## Trigger-Kontext\n{trigger_hint}")
    if retrieval_section:
        parts.append(f"## Relevante Erinnerungen\n{retrieval_section}")

    # Predictive topic hook (silent if no transitions persisted yet).
    # WHY(v2-stufe2.1): konkrete sqlite/import-Fehler schlucken, aber
    # alles andere (z.B. AttributeError nach API-Change) muss laut.
    import sqlite3 as _sqlite3
    try:
        from mayring_core.memory.predictive import load_transitions, predict_next_topics
        matrix = load_transitions(conn)
        if matrix and result.trigger_ids:
            current = None
            for tid in result.trigger_ids:
                if tid.startswith("cluster:"):
                    current = tid.split(":", 1)[1]
                    break
            if current:
                preds = predict_next_topics(current, matrix, top_k=3)
                if preds:
                    pred_lines = [f"- {p.to_topic} (p={p.probability:.2f})" for p in preds]
                    parts.append("## Wahrscheinlich relevant\n" + "\n".join(pred_lines))
    except (_sqlite3.OperationalError, ImportError) as e:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "predictive section skipped: %s", e,
        )

    return "\n\n".join(parts)
