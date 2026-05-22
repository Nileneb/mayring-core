"""Predictive topic transitions — Markov chain over conversation-summary topic sequences."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TopicTransition:
    from_topic: str
    to_topic: str
    count: int
    probability: float


def _extract_topics_from_text(text: str, keyword_index: dict[str, list[str]]) -> list[str]:
    """Return ordered list of cluster-topics mentioned in text (dedup preserving order)."""
    if not text or not keyword_index:
        return []
    text_lower = text.lower()
    seen: list[str] = []
    words = re.findall(r'\b[a-z][a-z0-9_-]{2,}\b', text_lower)
    for w in words:
        clusters = keyword_index.get(w, [])
        for c in clusters:
            if c not in seen:
                seen.append(c)
    return seen


# WHY(#185, security): path-traversal defence — workspace_slug kommt aus
# user-controlled Body im /conversation/micro-batch endpoint. Ein naive
# Path-Build wie `Path("cache") / f"{slug}_wiki_index.json"` resolved bei
# slug='../../payload/EVIL' ausserhalb von cache/. Plus-Glück mit dem
# _wiki_index.json-Suffix limitiert den Schaden nicht zuverlässig.
# CHANGE WITH CARE — lockerer Regex öffnet path-traversal sofort wieder.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}\Z")


def _load_keyword_index(repo_slug: str | None) -> dict[str, list[str]]:
    """Load wiki_index.json for repo_slug. Returns {} if missing OR
    repo_slug fails the strict slug-whitelist (path-traversal defence).

    Triple-layered defence so CodeQL's taint-tracker sees a clean path:
      1. Fail-closed regex whitelist on the user input.
      2. Lookup gegen os.listdir(cache_dir) — der gefilterte Filename
         stammt damit aus dem trusted Filesystem-Listing, nicht aus
         user input. Dies entkoppelt den path-build vollständig vom
         tainted repo_slug.
      3. resolve() + relative_to-Check als belt-and-suspenders.
    """
    if not repo_slug or not _SLUG_RE.match(repo_slug):
        return {}
    cache_dir = Path("cache").resolve()
    if not cache_dir.is_dir():
        return {}
    expected_name = f"{repo_slug}_wiki_index.json"
    # CodeQL-freundlich: filename kommt aus dem (trusted) Directory-Listing,
    # nicht direkt aus dem user-controlled string. Der whitelist-check oben
    # garantiert dass repo_slug exakt zu einem fs-Eintrag matcht oder gar
    # nicht — taint-flow wird hier hart unterbrochen.
    matching = [p for p in cache_dir.iterdir() if p.name == expected_name]
    if not matching:
        return {}
    path = matching[0]
    try:
        path.relative_to(cache_dir)
    except ValueError:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _slug_aliases(repo_slug: str) -> list[str]:
    """Return all slug variants that historically point to the same repo.

    Conversation-watcher und CLI haben in ihrer Geschichte verschiedene
    Slug-Konventionen verwendet:
      * volle URL → ``_repo_slug`` macht ``owner-name`` (z.B. ``nileneb-mayringcoder``)
      * watcher pre-2026-05 schrieb nur ``name`` (z.B. ``mayringcoder``)
      * mixed-case / case-only-Variationen
    Ohne Alias-Matching findet die Markov-Pipeline nur 1-2% ihrer
    eigentlich verfügbaren conversation-summaries.
    """
    if not repo_slug:
        return []
    out = [repo_slug]
    if "-" in repo_slug:
        # owner-name → name
        out.append(repo_slug.rsplit("-", 1)[-1])
    out.append(repo_slug.lower())
    if repo_slug.lower() != repo_slug:
        out.append(repo_slug)
    # dedup, preserve order
    seen: set[str] = set()
    deduped = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def build_transition_matrix(
    conn: Any,
    repo_slug: str = "",
    limit: int = 100,
) -> dict[str, dict[str, int]]:
    """Scan last `limit` conversation-summary chunks, extract topic sequences,
    build sparse Markov counts {from_topic: {to_topic: count}}.
    """
    aliases = _slug_aliases(repo_slug)
    if aliases:
        placeholders = ",".join(["?"] * len(aliases))
        rows = conn.execute(
            f"""SELECT c.text FROM chunks c
               JOIN sources s ON c.source_id = s.source_id
               WHERE s.source_type = 'conversation_summary'
                 AND s.repo IN ({placeholders})
                 AND c.is_active = 1
               ORDER BY s.captured_at DESC
               LIMIT ?""",
            (*aliases, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.text FROM chunks c
               JOIN sources s ON c.source_id = s.source_id
               WHERE s.source_type = 'conversation_summary'
                 AND c.is_active = 1
               ORDER BY s.captured_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    kw_index = _load_keyword_index(repo_slug)
    if not kw_index:
        return {}

    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        text = row[0]
        topics = _extract_topics_from_text(text, kw_index)
        for a, b in zip(topics, topics[1:]):
            if a == b:
                continue
            matrix.setdefault(a, {})
            matrix[a][b] = matrix[a].get(b, 0) + 1
    return matrix


def predict_next_topics(
    current_topic: str,
    matrix: dict[str, dict[str, int]],
    top_k: int = 3,
) -> list[TopicTransition]:
    """Return top_k TopicTransition entries for current_topic, sorted by probability desc."""
    transitions = matrix.get(current_topic, {})
    total = sum(transitions.values())
    if total == 0:
        return []
    scored = [
        TopicTransition(current_topic, to_t, cnt, cnt / total)
        for to_t, cnt in transitions.items()
    ]
    scored.sort(key=lambda t: (-t.probability, t.to_topic))
    return scored[:top_k]


def update_transitions_incremental(
    text: str,
    conn: Any,
    repo_slug: str,
) -> int:
    """Inkrementell pro neuer conversation_summary die Markov-counts bumpen.

    Issue #55-Followup: build_transition_matrix() scannt 100 chunks und ist
    deshalb teuer + nicht ohne Cron triggerbar. Diese Funktion macht den
    Update pro NEUEM summary direkt im /conversation/micro-batch-Hook —
    keine Cron, kein Plugin-systemd-Geraffel. Inkrementelle Updates
    bedeuten: Transitions sind immer fresh, Cost ~O(topics_in_text).

    Returns die Anzahl der gebumpten Topic-Paare (0 = keinen wiki_index
    oder zu wenig Topics für mindestens 1 Übergang).
    """
    if not text:
        return 0
    kw_index = _load_keyword_index(repo_slug)
    if not kw_index:
        return 0
    topics = _extract_topics_from_text(text, kw_index)
    if len(topics) < 2:
        return 0
    now = datetime.utcnow().isoformat()
    pairs = 0
    for a, b in zip(topics, topics[1:]):
        if a == b:
            continue
        conn.execute(
            """INSERT INTO topic_transitions(from_topic, to_topic, count, last_seen)
               VALUES(?,?,1,?)
               ON CONFLICT(from_topic, to_topic) DO UPDATE SET
                   count = count + 1,
                   last_seen = ?""",
            (a, b, now, now),
        )
        pairs += 1
    conn.commit()
    return pairs


def predict_next_topics_for_query(
    query: str,
    conn: Any,
    repo_slug: str = "",
    top_k: int = 3,
) -> list[TopicTransition]:
    """Map ``query`` → top-1 topic via wiki keyword-index, return predicted
    next-topics from the persisted transition matrix.

    Used by /memory/search to widen recall: if the user asks about topic
    A and the Markov chain says A→B with high probability, chunks tagged
    with B can be co-injected even when their vector/symbolic score is
    just below the cut. Empty list when no match in keyword-index OR no
    persisted transitions yet.
    """
    if not query:
        return []
    kw_index = _load_keyword_index(repo_slug)
    if not kw_index:
        return []
    topics = _extract_topics_from_text(query, kw_index)
    if not topics:
        return []
    matrix = load_transitions(conn)
    if not matrix:
        return []
    return predict_next_topics(topics[0], matrix, top_k=top_k)


def persist_transitions(matrix: dict[str, dict[str, int]], conn: Any) -> None:
    """Upsert matrix into topic_transitions table."""
    now = datetime.utcnow().isoformat()
    for from_t, inner in matrix.items():
        for to_t, cnt in inner.items():
            conn.execute(
                """INSERT INTO topic_transitions(from_topic, to_topic, count, last_seen)
                   VALUES(?,?,?,?)
                   ON CONFLICT(from_topic, to_topic) DO UPDATE SET
                       count = ?,
                       last_seen = ?""",
                (from_t, to_t, cnt, now, cnt, now),
            )
    conn.commit()


def load_transitions(conn: Any) -> dict[str, dict[str, int]]:
    """Load persisted matrix from topic_transitions table."""
    rows = conn.execute(
        "SELECT from_topic, to_topic, count FROM topic_transitions"
    ).fetchall()
    matrix: dict[str, dict[str, int]] = {}
    for from_t, to_t, cnt in rows:
        matrix.setdefault(from_t, {})[to_t] = cnt
    return matrix
