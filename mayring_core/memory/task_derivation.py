"""Research-question derivation: Userprompt → Forschungsfrage (Mayring-konform).

Mayring-Inhaltsanalyse arbeitet konstitutiv kategorien-geleitet aus einer
Forschungsfrage. Die alten tech-labels (api, ui, data_access, +
[neu]…-Inflation) erfüllten das nicht — sie clusterten nach Code-Domäne,
nicht nach "was hilft mir bei diesem Vorhaben".

Dieses Modul leitet aus einem User-Prompt eine **Forschungsfrage** ab
(5-12 Wörter, eine pro Prompt). Cascade:
  1. Embedding-similarity-check gegen existierende research_questions
  2. Wenn top-match sim > 0.85 → reuse (occurrence_count++)
  3. Sonst → mistral:7b-instruct call mit strikten JSON-prompt,
     persistiere neue research_questions row mit embedding.

Cascade verhindert vocabulary-explosion + nutzt three.linn.games-GPU
statt Cloud-Calls (siehe globale CLAUDE.md-Präferenz).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from typing import Optional

from mayring_core.memory.db_adapter import DBAdapter

_log = logging.getLogger(__name__)

# WHY(2026-05-11): 0.85 ist empirisch — gleicher prompt-thema clustert
# >0.85 mit nomic-embed-text. Niedriger = task-fragmentation; höher =
# duplikate. Anpassbar via env wenn nötig.
_TASK_SIM_THRESHOLD = 0.85

_TASK_DERIVATION_PROMPT = """Du extrahierst aus einem User-Prompt EINE Forschungsfrage/Task im Stil der Mayring-Inhaltsanalyse.

Eine Task ist:
- 5-12 Wörter
- Konkrete Aufgabe oder Frage des Users
- Sprachlich neutral (kein "ich will", "können wir", "bitte")
- Wenn möglich Deutsch (User-Sprache aus dem Prompt)

Antworte AUSSCHLIESSLICH mit JSON: {"task": "<task-title>"}
Keine Erklärung, keine Codeblöcke, kein Markdown.

Beispiele:
Prompt: "wie kann ich JWT-tokens validieren mit Sanctum?"
Antwort: {"task": "JWT-Token-Validierung mit Sanctum implementieren"}

Prompt: "der deploy ist schon wieder kaputt, was läuft falsch?"
Antwort: {"task": "Deploy-Fehler diagnostizieren und beheben"}

Prompt: "show me where we handle paper PDF download"
Antwort: {"task": "Paper-PDF-Download-Logik lokalisieren"}

User-Prompt: """


def _embed_text(text: str, ollama_url: str) -> Optional[list[float]]:
    """Embed via nomic-embed-text. Returns None on failure (caller must handle)."""
    try:
        import requests
        resp = requests.post(
            ollama_url.rstrip("/") + "/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=15,
        )
        if resp.status_code != 200:
            _log.warning("task-derive embed HTTP %s", resp.status_code)
            return None
        data = resp.json()
        embeddings = data.get("embeddings") or [data.get("embedding")]
        if embeddings and embeddings[0]:
            return list(embeddings[0])
        return None
    except Exception as e:
        _log.warning("task-derive embed failed: %s", e)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _load_research_question_embeddings(
    conn: DBAdapter, workspace_id: str
) -> list[tuple[str, str, list[float]]]:
    """Return [(research_question_id, title, embedding)] for non-empty embeddings in this workspace."""
    rows = conn.execute(
        """SELECT research_question_id, title, embedding_id FROM research_questions
           WHERE workspace_id = ? AND embedding_id != ''""",
        (workspace_id,),
    ).fetchall()

    # embedding_id holds a JSON-encoded vector (we keep it inline rather than
    # in Chroma to avoid a roundtrip per derive — research_questions corpus is small
    # (<10k expected) and dimensions are 768).
    result = []
    for row in rows:
        try:
            vec = json.loads(row[2])
            if isinstance(vec, list) and vec:
                result.append((row[0], row[1], vec))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


def _call_mistral_for_task(prompt: str, ollama_url: str, model: str) -> Optional[str]:
    """Returns task-title from mistral JSON-mode, or None on parse-fail."""
    try:
        import requests
        resp = requests.post(
            ollama_url.rstrip("/") + "/api/generate",
            json={
                "model": model,
                "prompt": _TASK_DERIVATION_PROMPT + prompt.strip()[:1500],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 100},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            _log.warning("task-derive mistral HTTP %s", resp.status_code)
            return None

        raw = resp.json().get("response", "").strip()
        # WHY: format=json should yield pure JSON, but mistral occasionally
        # wraps in markdown — strip fences defensively.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)
        task = data.get("task")
        if isinstance(task, str) and 5 <= len(task) <= 200:
            return task.strip()
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        _log.warning("task-derive mistral parse fail: %s", e)
        return None
    except Exception as e:
        _log.warning("task-derive mistral call fail: %s", e)
        return None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def derive_research_question_fast(
    prompt: str,
    conn: DBAdapter,
    ollama_url: str,
    workspace_id: str = "default",
    *,
    sim_threshold: float = _TASK_SIM_THRESHOLD,
) -> Optional[dict]:
    """Hot-path-safe research_question lookup: embed + similarity-check only, NO mistral.

    WHY(2026-05-11): derive_research_question() macht im cascade-fall-through einen
    mistral-call (bis 30s timeout) — das in /memory/search inline zu
    haben hat jeden such-call 5-30s langsamer gemacht (+ die hook-9s-
    budget-timeouts ausgelöst). Diese fast-variante macht NUR den
    embedding-sim-check (~50-150ms): wenn eine existierende research_question matched,
    return sie (+ occurrence_count++); sonst None — KEINE neue row.
    Neue research_questions werden async via derive_research_question_background() erstellt.

    Returns {"research_question_id", "title", "reused": True} bei match, sonst None.
    """
    prompt = (prompt or "").strip()
    if len(prompt) < 5:
        return None

    prompt_emb = _embed_text(prompt, ollama_url)
    if prompt_emb is None:
        return None

    existing = _load_research_question_embeddings(conn, workspace_id)
    best_rq_id: Optional[str] = None
    best_title: Optional[str] = None
    best_sim = 0.0
    for rq_id, title, vec in existing:
        sim = _cosine(prompt_emb, vec)
        if sim > best_sim:
            best_sim = sim
            best_rq_id = rq_id
            best_title = title

    if best_rq_id is not None and best_sim >= sim_threshold:
        conn.execute(
            """UPDATE research_questions
               SET occurrence_count = occurrence_count + 1,
                   last_used_at = ?
               WHERE research_question_id = ?""",
            (_now_iso(), best_rq_id),
        )
        return {"research_question_id": best_rq_id, "title": best_title, "reused": True}
    return None


def derive_research_question_background(
    prompt: str,
    db_path,
    ollama_url: str,
    workspace_id: str = "default",
) -> None:
    """Fire-and-forget: create a new research_question for a prompt in a daemon thread.

    WHY(2026-05-11): /memory/search calls derive_research_question_fast (cheap) for the
    boost. If no existing research_question matched, this kicks off the mistral-creation
    in the background so the search response isn't delayed — the NEXT query
    on the same topic will find the new research_question. Opens its own DB connection
    (caller's conn may be on another thread).
    """
    import threading

    def _work() -> None:
        try:
            from mayring_core.memory.store import init_memory_db
            conn = init_memory_db(db_path)
            try:
                derive_research_question(prompt, conn, ollama_url, workspace_id)
            finally:
                conn.close()
        except Exception as e:
            _log.warning("derive_research_question_background failed: %s", e)

    t = threading.Thread(target=_work, daemon=True)
    t.start()


def derive_research_question(
    prompt: str,
    conn: DBAdapter,
    ollama_url: str,
    workspace_id: str = "default",
    *,
    model: Optional[str] = None,
    sim_threshold: float = _TASK_SIM_THRESHOLD,
) -> Optional[dict]:
    """Derive (or reuse) a research_question for a user prompt — FULL cascade.

    NOT for the hot path (it may make a 30s mistral call) — use
    derive_research_question_fast() for /memory/search, derive_research_question_background() to
    create new research_questions async, and this directly only from explicit/batch
    callers (pi_task, a periodic job).

    Cascade:
      1. Embed prompt
      2. Compare to all existing research_question embeddings in this workspace
      3. If best > sim_threshold: increment occurrence_count, return existing
      4. Else: mistral-call → new research_questions row

    Returns {"research_question_id": ..., "title": ..., "reused": bool} or None on failure.
    """
    prompt = (prompt or "").strip()
    if len(prompt) < 5:
        return None

    prompt_emb = _embed_text(prompt, ollama_url)
    if prompt_emb is None:
        return None

    existing = _load_research_question_embeddings(conn, workspace_id)

    best_rq_id: Optional[str] = None
    best_title: Optional[str] = None
    best_sim = 0.0
    for rq_id, title, vec in existing:
        sim = _cosine(prompt_emb, vec)
        if sim > best_sim:
            best_sim = sim
            best_rq_id = rq_id
            best_title = title

    if best_rq_id is not None and best_sim >= sim_threshold:
        conn.execute(
            """UPDATE research_questions
               SET occurrence_count = occurrence_count + 1,
                   last_used_at = ?
               WHERE research_question_id = ?""",
            (_now_iso(), best_rq_id),
        )
        return {"research_question_id": best_rq_id, "title": best_title, "reused": True}

    # Cascade-fall-through → mistral
    if model is None:
        try:
            from mayring_core.model_router import ModelRouter
            router = ModelRouter(ollama_url=ollama_url)
            model = router.resolve("text")
        except Exception:
            model = "mistral:7b-instruct"

    task_title = _call_mistral_for_task(prompt, ollama_url, model)
    if not task_title:
        return None

    # Store the originating PROMPT embedding (already computed above as prompt_emb)
    # so the similarity match path — which always embeds the incoming prompt — is
    # comparing prompt-vs-prompt consistently. Storing the title embedding here was
    # the root cause of the dedup bug: prompt-embedding vs title-embedding is
    # apples-vs-oranges → near-identical prompts scored below 0.85 → never clustered.
    rq_id = "rq_" + hashlib.sha256(task_title.encode()).hexdigest()[:16]
    now = _now_iso()
    conn.execute(
        """INSERT OR IGNORE INTO research_questions
           (research_question_id, title, embedding_id, occurrence_count, first_seen_at, last_used_at, workspace_id)
           VALUES (?, ?, ?, 1, ?, ?, ?)""",
        (rq_id, task_title, json.dumps(prompt_emb), now, now, workspace_id),
    )

    return {"research_question_id": rq_id, "title": task_title, "reused": False}


def link_chunk_to_research_question(
    conn: DBAdapter,
    research_question_id: str,
    chunk_id: str,
    relevance_score: float = 1.0,
) -> None:
    """Insert or boost a research_question_chunk_links row. Idempotent: re-call increments score."""
    existing = conn.execute(
        "SELECT relevance_score FROM research_question_chunk_links WHERE research_question_id = ? AND chunk_id = ?",
        (research_question_id, chunk_id),
    ).fetchone()

    if existing is not None:
        new_score = min(5.0, float(existing[0]) + relevance_score)
        conn.execute(
            "UPDATE research_question_chunk_links SET relevance_score = ? WHERE research_question_id = ? AND chunk_id = ?",
            (new_score, research_question_id, chunk_id),
        )
    else:
        conn.execute(
            """INSERT INTO research_question_chunk_links (research_question_id, chunk_id, relevance_score, created_at)
               VALUES (?, ?, ?, ?)""",
            (research_question_id, chunk_id, relevance_score, _now_iso()),
        )


def get_research_question_boost(
    conn: DBAdapter,
    research_question_id: str,
    chunk_ids: list[str],
) -> dict[str, float]:
    """Return {chunk_id → relevance_score} for chunks linked to the given research_question."""
    if not research_question_id or not chunk_ids:
        return {}

    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""SELECT chunk_id, relevance_score FROM research_question_chunk_links
            WHERE research_question_id = ? AND chunk_id IN ({placeholders})""",
        [research_question_id] + chunk_ids,
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows}


