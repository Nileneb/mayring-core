"""Derive an actionable todo from a user prompt (server-side LLM).

Mirrors task_derivation's prompt-embedding dedup (prompt-vs-prompt — the bug
we fixed). NOT for the hot path; run in a daemon thread off micro-batch.
"""
from __future__ import annotations
import json, logging
from typing import Optional
from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.task_derivation import _embed_text, _cosine  # reuse
from mayring_core.memory import tasks as _t

_log = logging.getLogger(__name__)
_SIM = 0.85
_PROMPT = (
    "Du bekommst einen User-Prompt aus einer Coding-Session. Entscheide, ob er eine "
    "konkrete, umsetzbare Arbeits-Aufgabe (Todo) ausdrückt, die der User erledigt haben will "
    "(z.B. 'implementiere X', 'fixe Y', 'baue Z'). Reine Fragen, Smalltalk oder Status-Checks "
    "sind KEINE Todos. Antworte NUR mit JSON: "
    '{"actionable": true|false, "title": "<imperativer Titel, <=120 Zeichen, leer wenn nicht actionable>"}\n\nPrompt:\n'
)

def _llm_todo(prompt: str, ollama_url: str, model: str) -> Optional[dict]:
    try:
        import requests
        resp = requests.post(ollama_url.rstrip("/") + "/api/generate",
            json={"model": model, "prompt": _PROMPT + prompt.strip()[:1500],
                  "format": "json", "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 120}}, timeout=30)
        if resp.status_code != 200:
            return None
        data = json.loads(resp.json().get("response", "").strip())
        if not isinstance(data, dict):
            return None
        return {"actionable": bool(data.get("actionable")),
                "title": str(data.get("title") or "").strip()[:120]}
    except Exception as e:
        _log.warning("derive_todo LLM fail: %s", e)
        return None

def derive_todo(prompt: str, conn: DBAdapter, ollama_url: str, workspace_id: str,
                *, model: Optional[str] = None) -> Optional[dict]:
    prompt = (prompt or "").strip()
    if len(prompt) < 8:
        return None
    if model is None:
        try:
            from mayring_core.model_router import ModelRouter
            model = ModelRouter(ollama_url=ollama_url).resolve("text")
        except Exception:
            model = "mistral:7b-instruct"
    verdict = _llm_todo(prompt, ollama_url, model)
    if not verdict or not verdict["actionable"] or not verdict["title"]:
        return None
    prompt_emb = _embed_text(prompt, ollama_url)
    if prompt_emb is None:
        return None
    # dedup: prompt-vs-prompt against this workspace's OPEN derived todos
    for (eid,) in conn.execute(
        "SELECT derive_embedding FROM tasks WHERE workspace_id=? AND status!='done' "
        "AND derive_embedding IS NOT NULL", (workspace_id,)).fetchall():
        try:
            if _cosine(prompt_emb, json.loads(eid)) >= _SIM:
                return None
        except Exception:
            continue
    row = _t.create_task(conn, workspace_id=workspace_id, title=verdict["title"],
                         created_by="derived", tags="derived")
    conn.execute("UPDATE tasks SET derive_embedding=? WHERE task_id=?",
                 (json.dumps(prompt_emb), row["task_id"]))
    conn.commit()
    return {"task_id": row["task_id"], "title": verdict["title"], "created": True}
