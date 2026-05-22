"""Mayring-based chunk categorization — codebook resolution + LLM labelling."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mayring_core.memory.schema import Chunk
    from mayring_core.model_router import ModelRouter

try:
    import yaml as _yaml  # noqa: F401
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


_PROMPTS_DIR: Path = Path(__file__).parent.parent.parent.parent.parent / "prompts"  # +1 (#267)
_CODEBOOK_DIR: Path = Path(__file__).parent.parent.parent.parent.parent / "codebooks"  # +1 (#267)

def _is_plausible_neu_label(inner: str) -> bool:
    """Reject obvious gibberish from weak models (e.g. mistral:7b on PHP code).

    A plausible new category is 2–30 chars, alphanumeric/underscore/hyphen
    only (no spaces, parens, or commentary like 'X (für die Klasse)'). For
    labels of length ≥ 5 we also reject single-character dominance > 60%
    (catches 'xxxx', 'aaaaaaa'); shorter abbreviations like 'llm', 'css'
    are accepted even with repeated letters.
    """
    inner = inner.strip().lower()
    if not (2 <= len(inner) <= 30):
        return False
    if not re.fullmatch(r"[a-zäöüß0-9_-]+", inner):
        return False
    if len(inner) >= 5 and max((inner.count(c) for c in set(inner)), default=0) > len(inner) * 0.6:
        return False
    return True

_ORIGINAL_MAYRING_CATEGORIES: list[str] = [
    "Zusammenfassung",
    "Explikation",
    "Strukturierung",
    "Paraphrase",
    "Reduktion",
    "Kategoriensystem",
    "Ankerbeispiel",
]

_MODE_TO_TEMPLATE: dict[str, str] = {
    "deductive": "mayring_deduktiv",
    "inductive": "mayring_induktiv",
    "hybrid":    "mayring_hybrid",
}

# Ingest defaults per source_type. `codebook` is ALWAYS "auto" — the
# source_type → codebook mapping lives in ONE place (_AUTO in
# _resolve_codebook), keyed by the *logic* of what the content is, not by
# whoever happened to write the caller. Per-source-type entries here only
# vary `categorize` / `mode` / `multiview` (e.g. images aren't categorized,
# papers use multiview). If a source_type is missing here it falls through
# to _INGEST_DEFAULT_FALLBACK (also codebook="auto") → _resolve_codebook
# then FAILS LOUDLY rather than guessing.
_INGEST_DEFAULTS: dict[str, dict] = {
    "repo_file":            {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": False},
    "note":                 {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": False},
    "paper":                {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": True},
    "agent_result":         {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": True},
    "github_issue":         {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": True},
    "conversation_summary": {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": False},
    "session_knowledge":    {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": False},
    "session_note":         {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": False},
    "image":                {"categorize": False, "codebook": "auto", "mode": "hybrid", "multiview": False},
}
_INGEST_DEFAULT_FALLBACK: dict = {
    "categorize": True, "codebook": "auto", "mode": "hybrid", "multiview": False,
}

# source_type → codebook name. The mapping IS the logic:
#   - "code"   anchors (universal.yaml: api/auth/data_access/...) — content
#              that lives in / is about a codebase: repo files, notes, and
#              github issues (those are technical bug reports / feature
#              requests, NOT social-science prose — an "auth middleware bug"
#              issue should get `auth`, not `argumentation`).
#   - "social" anchors (social.yaml: argumentation/methodik/ergebnis/...) —
#              research / qualitative prose: papers, agent_result (= paper
#              full-text routed in from app.linn.games), conversations,
#              session knowledge.
# A source_type with NO entry here is a config bug → _resolve_codebook logs
# an error and tags the chunk with a "FAIL_..." category so it's visible in
# the UI, never silently shoehorned into whatever happens to be nearby.
_SOURCE_TYPE_TO_CODEBOOK: dict[str, str] = {
    "repo_file": "code",
    "note": "code",
    "github_issue": "code",
    "conversation": "social",
    "conversation_summary": "social",
    "session_knowledge": "social",
    "session_note": "social",
    "paper": "social",
    "agent_result": "social",
}


def _resolve_codebook(codebook: str, source_type: str) -> list[str]:
    """Return category names for the given codebook/source_type.

    Resolution order:
      1. "auto" → _SOURCE_TYPE_TO_CODEBOOK[source_type]; unmapped → FAIL-loud
      2. codebooks/<name>.yaml (code, social, or any custom)
      3. codebooks/profiles/<name>.yaml (generic, python, laravel, ...)
      4. Fallback → original Mayring categories
    """
    codebook = str(codebook).strip().lower()
    if codebook == "auto":
        mapped = _SOURCE_TYPE_TO_CODEBOOK.get(source_type)
        if mapped is None:
            # FAIL-loud: an unmapped source_type means somebody added a new
            # ingest path without updating _SOURCE_TYPE_TO_CODEBOOK. Don't
            # guess — log it and label the chunk so the gap is unmissable.
            logger.error(
                "codebook resolution: UNMAPPED source_type=%r — add it to "
                "_SOURCE_TYPE_TO_CODEBOOK. Chunk gets FAIL marker, not real labels.",
                source_type,
            )
            return [f"FAIL_unmapped_source_type:{source_type or 'EMPTY'}"]
        codebook = mapped

    if codebook == "original":
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    # Security: only alphanumeric + _ and - (blocks path traversal)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", codebook):
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    for candidate in [
        _CODEBOOK_DIR / f"{codebook}.yaml",
        _CODEBOOK_DIR / "profiles" / f"{codebook}.yaml",
    ]:
        if candidate.exists() and _HAS_YAML:
            try:
                import yaml as _yaml_local
                data = _yaml_local.safe_load(candidate.read_text(encoding="utf-8"))
                cats = data.get("categories", []) if isinstance(data, dict) else data
                names: list[str] = []
                for c in cats:
                    if isinstance(c, str):
                        names.append(c)
                    elif isinstance(c, dict):
                        n = c.get("label") or c.get("name", "")
                        if n:
                            names.append(n)
                if names:
                    return names
            except Exception:
                continue

    return list(_ORIGINAL_MAYRING_CATEGORIES)


def _path_fallback_category(path: str) -> list[str]:
    """Regex-based category from file path when LLM categorization fails."""
    _RULES = [
        (r"test_|_test\.|/tests/|conftest", "tests"),
        (r"/api/|/routes/|/controllers/|router", "api"),
        (r"/models?/|/db/|/migration|repositor", "data_access"),
        (r"/auth|/security|/guards?/|/policies?", "auth"),
        (r"/service|/domain|/usecase|/business", "domain"),
        (r"/middleware|/pipeline", "middleware"),
        (r"config.*\.(py|yaml|yml|env)|settings\.|constants\.", "config"),
        (r"/utils?/|/helpers?/|/tools?/", "utils"),
        (r"/cache|redis|memcache", "caching"),
        (r"/log|monitor|metric|trace", "logging"),
    ]
    for pattern, cat in _RULES:
        if re.search(pattern, path, re.IGNORECASE):
            return [cat]
    return []


_LABEL_LINE_PREFIX = re.compile(
    r"^\s*(?:kategorien|categories|labels|kategorie|category)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_TEST_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:tests?|spec|__tests__)/|test_[^/]+\.py|[^/]+_test\.(py|js|ts)|"
    r"[^/]+\.(?:test|spec)\.[a-z0-9]+|conftest",
    re.IGNORECASE,
)


def _looks_like_test_path(path: str) -> bool:
    """True if path looks like a test-file (pytest, jest, rspec, …)."""
    return bool(_TEST_PATH_PATTERN.search(path or ""))


def _extract_label_line(response: str) -> str:
    """Pull the comma-separated label line out of a structured LLM response.

    The hybrid prompt asks the model to think in three steps but emit only
    ``Kategorien: a, b, c``. If we find that line we use only its payload;
    otherwise we fall back to the full response (covers older prompts that
    just return a bare comma list).
    """
    match = _LABEL_LINE_PREFIX.search(response or "")
    if match:
        return match.group(1).strip()
    return response or ""


def _load_mayring_template(mode: str) -> str:
    """Load prompt template for the given mode. Falls back to inline default."""
    filename = _MODE_TO_TEMPLATE.get(mode, "mayring_hybrid") + ".md"
    template_path = _PROMPTS_DIR / filename
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError:
        return (
            "Categorize this text chunk using these categories if applicable: {{categories}}. "
            "Respond with ONLY a comma-separated list of labels."
        )


def _workspace_anchor_labels(conn: Any, workspace_id: str, *, exclude: set[str],
                             limit: int = 40) -> list[str]:
    """Top-`limit` category labels already used in this workspace, by frequency.

    Used to augment the hybrid prompt's anchor set so existing categories get
    reused across texts instead of re-invented. `[neu]X` labels are folded to
    `X` (a frequently-minted `[neu]` label IS a real category — promote it to
    an anchor). Excludes anything already in `exclude` (the static codebook),
    plus implausible / overlong tokens. Returns [] on any DB error — this is
    an enhancement, never a hard dependency.
    """
    try:
        rows = conn.execute(
            "SELECT category_labels FROM chunks "
            "WHERE workspace_id = ? AND is_active = 1 "
            "AND category_labels IS NOT NULL AND category_labels != ''",
            (workspace_id,),
        ).fetchall()
    except Exception:
        return []
    counts: dict[str, int] = {}
    for (csv,) in rows:
        for part in str(csv).split(","):
            tok = part.strip().lower()
            if tok.startswith("[neu]"):
                tok = tok[len("[neu]"):].strip()
            if not tok or tok in exclude:
                continue
            # reuse the same plausibility gate as for [neu] labels
            if not _is_plausible_neu_label(tok):
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tok for tok, _ in ranked[:max(0, limit)]]


def _task_relevant_categories(
    task: str,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    workspace_id: str,
    n_results: int = 20,
) -> list[str]:
    """Fetch existing category labels semantically relevant to `task` via ChromaDB.

    WHY: _workspace_anchor_labels() is frequency-based and topic-blind. A chunk
    about "Patientenautonomie" must see labels from OTHER chunks on that topic,
    not the globally most-used workspace labels. This embeds the task and queries
    ChromaDB so the LLM reuses established labels instead of inventing variants.

    Returns empty list on any failure — caller falls back to frequency anchors.
    """
    if not task or not chroma_collection or not ollama_url or not model:
        return []
    try:
        from mayring_core.providers import embed_texts as _embed_texts
        emb = _embed_texts([task[:500]], ollama_url)
        if not emb or not emb[0]:
            return []
        where = {"workspace_id": {"$eq": workspace_id}} if workspace_id else None
        results = chroma_collection.query(
            query_embeddings=[emb[0]],
            n_results=n_results,
            where=where,
            include=["metadatas"],
        )
    except Exception as exc:
        logger.warning("S3 task-category fetch failed: %s", exc)
        return []

    seen: set[str] = set()
    labels: list[str] = []
    for meta_list in (results.get("metadatas") or []):
        for meta in (meta_list or []):
            for tok in (meta.get("category_labels") or "").split(","):
                tok = tok.strip().lower()
                if tok.startswith("[neu]"):
                    tok = tok[len("[neu]"):]
                if tok and tok not in seen and _is_plausible_neu_label(tok):
                    seen.add(tok)
                    labels.append(tok)
    return labels


def mayring_categorize(
    chunks: "list[Chunk]",
    ollama_url: str,
    model: str,
    mode: str = "hybrid",
    codebook: str = "auto",
    source_type: str = "repo_file",
    conn: Any = None,
    router: "ModelRouter | None" = None,
    workspace_id: str = "default",
    task: str = "",
    chroma_collection: Any = None,
) -> "list[Chunk]":
    """Assign Mayring category labels to each chunk via LLM.

    Args:
        mode: "deductive" (closed category set), "inductive" (free derivation),
              "hybrid" (anchors + new categories marked with [neu])
        codebook: "auto" (detect from source_type), "code", "social", or profile name
        source_type: used for auto-detection of codebook
        conn: optional SQLite connection for error logging
        router: optional ModelRouter for task-based model selection
        task: the task/topic the categorization anchors to (Mayring
              Selektionskriterium). Empty → prompt derives the topic from
              the chunk itself. Domain-neutral string; callers just pass
              whatever they're working on (a question, a research topic,
              a code-task) — no special-casing here.
    """
    if router is not None and not model and router.is_available("text"):
        model = router.resolve("text")

    if not model or not ollama_url:
        return chunks

    try:
        from mayring_core.providers import generate_text as _ollama_generate
    except ImportError:
        return chunks

    categories = _resolve_codebook(codebook, source_type)
    static_set = {c.lower() for c in categories if c}
    # #244 TODO2: in hybrid mode, also offer the categories that already exist
    # in this workspace as anchors — so a second paper about "Patientenautonomie"
    # reuses that label instead of minting `[neu]patientenautonomie`. Without
    # this, every text invents its own variant and cross-text matching breaks.
    if mode == "hybrid" and conn is not None:
        extra = _workspace_anchor_labels(conn, workspace_id, exclude=static_set)
        if extra:
            categories = categories + extra

    # Task-aware semantic injection: embed task → ChromaDB → topic-specific labels
    # WHY(#task-hybrid): frequency anchors are topic-blind. A 128-section paper
    # produces 126 labels because "auth-check"/"auth-validation"/"authentication-check"
    # are never seen as equivalent. Injecting past labels for THIS topic lets the LLM
    # reuse them. Forces hybrid so [neu] is still possible for genuine novelty.
    _task_lbls = _task_relevant_categories(
        task, conn, chroma_collection, ollama_url, model, workspace_id
    )
    if _task_lbls:
        _existing_lower = {c.lower() for c in categories}
        for _lbl in _task_lbls:
            if _lbl not in _existing_lower:
                categories.append(_lbl)
                _existing_lower.add(_lbl)
        mode = "hybrid"

    valid_set = {c.lower() for c in categories if c}
    template = _load_mayring_template(mode)
    task_str = task.strip() if task and task.strip() else "(kein Task angegeben)"
    system_prompt = (
        template
        .replace("{{categories}}", ", ".join(categories))
        .replace("{{task}}", task_str)
    )

    for chunk in chunks:
        try:
            import time
            prompt = f"Text chunk:\n\n{chunk.text[:1200]}"
            _t0 = time.monotonic()
            response = _ollama_generate(
                prompt=prompt,
                ollama_url=ollama_url,
                model=model,
                label=f"mayring:{chunk.chunk_id[:8]}",
                system_prompt=system_prompt,
            )
            _elapsed_ms = int((time.monotonic() - _t0) * 1000)
            if conn is not None:
                try:
                    from mayring_core.memory.store import log_llm_call
                    log_llm_call(
                        conn=conn,
                        call_type="categorization",
                        model=model,
                        prompt=prompt,
                        response=response,
                        duration_ms=_elapsed_ms,
                        workspace_id=workspace_id,
                    )
                except Exception:
                    pass
            # The hybrid prompt asks for `Kategorien: a, b, c` — extract that
            # line if present, otherwise treat the full response as a raw list
            # (backwards-compat with older deductive/inductive templates).
            payload = _extract_label_line(response)
            raw = [re.sub(r"^[-•*]\s*", "", p).strip()
                   for p in re.split(r"[,\n]", payload)]

            validated: list[str] = []
            for lbl in raw:
                if not lbl or len(lbl) > 60 or "," in lbl or len(lbl.split()) > 4:
                    continue
                if mode == "inductive":
                    validated.append(lbl)
                elif mode == "hybrid" and lbl.lower().startswith("[neu]"):
                    # Anti-[neu]-Missbrauch: Wenn das Modell "[neu]X" emittiert
                    # aber X bereits im Anker-Set ist, normalisieren wir auf X.
                    inner = lbl[len("[neu]"):].strip().lower()
                    if inner in valid_set:
                        validated.append(inner)
                    elif _is_plausible_neu_label(inner):
                        validated.append(lbl)
                    # else: schwache Modelle (mistral:7b) halluzinieren Gibberish —
                    # silently dropped, kein Append.
                elif lbl.lower() in valid_set:
                    validated.append(lbl.lower())

            # Pfad-basierter Override: Test-Files MÜSSEN als tests erkennbar
            # bleiben, auch wenn das Modell "nur" inhaltlich kategorisiert hat
            # (z.B. test_user.py → "data_access" statt "tests, data_access").
            if _looks_like_test_path(chunk.source_id or "") and "tests" in valid_set \
                    and "tests" not in validated and "[neu]tests" not in validated:
                validated.append("tests")

            chunk.category_labels = validated[:5]
            chunk.category_source = mode
            chunk.category_confidence = 1.0 if validated else 0.0

        except Exception as exc:
            chunk.category_labels = _path_fallback_category(chunk.source_id or "")
            chunk.category_source = "fallback"
            chunk.category_confidence = 0.5 if chunk.category_labels else 0.0
            if conn is not None:
                try:
                    from mayring_core.memory.store import log_ingestion_event
                    log_ingestion_event(conn, chunk.chunk_id, "categorize_error",
                                        {"error": str(exc)[:200]})
                except Exception:
                    pass

    return chunks


_S7_CHROMA_LOCK = __import__("threading").Lock()
# WHY: _CHROMA_WRITE_LOCK lives in core.py — importing it here creates a circular
# dependency. A separate lock is safe: ChromaDB serializes internally; this just
# prevents concurrent S7 upserts from this module.


def reduce_categories(
    chunk_ids: list[str],
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    workspace_id: str = "default",
    router: "ModelRouter | None" = None,
    threshold: int = 15,
) -> dict:
    """Mayring S7 — Generalisierung + Bündelung (on-demand MCP tool only).

    WHY(S7): With task-aware S3 injection most label explosion is prevented.
    S7 remains as a cleanup tool for existing workspaces or edge cases where S3
    still produces near-duplicates (e.g. first ingest when no prior labels exist).
    NOT called automatically from ingest() — only via MCP reduce_categories tool.

    Collects unique labels from chunk_ids, calls LLM once with consolidation prompt,
    applies old→canonical mapping to SQLite + ChromaDB.
    Returns {mapping, chunks_updated, unique_before, unique_after, skipped}.
    """
    import json as _json
    import re as _re
    import time as _time

    if router is not None and not model and router.is_available("text"):
        model = router.resolve("text")

    if not model or not ollama_url:
        return {"skipped": True, "reason": "no model/url", "mapping": {},
                "chunks_updated": 0, "unique_before": 0, "unique_after": 0}
    if not chunk_ids:
        return {"skipped": True, "reason": "no chunk_ids", "mapping": {},
                "chunks_updated": 0, "unique_before": 0, "unique_after": 0}

    rows = conn.execute(
        f"SELECT category_labels FROM chunks "
        f"WHERE chunk_id IN ({','.join('?' * len(chunk_ids))}) AND is_active = 1",
        chunk_ids,
    ).fetchall()

    label_counter: dict[str, int] = {}
    for (csv,) in rows:
        for tok in (csv or "").split(","):
            tok = tok.strip().lower()
            if tok.startswith("[neu]"):
                tok = tok[len("[neu]"):]
            if tok and _is_plausible_neu_label(tok):
                label_counter[tok] = label_counter.get(tok, 0) + 1

    unique_labels = list(label_counter.keys())
    unique_before = len(unique_labels)

    if unique_before <= threshold:
        return {"skipped": True,
                "reason": f"unique_labels={unique_before} <= threshold={threshold}",
                "mapping": {}, "chunks_updated": 0,
                "unique_before": unique_before, "unique_after": unique_before}

    try:
        from mayring_core.providers import generate_text as _ollama_generate
    except ImportError:
        return {"skipped": True, "reason": "analyzer not available", "mapping": {},
                "chunks_updated": 0, "unique_before": unique_before, "unique_after": unique_before}

    template_path = _PROMPTS_DIR / "mayring_s7_reduktion.md"
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("S7 prompt missing: %s — skipping S7", template_path)
        return {"skipped": True, "reason": "prompt file missing", "mapping": {},
                "chunks_updated": 0, "unique_before": unique_before, "unique_after": unique_before}

    labels_json = _json.dumps(unique_labels, ensure_ascii=False)
    system_prompt = template.replace("{{labels}}", labels_json)

    _t0 = _time.monotonic()
    response = _ollama_generate(
        prompt=f"Labels:\n{labels_json}",
        ollama_url=ollama_url,
        model=model,
        label="mayring:s7",
        system_prompt=system_prompt,
    )
    _elapsed_ms = int((_time.monotonic() - _t0) * 1000)

    try:
        from mayring_core.memory.store import log_llm_call
        log_llm_call(conn=conn, call_type="s7_reduction", model=model,
                     prompt=labels_json[:500], response=response[:800],
                     duration_ms=_elapsed_ms, workspace_id=workspace_id)
    except Exception as log_exc:
        logger.warning("S7 log_llm_call failed: %s", log_exc)

    mapping: dict[str, str] = {}
    json_match = _re.search(r"\{[^{}]+\}", response, _re.DOTALL)
    if json_match:
        try:
            raw = _json.loads(json_match.group(0))
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    kc, vc = k.strip().lower(), v.strip().lower()
                    if _is_plausible_neu_label(kc) and _is_plausible_neu_label(vc):
                        mapping[kc] = vc
        except (_json.JSONDecodeError, ValueError) as exc:
            logger.warning("S7 JSON parse failed: %s — raw: %s", exc, response[:200])

    if not mapping:
        logger.warning("S7: empty mapping (model=%s, unique=%d) — skipping apply",
                       model, unique_before)
        return {"skipped": True, "reason": "empty mapping", "mapping": {},
                "chunks_updated": 0, "unique_before": unique_before, "unique_after": unique_before}

    from mayring_core.memory.store import update_chunk_category_labels
    chunks_updated = update_chunk_category_labels(conn, chunk_ids, mapping)

    chroma_updated = 0
    if chroma_collection is not None and chunks_updated > 0:
        updated_rows = conn.execute(
            f"SELECT chunk_id, category_labels, chunk_level, source_id, "
            f"category_source, category_confidence FROM chunks "
            f"WHERE chunk_id IN ({','.join('?' * len(chunk_ids))}) "
            f"AND category_source = 's7-reduced' AND is_active = 1",
            chunk_ids,
        ).fetchall()
        for cid, new_csv, chunk_level, source_id, cat_src, cat_conf in updated_rows:
            try:
                with _S7_CHROMA_LOCK:
                    chroma_collection.upsert(
                        ids=[cid],
                        metadatas=[{
                            "workspace_id": workspace_id,
                            "source_id": source_id or "",
                            "chunk_level": chunk_level or "",
                            "category_labels": new_csv or "",
                            "category_source": cat_src or "s7-reduced",
                            "category_confidence": cat_conf or 0.0,
                            "is_active": 1,
                        }],
                    )
                chroma_updated += 1
            except Exception as chroma_exc:
                logger.warning("S7 chroma upsert failed %s: %s", cid[:12], chroma_exc)

    unique_after = len({v for v in mapping.values()})
    logger.info("S7: workspace=%s %d→%d labels, %d/%d chunks updated, chroma=%d",
                workspace_id, unique_before, unique_after,
                chunks_updated, len(chunk_ids), chroma_updated)

    return {"skipped": False, "mapping": mapping, "chunks_updated": chunks_updated,
            "unique_before": unique_before, "unique_after": unique_after}
