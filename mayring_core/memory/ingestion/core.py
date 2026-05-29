"""Core ingestion orchestrator: chunking → dedup → embed → store → log."""
from __future__ import annotations

import logging
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import threading as _threading

_log = logging.getLogger(__name__)
_CHROMA_WRITE_LOCK = _threading.Lock()

if TYPE_CHECKING:
    from mayring_core.model_router import ModelRouter

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **_kw):  # type: ignore[misc]
        return it

try:
    import chromadb as _chromadb  # noqa: F401
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from mayring_core.config import CACHE_DIR
from mayring_core.memory.chunker import structural_chunk
from mayring_core.memory.ingestion.categorization import (
    _INGEST_DEFAULTS,
    _INGEST_DEFAULT_FALLBACK,
)
from mayring_core.memory.ingestion.image import (
    _IMAGE_EXTENSIONS,
    _is_image_file,
    ingest_image,
)
from mayring_core.memory.ingestion.multiview import generate_multiview_chunks
from mayring_core.memory.ingestion.utils import log_memory_event, now_iso
from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import (
    add_source_ref,
    batch_context,
    deactivate_chunks_by_source,
    find_by_text_hash,
    get_source,
    insert_chunk,
    kv_get,
    kv_put,
    log_ingestion_event,
    upsert_source,
)


MEMORY_CHROMA_DIR: Path = CACHE_DIR / "memory_chroma"


def _parse_label_array(raw: str, n: int) -> list[str]:
    """Parse the batched reduction's reply into exactly n labels. Tries a JSON array
    first, then a line-based fallback; raises (NOT silenced) if neither yields n —
    the caller logs loud and the source stays uncategorized rather than mislabeled."""
    import json
    import re
    s = (raw or "").strip()
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and len(arr) == n:
                return [str(x).strip() for x in arr]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip().strip('",[]').strip() for ln in s.splitlines()]
    lines = [re.sub(r"^\[?\d+\]?[.):\s-]*", "", ln).strip() for ln in lines if ln]
    if len(lines) == n:
        return lines
    raise ValueError(
        f"_parse_label_array: konnte {n} Labels nicht aus LLM-Antwort lesen: {s[:200]!r}")


def _batch_reduce_labels(items: list[tuple[str, str]], ollama_url: str,
                         model: str | None) -> list[str]:
    """Eine gebatchte Mayring-Reduktion: N Textstellen (alle mit DEMSELBEN Ziel) → N
    snake_case-Kategorie-Labels in Reihenfolge. Über die zentrale generate_text-Route
    (PiQueue/cloud-split). Ein Retry, dann lautes Scheitern (kein stummes Mislabeling)."""
    from mayring_core.providers import generate_text
    if not items:
        return []
    task = items[0][1]
    numbered = "\n".join(f"[{i}] {text[:800]}" for i, (text, _t) in enumerate(items))
    prompt = (
        "Du bildest Kategorien nach qualitativer Inhaltsanalyse (Mayring).\n"
        f"ZIEL/Aufgabe (obligatorischer Bezug für ALLE Stellen): {task[:300]}\n\n"
        f"Für jede der {len(items)} nummerierten Textstellen: gehe Paraphrase→"
        "Generalisierung→Reduktion und bilde EINE prägnante Kategorie (snake_case, "
        "max 4 Wörter), die den Bezug zum Ziel wahrt.\n\n"
        f"{numbered}\n\n"
        f"Antworte AUSSCHLIESSLICH mit einem JSON-Array aus genau {len(items)} Strings "
        'in derselben Reihenfolge, z.B. ["label_0", "label_1"]. Kein weiterer Text.'
    )
    last_err: Exception | None = None
    for _ in range(2):
        raw = generate_text(prompt=prompt, ollama_url=ollama_url, model=model,
                            label="mayring_reduce_batch")
        try:
            return _parse_label_array(raw, len(items))
        except ValueError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def _resolve_target_codebook_id(conn: Any, source_type: str) -> int:
    """Wohin echte Neu-Kategorien geschrieben werden: das Basis-Codebook der Domäne
    der Quelle (code→'generic', social→'sozialforschung'). Der deduktive Match bleibt
    cross-codebook; nur das induktive Neu-Bilden braucht ein Ziel. Fallback 'generic'."""
    from mayring_core.memory.ingestion.categorization import _SOURCE_TYPE_TO_CODEBOOK
    domain = _SOURCE_TYPE_TO_CODEBOOK.get(source_type)
    slug = {"code": "generic", "social": "sozialforschung"}.get(domain, "generic")
    row = conn.execute("SELECT id FROM codebooks WHERE slug=?", (slug,)).fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM codebooks ORDER BY id LIMIT 1").fetchone()
    return int(row[0]) if row else 1


def resolve_dedup(
    conn: Any,
    chunk: Chunk,
    workspace_id: str = "default",
) -> tuple[Chunk, bool]:
    """Exact dedup via text_hash (workspace-scoped).

    Returns (existing_chunk, True) if duplicate found in same workspace.
    Returns (chunk, False) if no duplicate — caller should insert.
    """
    existing = find_by_text_hash(conn, chunk.text_hash, workspace_id=workspace_id)
    if existing is not None:
        return existing, True
    return chunk, False


def get_or_create_chroma_collection(chroma_dir: Path | None = None):
    """Get or create the 'memory_chunks' ChromaDB collection (process singleton)."""
    if not _HAS_CHROMADB:
        return None
    from mayring_core.memory.store import get_chroma_collection as get_collection
    return get_collection("memory_chunks", path=chroma_dir)


def ingest(
    source: Source,
    content: str,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    opts: dict | None = None,
    router: "ModelRouter | None" = None,
    workspace_id: str = "default",
) -> dict:
    """Orchestrate the full ingestion pipeline for one source.

    opts:
        categorize (bool, default True): run mayring_categorize()
        log        (bool, default False): write JSONL event
        codebook   (str):  codebook name (default "auto")
        mode       (str):  categorisation mode (default "hybrid")
        multiview  (bool): use view-chunking for github_issue

    Returns:
        {source_id, state, chunk_ids, indexed, deduped, superseded}

        state ∈ {"new", "changed", "unchanged"}:
          - "new":       source_id never seen before
          - "changed":   source_id known, content_hash differs (old chunks deactivated)
          - "unchanged": source_id known, content_hash identical (fast-path skip)
    """
    opts = opts or {}

    # Image fast-path — route known image extensions to ingest_image() when
    # a vision model is registered in the router. Keeps a single pipeline
    # with consistent chunk_level="image_caption".
    if _is_image_file(source.path):
        if source.source_type != "image":
            source = _dc_replace(source, source_type="image")
        if router is not None and router.is_available("vision"):
            try:
                return ingest_image(
                    source=source,
                    image_path=Path(source.path),
                    conn=conn,
                    chroma_collection=chroma_collection,
                    ollama_url=ollama_url,
                    model=model,
                    vision_model=router.resolve("vision"),
                    workspace_id=workspace_id,
                )
            except Exception:
                pass  # fall through to generic pipeline with source_type=image

    if router is not None and not model and router.is_available("text"):
        model = router.resolve("text")

    defaults = _INGEST_DEFAULTS.get(source.source_type, _INGEST_DEFAULT_FALLBACK)
    effective = {**defaults, **opts}

    do_categorize: bool = bool(effective.get("categorize", True)) and bool(model)
    do_log:        bool = bool(effective.get("log", False))
    do_multiview:  bool = bool(effective.get("multiview", False))
    do_force:      bool = bool(effective.get("force", False))
    mode:          str  = effective.get("mode", "hybrid")
    codebook_choice: str = effective.get("codebook", "auto")
    # WHY(2026-05-11): Mayring Selektionskriterium — das Thema worauf
    # kategorisiert wird. Callers (app.linn.games paper-ingest: die
    # forschungsfrage; memory-hook: der derived task) geben das mit.
    # Empty → die prompts leiten das Thema aus dem chunk selbst ab.
    categorize_task: str = str(effective.get("task", "") or "")
    # Phase 3.2 (project-scoped codebook): after chunks are embedded, link each
    # to its closest active codebook category (deductive, LLM-free) → fills
    # chunk_categories for the reranker. Scope = shared base ∪ this project's own.
    link_project_id = effective.get("project_id") or None
    do_link: bool = bool(effective.get("link_categories", True))

    from mayring_core.providers import embed_texts as _embed_texts

    # State detection: NEW (source unseen) vs CHANGED (hash differs) vs
    # UNCHANGED (hash identical → fast-path skip). do_force lifts the cache
    # completely (used by /populate?force_reingest=true) and forces re-ingest
    # of an existing source as CHANGED.
    existing_src = get_source(conn, source.source_id) if source.content_hash else None

    if (
        existing_src
        and source.content_hash
        and existing_src.content_hash == source.content_hash
        and not do_force
    ):
        return {
            "source_id": source.source_id,
            "state": "unchanged",
            "chunk_ids": [], "indexed": False,
            "deduped": 0, "superseded": 0,
        }

    state = "changed" if existing_src else "new"

    # CHANGED: deactivate old chunks before re-ingest so retrieval no longer
    # returns stale content. The CHANGED-with-force path is already handled by
    # workflows/memory_ingest.py before calling ingest(); this covers the
    # natural CHANGED case (different hash, no --force-reingest).
    if state == "changed" and not do_force:
        deactivate_chunks_by_source(conn, source.source_id)

    # Ganze Pipeline (upsert_source + alle insert_chunk + log_events) läuft
    # unter einem Commit. Das eliminiert bei einem typischen Populate
    # (500 Files × 5 Chunks) ca. 2500 einzelne Commits und spart messbar
    # Zeit ohne Transaktionssemantik zu brechen — Rollback bei Exception
    # ist in batch_context enthalten.
    with batch_context(conn):
        upsert_source(conn, source, workspace_id=workspace_id)
        log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

        if do_multiview and source.source_type == "github_issue" and model:
            chunks = generate_multiview_chunks(source.source_id, content, ollama_url, model)
        else:
            chunks = structural_chunk(content, source.source_id, source.path)

        # WHY(canonical Mayring method): Kategorisierung = die EINE goal-anchored mixed
        # Pipeline (categorize_chunks unten) — Ziel → Paraphrase/Generalisierung/Reduktion
        # → cosine 0.75 → vorhandene nutzen, sonst induktiv neu. KEIN separater cheap
        # cosine-only-Pfad mehr. Die free-string category_labels bleiben FK-abgeleitet
        # (eine SoT chunk_categories). Siehe [[feedback-mayring-canonical-method]].
        _ = (do_categorize, mode, codebook_choice)  # legacy flags, von categorize_chunks ersetzt

        new_chunk_ids: list[str] = []
        chunks_to_categorize: list[tuple[str, str]] = []
        deduped_count = 0
        skipped_filter = 0
        indexed = False

        from mayring_core.memory.ingestion.conversation_filter import should_skip_chunk

        for chunk in _tqdm(chunks, desc="Chunks embedden", unit="chunk", leave=False):
            skip, reason = should_skip_chunk(chunk.text, source.source_type)
            if skip:
                skipped_filter += 1
                _log.info(
                    "pre-ingest filter skipped chunk %s (source=%s): %s",
                    chunk.chunk_id[:12], source.source_id, reason,
                )
                continue

            canonical, is_dup = resolve_dedup(conn, chunk, workspace_id=workspace_id)
            if is_dup:
                deduped_count += 1
                add_source_ref(conn, canonical.chunk_id, source.source_id, workspace_id)
                continue

            insert_chunk(conn, chunk, workspace_id=workspace_id)
            add_source_ref(conn, chunk.chunk_id, source.source_id, workspace_id)

            try:
                emb = _embed_texts([chunk.text[:500]], ollama_url)[0]
            except Exception as exc:
                _log.warning("embed failed for %s: %s", chunk.chunk_id[:12], exc)
                emb = None

            if chroma_collection is not None and emb is not None:
                try:
                    with _CHROMA_WRITE_LOCK:
                        chroma_collection.upsert(
                            ids=[chunk.chunk_id],
                            documents=[chunk.text[:500]],
                            embeddings=[emb],
                            metadatas=[{
                                "workspace_id": workspace_id,
                                "source_id": chunk.source_id,
                                "chunk_level": chunk.chunk_level,
                                "category_labels": ",".join(chunk.category_labels),
                                "category_source": chunk.category_source,
                                "category_confidence": chunk.category_confidence,
                                "is_active": 1,
                            }],
                        )
                    indexed = True
                except Exception as exc:
                    _log.warning("chroma upsert failed for %s: %s",
                                 chunk.chunk_id[:12], exc)

            kv_put(chunk.chunk_id, chunk.to_dict())
            new_chunk_ids.append(chunk.chunk_id)
            # Die Reduktion (categorize_chunks) embedded das abgeleitete Kategorie-Label,
            # nicht das Chunk-Embedding → hier nur (chunk_id, text) sammeln.
            chunks_to_categorize.append((chunk.chunk_id, chunk.text))

        log_ingestion_event(
            conn,
            source.source_id,
            "ingest_done",
            {"chunks": len(new_chunk_ids), "deduped": deduped_count, "filtered": skipped_filter},
        )

    # WHY(canonical Mayring method): die EINE goal-anchored mixed Kategorisierung läuft
    # OUTSIDE batch_context auf den schon committeten Chunks. Ein Fehler (LLM/Chroma kalt)
    # darf die gespeicherten Chunks NIE zurückrollen — laut geloggt, nie stumm. Ziel ist
    # obligatorisch: explizite Aufgabe (paper=Forschungsfrage, hook=derived task) oder, wenn
    # keine, ein aus der Quelle abgeleiteter Anker — nie ganz ohne Bezug (sonst random).
    category_links = 0
    goal = categorize_task.strip() or f"Inhalte aus {source.source_type}: {source.path}"
    if do_link and chunks_to_categorize and model:
        from mayring_core.memory.ingestion.mayring_process import categorize_chunks
        from mayring_core.memory.store import get_chroma_collection
        from mayring_core.providers import embed_texts as _embed_batch
        try:
            results = categorize_chunks(
                chunks_to_categorize, goal,
                _resolve_target_codebook_id(conn, source.source_type),
                conn=conn, chroma_categories=get_chroma_collection("codebook_categories"),
                batch_embed_fn=lambda texts: _embed_batch(texts, ollama_url),
                batch_reduce_fn=lambda pairs: _batch_reduce_labels(pairs, ollama_url, model),
                match_codebook_id=None, project_id=link_project_id)
            category_links = sum(1 for r in results if r.category_id is not None)
            _log.info("mayring categorize: %d/%d chunks kategorisiert (source=%s, project=%s, ziel=%r)",
                      category_links, len(chunks_to_categorize), source.source_id,
                      link_project_id, goal[:60])
            # Free-string category_labels FK-abgeleitet (eine SoT). insert_chunk schrieb sie
            # leer; die ~10 Konsumenten (IGIO-Classifier, wiki-edges, recap, Such-Anzeige,
            # paper_rules) laufen unverändert weiter — gleiche Spalte, FK-Quelle.
            from mayring_core.memory.ingestion.mayring_process import derive_labels_from_categories
            label_map = derive_labels_from_categories(conn, new_chunk_ids)
            for cid, labels in label_map.items():
                top = labels[:5]
                conn.execute(
                    "UPDATE chunks SET category_labels = ?, category_source = 'deductive-link' "
                    "WHERE chunk_id = ? AND is_active = 1",
                    (",".join(top), cid),
                )
                # WHY(2026-05-29): the loop above kv_put the chunk with EMPTY labels
                # (insert ran before this derive); refresh the kv entry so retrieval
                # (which hydrates from kv_get first) doesn't serve stale-empty
                # category_labels for freshly-ingested chunks. The FK-based cat_match
                # reads chunk_categories fresh, so only the labels view was stale.
                cached = kv_get(cid)
                if cached is not None:
                    cached["category_labels"] = top
                    cached["category_source"] = "deductive-link"
                    kv_put(cid, cached)
            conn.commit()
        except Exception as exc:
            _log.warning("deductive category-link failed (source=%s): %s",
                         source.source_id, exc)

    result = {
        "source_id": source.source_id,
        "state": state,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "filtered": skipped_filter,
        "superseded": 0,
        "category_links": category_links,
    }

    if do_log:
        log_memory_event({"event": "ingest", "ts": now_iso(), **result})

    return result


__all__ = [
    "MEMORY_CHROMA_DIR",
    "_HAS_CHROMADB",
    "_IMAGE_EXTENSIONS",
    "_is_image_file",
    "get_or_create_chroma_collection",
    "ingest",
    "resolve_dedup",
]
