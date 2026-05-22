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
    mayring_categorize,
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
    kv_put,
    log_ingestion_event,
    upsert_source,
)


MEMORY_CHROMA_DIR: Path = CACHE_DIR / "memory_chroma"


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

        if do_categorize and model:
            chunks = mayring_categorize(
                chunks, ollama_url, model,
                mode=mode, codebook=codebook_choice,
                source_type=source.source_type,
                conn=conn,
                router=router,
                workspace_id=workspace_id,
                task=categorize_task,
                chroma_collection=chroma_collection,
            )

        new_chunk_ids: list[str] = []
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

        log_ingestion_event(
            conn,
            source.source_id,
            "ingest_done",
            {"chunks": len(new_chunk_ids), "deduped": deduped_count, "filtered": skipped_filter},
        )

    result = {
        "source_id": source.source_id,
        "state": state,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "filtered": skipped_filter,
        "superseded": 0,
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
