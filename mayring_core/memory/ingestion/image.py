"""Image ingestion — vision captioning for single image files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import (
    add_source_ref,
    insert_chunk,
    kv_put,
    log_ingestion_event,
    upsert_source,
)
from mayring_core.memory.ingestion.utils import now_iso


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def ingest_image(
    source: Source,
    image_path: Path,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    vision_model: str | None = None,
    workspace_id: str = "default",
) -> dict:
    """Ingest a single image file via vision captioning.

    SVGs are ingested as raw text. Raster images are captioned via the Ollama
    multimodal model (vision_model) and stored as text chunks with embeddings.

    Returns:
        {source_id, state, chunk_ids, indexed, deduped, superseded}

        state ∈ {"new", "changed", "unchanged"} — same contract as core.ingest().
    """
    from mayring_core.providers import vision_caption as caption_image, vision_metadata as get_image_metadata
    from mayring_core.providers import embed_texts as _embed_texts
    if vision_model is None:  # resolve from the single source, never a literal
        from mayring_core.model_router import ModelRouter
        vision_model = ModelRouter(ollama_url).resolve("vision")
    # Late import avoids circular dep core -> image -> core
    from mayring_core.memory.ingestion.core import resolve_dedup
    from mayring_core.memory.store import deactivate_chunks_by_source, get_source

    existing_src = get_source(conn, source.source_id) if source.content_hash else None

    if (
        existing_src
        and source.content_hash
        and existing_src.content_hash == source.content_hash
    ):
        return {
            "source_id": source.source_id,
            "state": "unchanged",
            "chunk_ids": [], "indexed": False,
            "deduped": 0, "superseded": 0,
        }

    state = "changed" if existing_src else "new"
    if state == "changed":
        deactivate_chunks_by_source(conn, source.source_id)

    upsert_source(conn, source, workspace_id=workspace_id)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

    metadata = get_image_metadata(image_path) or {}

    caption = caption_image(image_path, ollama_url, vision_model)
    if not caption.strip():
        fmt = metadata.get("format", "")
        w = metadata.get("width", 0)
        h = metadata.get("height", 0)
        size = metadata.get("file_size", 0)
        caption = (
            f"Image file: {image_path.name}"
            + (f" ({fmt}, {w}x{h} px, {size} bytes)" if fmt else f" ({size} bytes)")
        )

    is_svg = Path(source.path).suffix.lower() == ".svg"
    category_labels = ["diagram"] if is_svg else ["image"]

    text_hash = Chunk.compute_text_hash(caption)
    chunk = Chunk(
        chunk_id=Chunk.make_id(source.source_id, 0, "image_caption"),
        source_id=source.source_id,
        chunk_level="image_caption",
        ordinal=0,
        start_offset=0,
        end_offset=len(caption),
        text=caption,
        text_hash=text_hash,
        dedup_key=text_hash,
        category_labels=category_labels,
        created_at=now_iso(),
    )

    canonical, is_dup = resolve_dedup(conn, chunk, workspace_id=workspace_id)
    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    if is_dup:
        deduped_count += 1
        add_source_ref(conn, canonical.chunk_id, source.source_id, workspace_id)
    else:
        insert_chunk(conn, chunk, workspace_id=workspace_id)
        add_source_ref(conn, chunk.chunk_id, source.source_id, workspace_id)

        try:
            emb = _embed_texts([chunk.text[:500]], ollama_url)[0]
        except Exception as exc:
            _log.warning("image embed failed: %s", exc)
            emb = None

        if chroma_collection is not None and emb is not None:
            try:
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
                _log.warning("chroma upsert failed (image %s): %s",
                             chunk.chunk_id[:12], exc)

        kv_put(chunk.chunk_id, chunk.to_dict())
        new_chunk_ids.append(chunk.chunk_id)

    log_ingestion_event(
        conn,
        source.source_id,
        "ingest_done",
        {"chunks": len(new_chunk_ids), "deduped": deduped_count},
    )

    return {
        "source_id": source.source_id,
        "state": state,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "superseded": 0,
    }
