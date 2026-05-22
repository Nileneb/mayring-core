"""Claude /compact conversation-summary ingestion."""
from __future__ import annotations

import hashlib
from typing import Any

from mayring_core.memory.schema import Source
from mayring_core.memory.ingestion.utils import now_iso


def ingest_conversation_summary(
    summary_text: str,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    session_id: str | None = None,
    run_id: str | None = None,
    workspace_id: str = "default",
) -> dict:
    """Ingest a Claude /compact summary as a conversation_summary source.

    Args:
        summary_text: Raw Markdown text of the compaction summary.
        conn: SQLite connection (from init_memory_db()).
        chroma_collection: ChromaDB collection or None.
        ollama_url: Ollama base URL.
        model: Ollama model name (empty string = no embedding / categorization).
        session_id: Optional session identifier, stored in Source.branch.
        run_id: Optional run identifier, stored in Source.commit.

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    from mayring_core.memory.ingestion.core import ingest  # late import (cycle)

    content_hash = "sha256:" + hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
    path = f"summary/{session_id or 'unknown'}.md"
    source_id = Source.make_id("conversation", path)

    source = Source(
        source_id=source_id,
        source_type="conversation_summary",
        repo="conversation",
        path=path,
        branch=session_id or "",
        commit=run_id or "",
        content_hash=content_hash,
        captured_at=now_iso(),
    )

    return ingest(
        source=source,
        content=summary_text,
        conn=conn,
        chroma_collection=chroma_collection,
        ollama_url=ollama_url,
        model=model,
        opts={
            "categorize": bool(model),
            "codebook": "social",
            "mode": "hybrid",
            "log": True,
        },
        workspace_id=workspace_id,
    )
