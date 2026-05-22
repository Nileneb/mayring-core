"""Backward-compat Re-Export Shim.

Die Ingest-Pipeline wohnt jetzt in `src/memory/ingestion/*.py`.
Dieser Shim bleibt, damit bestehende Imports (`from mayring_core.memory.ingest
import ingest, mayring_categorize, ...`) und Test-Mocks
(`patch("mayring_core.memory.ingest.<name>")`) weiter funktionieren.

Neue Aufrufer: direkt aus `mayring_core.memory.ingestion.<submodul>` importieren.
"""
from __future__ import annotations

# Re-export subprocess so tests can still do
# `patch("mayring_core.memory.ingest.subprocess.run", ...)`.
import subprocess  # noqa: F401

from mayring_core.memory.chunker import (  # noqa: F401  — historically re-exported
    _chunk_js,
    _chunk_markdown,
    _chunk_python,
    _chunk_yaml_json,
    _make_file_chunk,
    structural_chunk,
)
from mayring_core.memory.ingestion.categorization import (  # noqa: F401
    _INGEST_DEFAULT_FALLBACK,
    _INGEST_DEFAULTS,
    _MODE_TO_TEMPLATE,
    _ORIGINAL_MAYRING_CATEGORIES,
    _load_mayring_template,
    _path_fallback_category,
    _resolve_codebook,
    _task_relevant_categories,
    mayring_categorize,
    reduce_categories,
)
from mayring_core.memory.ingestion.conversation import ingest_conversation_summary  # noqa: F401
from mayring_core.memory.ingestion.core import (  # noqa: F401
    MEMORY_CHROMA_DIR,
    _HAS_CHROMADB,
    _IMAGE_EXTENSIONS,
    _is_image_file,
    get_or_create_chroma_collection,
    ingest,
    resolve_dedup,
)
from mayring_core.memory.ingestion.github_issues import fetch_issues, issues_to_sources  # noqa: F401
from mayring_core.memory.ingestion.image import ingest_image  # noqa: F401
from mayring_core.memory.ingestion.multiview import generate_multiview_chunks  # noqa: F401
from mayring_core.memory.ingestion.utils import (  # noqa: F401
    _coerce_str,
    _log_memory_event,
    _MEMORY_LOG_PATH,
    _now_iso,
    configure_memory_log,
)
# Backward-compat re-export of image discovery workflow
from mayring_core.memory.image_ingest import discover_images, run_image_ingest  # noqa: F401

__all__ = [
    "MEMORY_CHROMA_DIR",
    "configure_memory_log",
    "discover_images",
    "fetch_issues",
    "generate_multiview_chunks",
    "get_or_create_chroma_collection",
    "ingest",
    "ingest_conversation_summary",
    "ingest_image",
    "issues_to_sources",
    "mayring_categorize",
    "reduce_categories",
    "resolve_dedup",
    "run_image_ingest",
    "structural_chunk",
    "_task_relevant_categories",
]
