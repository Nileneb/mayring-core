"""Utilities shared across the ingestion submodules."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mayring_core.config import CACHE_DIR


def coerce_str(val: object) -> str:
    """Normalise LLM-JSON values (list/int/None) to plain string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Private aliases (backward-compat — some callers still use leading-underscore names)
_coerce_str = coerce_str
_now_iso = now_iso


# ---------------------------------------------------------------------------
# Optional JSONL ingestion logging (opt-in per slug)
# ---------------------------------------------------------------------------

_MEMORY_LOG_PATH: Path | None = None


def configure_memory_log(slug: str) -> None:
    """Enable JSONL logging to cache/<slug>_memory_log.jsonl."""
    global _MEMORY_LOG_PATH
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_LOG_PATH = CACHE_DIR / f"{slug}_memory_log.jsonl"


def log_memory_event(event: dict) -> None:
    """Append one JSON line to the memory log. No-op if not configured."""
    if _MEMORY_LOG_PATH is None:
        return
    try:
        with _MEMORY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass


_log_memory_event = log_memory_event
