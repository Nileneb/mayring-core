"""qwen text-route activity stamp — feeds the GPU-idle probe (Spec B1).

The text hot-path (Advisor/categorize/search/pi_run) loads qwen3.5-mayring:2b
into VRAM. Spec B routes the heavy Gemma research batch local-first ONLY while
qwen is idle, so app.linn.games must be able to ask "is qwen idle?". This module
records a single wall-clock timestamp on every text generate, and reads it back.

Storage is Redis (key ``mayring:qwen_last_activity``) when ``MAYRING_REDIS_URL``
is set — that is the idiomatic cross-worker store (see src/api/shared_state.py).
Without Redis it falls back to an atomically-written cache-file
(``<cache_dir>/qwen_last_activity``). Both paths are fail-soft: a write failure
must never break a generate call, and a read failure reports "no activity"
(→ idle), which is the safe default (Spec B: a dead idle-signal degrades to
"escalate to Cloud", never to a stuck local batch).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_REDIS_KEY = "mayring:qwen_last_activity"
_FILE_NAME = "qwen_last_activity"

_redis_client = None  # lazy, cached across calls
_redis_tried = False


def _get_redis():
    global _redis_client, _redis_tried
    if _redis_client is not None:
        return _redis_client
    if _redis_tried:
        return None
    _redis_tried = True
    url = os.getenv("MAYRING_REDIS_URL", "")
    if not url:
        return None
    try:
        import redis
        c = redis.Redis.from_url(
            url, decode_responses=True,
            socket_connect_timeout=0.5, socket_timeout=0.5,
        )
        c.ping()
        _redis_client = c
        return c
    except Exception:
        return None


def _stamp_file() -> Path:
    from mayring_core.config import _resolve_cache_dir, BASE_DIR
    return _resolve_cache_dir(BASE_DIR) / _FILE_NAME


def stamp(now: float | None = None) -> None:
    """Record the current wall-clock epoch as qwen's last text-route activity.

    Cheap: one Redis SET or one atomic file write per call, no lock. Never raises
    — a stamp failure must not break the generate it is observing.
    """
    ts = time.time() if now is None else now
    c = _get_redis()
    if c is not None:
        try:
            c.set(_REDIS_KEY, repr(ts))
            return
        except Exception:
            pass  # WHY: redis hiccup must not break the hot-path → fall through to file
    try:
        path = _stamp_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(repr(ts), encoding="utf-8")
        os.replace(tmp, path)  # atomic
    except OSError:
        pass  # WHY: best-effort observability stamp, never fail the caller


def last_activity() -> float | None:
    """Last text-route activity as epoch seconds, or None if never recorded.

    None means "no qwen activity seen" → the probe treats it as idle (safe).
    """
    c = _get_redis()
    if c is not None:
        try:
            raw = c.get(_REDIS_KEY)
            if raw is not None:
                return float(raw)
        except Exception:
            pass  # fall through to file
    try:
        return float(_stamp_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
