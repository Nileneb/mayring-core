"""User-API-Key-Callback zu app.linn.games.

Wenn ein User-JWT die Claim `llm_requires_key=true` trägt, müssen die Worker
den entschlüsselten `llm_api_key` vom Laravel-Endpoint abrufen, bevor sie den
Provider ansprechen.

Laravel-Endpoint: POST /api/mcp/user-llm-key
  Outer-Auth: Bearer MCP_SERVICE_TOKEN (service_only-Mode)
  Body:       {"jwt": "<user-jwt>"}
  Response:   {"api_key": "...", "provider": "..."} oder 404

Cache-Key: (sub, iat). Neues JWT durch /mayring/refresh-token → neues iat →
automatischer Cache-Miss. TTL = JWT_TTL_SECONDS minus 60s Puffer, damit der
Cache nie länger lebt als das JWT selbst.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import httpx

_log = logging.getLogger(__name__)

_key_cache: dict[tuple[str, int], tuple[float, str]] = {}
_key_lock = threading.Lock()


class LlmKeyUnavailableError(RuntimeError):
    """Raised when a user requires a key but the callback can't deliver one."""


def _cache_ttl_seconds() -> float:
    try:
        ttl = float(os.getenv("JWT_TTL_SECONDS", "28800"))
    except ValueError:
        ttl = 28800.0
    return max(0.0, ttl - 60.0)


def _laravel_base_url() -> str:
    return os.getenv("LARAVEL_INTERNAL_URL", "http://web").rstrip("/")


def _service_token() -> str:
    return os.getenv("MCP_SERVICE_TOKEN", "")


def fetch_user_key(
    sub: str,
    iat: int,
    user_jwt: str,
    *,
    timeout: float = 3.0,
) -> str | None:
    """Holt den entschlüsselten API-Key. None bei 404/Fehler, Wert bei Erfolg.

    Caller entscheidet über Hardfail vs. Fallback — diese Funktion wirft keine
    Exceptions außer bei Argument-Fehlern.
    """
    if not sub or iat is None or not user_jwt:
        return None
    service_token = _service_token()
    if not service_token:
        _log.warning("fetch_user_key: MCP_SERVICE_TOKEN not configured")
        return None

    cache_key = (sub, int(iat))
    now = time.time()
    ttl = _cache_ttl_seconds()

    with _key_lock:
        cached = _key_cache.get(cache_key)
        if cached and (now - cached[0]) < ttl:
            return cached[1]

    url = f"{_laravel_base_url()}/api/mcp/user-llm-key"
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {service_token}",
                "Content-Type": "application/json",
            },
            json={"jwt": user_jwt},
            timeout=timeout,
        )
    except Exception as exc:
        _log.warning("fetch_user_key: request failed: %s", exc)
        return None

    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        _log.warning("fetch_user_key: unexpected status %s: %s", resp.status_code, resp.text[:200])
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    api_key = data.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        return None

    with _key_lock:
        _key_cache[cache_key] = (now, api_key)
    return api_key


def invalidate_cache(sub: str | None = None) -> None:
    """Drop cached keys for a user, or all if sub is None."""
    with _key_lock:
        if sub is None:
            _key_cache.clear()
            return
        for key in [k for k in _key_cache if k[0] == sub]:
            _key_cache.pop(key, None)
