"""Anthropic Messages API client (non-streaming)."""
from __future__ import annotations

from typing import Any

import httpx

from mayring_core.llm.endpoint import LLMEndpoint


def generate(
    endpoint: LLMEndpoint,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    timeout: float = 240.0,
) -> str:
    """POST to /v1/messages and return the combined text output."""
    url = f"{endpoint.base_url.rstrip('/')}/v1/messages"
    body: dict[str, Any] = {
        "model": endpoint.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    resp = httpx.post(url, json=body, headers=endpoint.headers(), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    parts = data.get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
