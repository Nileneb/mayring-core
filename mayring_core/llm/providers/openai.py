"""OpenAI Chat Completions client (non-streaming)."""
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
    """POST to /v1/chat/completions and return the message content."""
    url = f"{endpoint.base_url.rstrip('/')}/v1/chat/completions"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": endpoint.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    resp = httpx.post(url, json=body, headers=endpoint.headers(), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""
