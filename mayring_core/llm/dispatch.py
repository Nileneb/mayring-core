"""Provider-agnostic generate() dispatcher.

    generate(endpoint, prompt, system=..., stream=..., ...) -> str

Routes to the right backend:
    ollama / platform  → mayring_core.ollama_client.generate
    anthropic          → mayring_core.llm.providers.anthropic.generate
    openai             → mayring_core.llm.providers.openai.generate
"""
from __future__ import annotations

from mayring_core.llm.endpoint import LLMEndpoint


def generate(
    endpoint: LLMEndpoint,
    prompt: str,
    *,
    system: str | None = None,
    stream: bool = True,
    timeout: float = 240.0,
    label: str = "",
    max_tokens: int = 4096,
) -> str:
    """Generate text using the endpoint's configured backend."""
    if endpoint.provider in ("ollama", "platform"):
        from mayring_core.ollama_client import generate as _ollama_generate
        return _ollama_generate(
            endpoint.base_url, endpoint.model, prompt,
            system=system, stream=stream, timeout=timeout, label=label,
        )
    if endpoint.provider == "anthropic":
        from mayring_core.llm.providers.anthropic import generate as _anth_generate
        return _anth_generate(
            endpoint, prompt, system=system, max_tokens=max_tokens, timeout=timeout,
        )
    if endpoint.provider == "openai":
        from mayring_core.llm.providers.openai import generate as _oai_generate
        return _oai_generate(
            endpoint, prompt, system=system, max_tokens=max_tokens, timeout=timeout,
        )
    raise ValueError(f"Unknown provider: {endpoint.provider!r}")
