"""Provider registry — the dependency-injection boundary for mayring_core.

mayring_core must not import from non-core packages (``src.analysis``,
``src.agents``) at runtime. The richer host implementations are injected at boot
via ``register_*()``:

* embedder  → ``src/analysis/context_rag.py::_embed_texts`` (cached + batched)
* generator → ``src/analysis/analyzer.py::_ollama_generate`` (Ollama streaming)
* vision    → ``src/agents/vision.py::caption_image`` / ``get_image_metadata``

Standalone (core-only) installs that never call ``register_*`` fall back to thin
defaults built directly on ``mayring_core.ollama_client`` for embed/generate.
Vision has no safe core default (needs Pillow + the captioner), so it raises
``ProviderNotConfigured`` until registered.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

EmbedFn = Callable[[list[str], str], list[list[float]]]
GenerateFn = Callable[..., str]
VisionCaptionFn = Callable[..., str]
VisionMetadataFn = Callable[[Path], Optional[dict]]


class ProviderNotConfigured(RuntimeError):
    """Raised when a provider without a safe default is used before registration."""


_embed_fn: EmbedFn | None = None
_generate_fn: GenerateFn | None = None
_vision_caption_fn: VisionCaptionFn | None = None
_vision_metadata_fn: VisionMetadataFn | None = None


def register_embedder(fn: EmbedFn) -> None:
    global _embed_fn
    _embed_fn = fn


def register_generator(fn: GenerateFn) -> None:
    global _generate_fn
    _generate_fn = fn


def register_vision(caption_fn: VisionCaptionFn, metadata_fn: VisionMetadataFn) -> None:
    global _vision_caption_fn, _vision_metadata_fn
    _vision_caption_fn = caption_fn
    _vision_metadata_fn = metadata_fn


def reset_providers() -> None:
    """Test helper — drop all registrations so defaults take over again."""
    global _embed_fn, _generate_fn, _vision_caption_fn, _vision_metadata_fn
    _embed_fn = _generate_fn = _vision_caption_fn = _vision_metadata_fn = None


# --- standalone-safe defaults on core ollama_client -------------------------

def _default_embed(texts: list[str], ollama_url: str) -> list[list[float]]:
    from mayring_core.config import EMBEDDING_MODEL, OLLAMA_EMBED_TIMEOUT
    from mayring_core.ollama_client import embed_batch, embed_single

    out = embed_batch(ollama_url, EMBEDDING_MODEL, texts, timeout=OLLAMA_EMBED_TIMEOUT)
    if out is not None:
        return out
    return [
        embed_single(ollama_url, EMBEDDING_MODEL, t, timeout=OLLAMA_EMBED_TIMEOUT)
        for t in texts
    ]


def _default_generate(
    prompt: str,
    ollama_url: str,
    model: str,
    label: str,
    *,
    system_prompt: str | None = None,
    keep_alive: str | None = None,
    options: dict | None = None,
    response_format: str | None = None,
    num_predict: int = 4096,
    think: bool | None = None,
) -> str:
    from mayring_core.config import OLLAMA_TIMEOUT
    from mayring_core.ollama_client import generate

    return generate(
        ollama_url, model, prompt,
        system=system_prompt,
        stream=True,
        timeout=OLLAMA_TIMEOUT,
        label=label,
        keep_alive=keep_alive,
        options=options,
        response_format=response_format,
        num_predict=num_predict,
        think=think,
    )


# --- public accessors used by mayring_core.memory --------------------------
# Transparent passthrough: callers' args flow verbatim to the registered impl
# (or the default), so the canonical function — and any test mock standing in
# for it — sees exactly the arguments the caller passed, with no injected kwargs.

def embed_texts(*args, **kwargs) -> list[list[float]]:
    return (_embed_fn or _default_embed)(*args, **kwargs)


def generate_text(*args, **kwargs) -> str:
    return (_generate_fn or _default_generate)(*args, **kwargs)


def vision_caption(*args, **kwargs) -> str:
    if _vision_caption_fn is None:
        raise ProviderNotConfigured(
            "vision provider not registered; call mayring_core.providers.register_vision()"
        )
    return _vision_caption_fn(*args, **kwargs)


def vision_metadata(*args, **kwargs) -> Optional[dict]:
    if _vision_metadata_fn is None:
        raise ProviderNotConfigured(
            "vision provider not registered; call mayring_core.providers.register_vision()"
        )
    return _vision_metadata_fn(*args, **kwargs)
