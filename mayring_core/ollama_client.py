"""Unified Ollama HTTP client + availability check.

Public API:
    generate(url, model, prompt, *, system, images, stream, timeout, ...) -> str
    embed_batch(url, model, texts, *, timeout) -> list[list[float]] | None
    embed_single(url, model, text, *, timeout, label, ...) -> list[float]
    chat(url, model, messages, *, system, tools, options, timeout) -> dict
    check_ollama(url) -> tuple[bool, list[str]]
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

import httpx

_SSL_VERIFY: bool = os.getenv("OLLAMA_SSL_VERIFY", "true").lower() not in ("false", "0", "no")

_GENERATE_MAX_RETRIES = 3
_GENERATE_RETRY_DELAYS = (2, 5, 10)

_EMBED_MAX_RETRIES = 5
_EMBED_RETRY_DELAYS = (3, 6, 12, 20, 30)

# WHY(#192, B+): Cloud-Fallback wenn lokales three.linn.games überlastet
# ist. OLLAMA_CLOUD_API_KEY in env (GH Actions secret OLLAMA_API_KEY_CLOUD)
# triggert den Pfad. CHANGE WITH CARE — Cloud-Calls kosten Geld, daher
# nur bei echten Connection/Timeout-Errors am letzten Retry, nicht bei
# 4xx-Client-Fehlern (die wären auf Cloud gleich kaputt).
_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com").rstrip("/")
_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")


def _default_keep_alive():
    """Default keep_alive for LOCAL embed/generate calls.

    WHY(VRAM-pin 2026-06-08): with the hot path consolidated to two small models
    (bge-m3 embed ~0.65GB + qwen3.5-mayring:2b ~2GB = ~2.7GB, proven to coexist),
    Ollama's default 5-min keep_alive still let each model evict the other under
    concurrent embed+generate load → every call cold-loaded (2-17s) → /memory/search
    + prompt-categorize blew the hook's 9s budget. Pinning both (keep_alive=-1) keeps
    them resident. Override via OLLAMA_KEEP_ALIVE (-1=forever, "30m", 0=unload now)."""
    v = os.getenv("OLLAMA_KEEP_ALIVE", "-1").strip()
    return -1 if v == "-1" else v


_KEEP_ALIVE = _default_keep_alive()

# WHY(2026-05-10): User-Auftrag — cloud-fallback war über wochen bei 0%
# usage weil cloud nur als RETRY-fallback bei local-error eingesprungen
# wäre. Die freie tier-quota soll aktiv genutzt werden. Anteil X% (default
# 20%) der calls geht primär direkt an cloud, mit local als reverse-
# fallback. Override via OLLAMA_CLOUD_PRIMARY_RATIO (0.0..1.0).
_CLOUD_PRIMARY_RATIO = max(0.0, min(1.0, float(
    os.getenv("OLLAMA_CLOUD_PRIMARY_RATIO", "0.2")
)))

# WHY(2026-05-10): ollama-cloud hat ein eigenes modell-set
# (glm-4.6, ministral-3:3b, gemma3:4b, qwen3-coder:480b, ...) — die
# local-modelle (mistral:7b-instruct, qwen2.5-coder:7b) gibts NICHT.
# /api/tags via cloud-API zeigt die liste; siehe ollama.com/cloud.
# Mapping: kleine/mittlere general-purpose → ministral-3:3b oder
# gemma3:4b (free-tier-tauglich); code-tasks → qwen3-coder-next.
# Override per env: OLLAMA_CLOUD_MODEL_MAP="local1:cloud1,local2:cloud2"
_CLOUD_MODEL_MAP_DEFAULT = {
    "mistral:7b-instruct": "gemma3:4b",
    "mistral:7b": "gemma3:4b",
    "mistral": "gemma3:4b",
    "qwen2.5-coder:7b": "qwen3-coder-next",
    "qwen2.5-coder": "qwen3-coder-next",
    "qwen3.5:2b": "ministral-3:3b",
    "qwen3:2b": "ministral-3:3b",
    "phi3:3.8b": "ministral-3:3b",
}


def _parse_cloud_map() -> dict[str, str]:
    raw = os.getenv("OLLAMA_CLOUD_MODEL_MAP", "")
    mapping = dict(_CLOUD_MODEL_MAP_DEFAULT)
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        # split nur am ersten ":" — model-names enthalten oft ":7b"
        local_part, cloud_part = pair.split(":", 1)
        if local_part and cloud_part:
            mapping[local_part.strip()] = cloud_part.strip()
    return mapping


_CLOUD_MODEL_MAP = _parse_cloud_map()
# Default cloud-modell wenn local-name nicht in map — gemma3:4b ist klein,
# schnell und für 90% der pi-task-prompts ausreichend.
_CLOUD_DEFAULT_MODEL = os.getenv("OLLAMA_CLOUD_DEFAULT_MODEL", "gemma3:4b")


def _resolve_cloud_model(local_model: str) -> str:
    """Map local model-name → cloud-equivalent. Falls unmapped: default."""
    if local_model in _CLOUD_MODEL_MAP:
        return _CLOUD_MODEL_MAP[local_model]
    # Family-prefix-lookup: "mistral:7b-instruct-q4" → versucht "mistral"
    family = local_model.split(":", 1)[0]
    if family in _CLOUD_MODEL_MAP:
        return _CLOUD_MODEL_MAP[family]
    return _CLOUD_DEFAULT_MODEL


def _should_route_cloud_primary() -> bool:
    """Per-call random sampling: returns True wenn diese call cloud-primär läuft.

    Hash-frei + thread-safe — `random.random()` reicht da der entscheidung-
    state nicht persistiert sein muss (anteil mittelt sich über N calls aus).
    Skip wenn kein API-key gesetzt.
    """
    if not _CLOUD_API_KEY or _CLOUD_PRIMARY_RATIO <= 0:
        return False
    import random as _rnd
    return _rnd.random() < _CLOUD_PRIMARY_RATIO


def _generate_call(host: str, body: dict, *, stream: bool, timeout: float,
                    auth_token: str = "") -> str:
    """One /api/generate-call against the given host. Auth-token optional
    (cloud needs it, local doesn't). Raises httpx-exceptions on failure
    so the caller can decide retry/fallback."""
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if stream:
        chunks: list[str] = []
        with httpx.stream(
            "POST", f"{host}/api/generate", json=body, headers=headers,
            timeout=timeout, verify=_SSL_VERIFY,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunks.append(data.get("response", ""))
                if data.get("done"):
                    break
        return "".join(chunks)
    resp = httpx.post(
        f"{host}/api/generate", json=body, headers=headers,
        timeout=timeout, verify=_SSL_VERIFY,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def generate(
    url: str,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    images: list[str] | None = None,
    stream: bool = True,
    timeout: float = 240.0,
    max_retries: int = _GENERATE_MAX_RETRIES,
    retry_delays: tuple[int, ...] = _GENERATE_RETRY_DELAYS,
    label: str = "",
    think: bool | None = None,
    num_predict: int = 4096,
    options: dict | None = None,
    keep_alive: str | None = None,
    response_format: str | None = None,
    cloud_primary: bool | None = None,
) -> str:
    """POST to /api/generate and return the complete response text.

    Uses stream=True by default to prevent read-timeout hangs on large models.
    Pass stream=False (e.g. for image captioning) to get a single-shot response.

    ``num_predict`` defaults to 4096 (not Ollama's 128). The first prod
    smoke showed thinking-capable models (qwen3, deepseek-r1) never closed
    ``</think>`` at 128 tokens, so ``response`` stayed empty. 4096 gives
    Mayring categorization room to reason AND emit the label line. Pass a
    lower value for cheap single-shot calls if you know the budget.

    ``think`` is left to the model's own default (``None`` means: don't send
    the field, let Ollama decide). Callers can force it (True/False) per call
    — e.g. image captioning with stream=False might opt out explicitly.
    """
    base = url.rstrip("/")
    body: dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
    if think is not None:
        body["think"] = think
    if system:
        body["system"] = system
    if images:
        body["images"] = images
    # Pin the model by default (env OLLAMA_KEEP_ALIVE, -1=forever) so concurrent
    # embed+generate don't evict each other; explicit caller value still wins.
    body["keep_alive"] = keep_alive if keep_alive is not None else _KEEP_ALIVE

    merged_options: dict[str, Any] = {"num_predict": num_predict}
    if options:
        merged_options.update(options)
    body["options"] = merged_options

    # WHY(2026-05-28 central-queue): `format` (e.g. "json") is a TOP-LEVEL
    # Ollama field, not an option — set before the cloud_body copy so JSON-mode
    # holds on both the cloud-primary and local paths. Lets the queue-routed
    # pi_* tools request structured output without a direct Ollama call.
    if response_format:
        body["format"] = response_format

    # WHY(2026-05-10 cloud-primary-routing): X% der calls direkt an cloud
    # (free-tier quota nutzen) — wenn cloud fail't, fallback auf local.
    # Reduziert local-overload UND nutzt die kostenlose cloud-quota.
    # cloud_primary=False forces a latency-critical hot-path call (e.g. the inline search
    # advisor) to stay LOCAL — the 50% cloud-split routes to a slower cloud model
    # (gemma3:4b) and added 4-8s of variance per /memory/search, blowing the hook budget.
    if cloud_primary is not False and _should_route_cloud_primary():
        try:
            cloud_model = _resolve_cloud_model(model)
            cloud_body = dict(body, model=cloud_model)
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "ollama generate cloud-primary: %s → %s (%.0f%% ratio)",
                model, cloud_model, _CLOUD_PRIMARY_RATIO * 100,
            )
            return _generate_call(
                _CLOUD_URL, cloud_body, stream=stream, timeout=timeout,
                auth_token=_CLOUD_API_KEY,
            )
        except Exception as cloud_exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "cloud-primary failed (%s), fallback local",
                type(cloud_exc).__name__,
            )
            # weiter in den normalen local-pfad fallen

    for attempt in range(max_retries):
        try:
            return _generate_call(base, body, stream=stream, timeout=timeout)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            # WHY(#192, B+): am letzten Retry → Cloud-Fallback wenn API-Key
            # gesetzt. ConnectError/Timeout heißt local Ollama ist down
            # oder überlastet; Cloud ist eine teurere aber verfügbare
            # alternative. 4xx/5xx-Errors überspringen wir (4xx = Bug,
            # 5xx-overload-Pattern wird unten behandelt).
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _log_retry("generate", label, attempt + 1, max_retries, delay)
                time.sleep(delay)
                continue
            if _CLOUD_API_KEY and _CLOUD_URL != base:
                try:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "ollama local→cloud fallback: %s", type(exc).__name__,
                    )
                    return _generate_call(
                        _CLOUD_URL, body, stream=stream, timeout=timeout,
                        auth_token=_CLOUD_API_KEY,
                    )
                except Exception:
                    raise exc  # cloud auch broken → originaler Fehler
            raise
        except httpx.HTTPStatusError as exc:
            # 503 (overload) / 429 (rate-limit) auf der LETZTEN attempt → Cloud-fallback
            status = exc.response.status_code if exc.response is not None else 0
            if status in (503, 529, 429) and attempt == max_retries - 1 and \
               _CLOUD_API_KEY and _CLOUD_URL != base:
                try:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "ollama local→cloud fallback: HTTP %s", status,
                    )
                    return _generate_call(
                        _CLOUD_URL, body, stream=stream, timeout=timeout,
                        auth_token=_CLOUD_API_KEY,
                    )
                except Exception:
                    raise exc
            raise
    raise RuntimeError("unreachable")


# WHY(2026-05-10 embed-cloud-immunity): _should_route_cloud_primary()
# wird hier BEWUSST NICHT aufgerufen. Ollama-cloud (ollama.com) hat keine
# embedding-modelle wie nomic-embed-text — ein cloud-route würde 404
# "model not found" zurückgeben. Falls eine zukünftige refactor-runde
# das "cloud für alles"-routing erweitern will: embeds MÜSSEN separates
# routing/skipping bekommen, niemals dieselbe random-ratio. Verify mit
# `tests/test_ollama_cloud_routing.py::test_embed_does_not_route_cloud`.
def embed_batch(
    url: str,
    model: str,
    texts: list[str],
    *,
    timeout: float = 240.0,
) -> list[list[float]] | None:
    """POST to /api/embed (batch endpoint).

    Returns a list of float vectors in the same order as *texts*, or None if
    the endpoint is unavailable or returns an unexpected shape.
    """
    try:
        resp = httpx.post(
            f"{url.rstrip('/')}/api/embed",
            json={"model": model, "input": texts, "keep_alive": _KEEP_ALIVE},
            timeout=timeout,
            verify=_SSL_VERIFY,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("embeddings")
        if isinstance(result, list) and len(result) == len(texts):
            return result
        print(f"[embed_batch] unexpected response shape: {str(data)[:200]}", flush=True)
    except httpx.ConnectError as exc:
        print(f"[embed_batch] ConnectError → {url}: {exc}", flush=True)
    except httpx.HTTPStatusError as exc:
        print(f"[embed_batch] HTTP {exc.response.status_code} from {url}: {exc.response.text[:200]}", flush=True)
    except httpx.TimeoutException:
        print(f"[embed_batch] Timeout after {timeout}s → {url}", flush=True)
    except Exception as exc:
        print(f"[embed_batch] {type(exc).__name__}: {exc}", flush=True)
    return None


def embed_single(
    url: str,
    model: str,
    text: str,
    *,
    timeout: float = 240.0,
    label: str = "",
    max_retries: int = _EMBED_MAX_RETRIES,
    retry_delays: tuple[int, ...] = _EMBED_RETRY_DELAYS,
) -> list[float]:
    """POST to /api/embeddings (legacy single-text endpoint) with retry."""
    base = url.rstrip("/")
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{base}/api/embeddings",
                json={"model": model, "prompt": text, "keep_alive": _KEEP_ALIVE},
                timeout=timeout,
                verify=_SSL_VERIFY,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _log_retry("embed", label, attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")


def _chat_call(host: str, body: dict, *, timeout: float,
               auth_token: str = "") -> dict:
    """Single /api/chat POST gegen den gegebenen host. Auth optional."""
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    resp = httpx.post(
        f"{host}/api/chat", json=body, headers=headers,
        timeout=timeout, verify=_SSL_VERIFY,
    )
    resp.raise_for_status()
    return resp.json()


def chat(
    url: str,
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    options: dict | None = None,
    stream: bool = False,
    timeout: float = 120.0,
    keep_alive: str | None = None,
) -> dict:
    """POST to /api/chat and return the parsed JSON response dict.

    Routes X% (env OLLAMA_CLOUD_PRIMARY_RATIO, default 0.2) primär an
    cloud — matched routing-logik in generate(). Pi-Agent nutzt chat() für
    tool-calling, weshalb cloud-quota sonst ungenutzt bliebe.

    Raises httpx exceptions on failure — callers handle retries/errors.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": False,
    }
    if system is not None:
        body["system"] = system
    if tools is not None:
        body["tools"] = tools
    if options is not None:
        body["options"] = options
    # Pin the model by default (env OLLAMA_KEEP_ALIVE, -1=forever) so concurrent
    # embed+generate don't evict each other; explicit caller value still wins.
    body["keep_alive"] = keep_alive if keep_alive is not None else _KEEP_ALIVE

    base = url.rstrip("/")
    # Cloud-primary X% mit local als reverse-fallback (gleicher anteil wie
    # generate() — siehe _should_route_cloud_primary docstring).
    if _should_route_cloud_primary():
        try:
            cloud_model = _resolve_cloud_model(model)
            cloud_body = dict(body, model=cloud_model)
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "ollama chat cloud-primary: %s → %s (%.0f%% ratio)",
                model, cloud_model, _CLOUD_PRIMARY_RATIO * 100,
            )
            return _chat_call(_CLOUD_URL, cloud_body, timeout=timeout,
                              auth_token=_CLOUD_API_KEY)
        except Exception as cloud_exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "chat cloud-primary failed (%s), fallback local",
                type(cloud_exc).__name__,
            )

    return _chat_call(base, body, timeout=timeout)


def check_ollama(url: str) -> tuple[bool, list[str]]:
    """Return (reachable, model_list). Never raises. Tries subprocess then HTTP."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            models = [l.split()[0] for l in lines[1:] if l.split()]
            return True, models
    except Exception:
        pass
    try:
        resp = httpx.get(url.rstrip("/") + "/api/tags", timeout=3.0, verify=_SSL_VERIFY)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            return True, models
    except Exception:
        pass
    return False, []


def _log_retry(kind: str, label: str, attempt: int, max_retries: int, delay: int) -> None:
    tag = f" [{label}]" if label else ""
    print(f"    ⟳ {kind}-Retry {attempt}/{max_retries}{tag} (in {delay}s) …", flush=True)
