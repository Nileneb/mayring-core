"""LLMEndpoint: per-workspace model config fetched from app.linn.games.

Each workspace can point at its own backend — own local Ollama (LAN/Tailscale),
own Anthropic/OpenAI API key, or fall through to platform-managed default.

Laravel side serves GET /api/mcp-service/llm-endpoint/{workspace_id} (auth
via MCP_SERVICE_TOKEN). See docs/laravel_llm_endpoints_spec.md for the
Laravel implementation contract.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx

if TYPE_CHECKING:
    from mayring_core.identity.types import TokenInfoLike

Provider = Literal["ollama", "anthropic", "openai", "platform"]

# JWT llm_provider-Claim → dispatcher provider mapping.
_JWT_PROVIDER_MAP: dict[str, Provider] = {
    "platform": "platform",
    "anthropic-byo": "anthropic",
    "openai-compatible": "openai",
}


@dataclass(frozen=True)
class LLMEndpoint:
    provider: Provider
    base_url: str
    model: str
    api_key: str | None = None
    # Provider-specific extras the dispatcher may use (e.g. organization id)
    extra: tuple[tuple[str, str], ...] = ()

    def headers(self) -> dict[str, str]:
        if self.provider == "anthropic" and self.api_key:
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        if self.provider == "openai" and self.api_key:
            return {
                "authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            }
        return {"content-type": "application/json"}


_CACHE_TTL_SECONDS = int(os.getenv("LLM_ENDPOINT_CACHE_TTL", "300"))
_cache: dict[str, tuple[float, LLMEndpoint]] = {}
_cache_lock = threading.Lock()


def _platform_default() -> LLMEndpoint:
    from mayring_core.model_router import ModelRouter
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    return LLMEndpoint(
        provider="platform",
        base_url=ollama_url,
        model=ModelRouter(ollama_url).resolve("text"),
    )


def _service_base_url() -> str:
    return os.getenv("LARAVEL_INTERNAL_URL", "http://web").rstrip("/")


def _service_token() -> str:
    return os.getenv("MCP_SERVICE_TOKEN", "")


def fetch_endpoint(workspace_id: str, *, timeout: float = 3.0) -> LLMEndpoint | None:
    """Fetch endpoint config from Laravel; None on any failure → caller falls back."""
    token = _service_token()
    if not token or not workspace_id:
        return None
    url = f"{_service_base_url()}/api/mcp-service/llm-endpoint/{workspace_id}"
    try:
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    provider = data.get("provider")
    if provider not in ("ollama", "anthropic", "openai", "platform"):
        return None
    base_url = str(data.get("base_url") or "").rstrip("/")
    model = str(data.get("model") or "").strip()
    api_key = data.get("api_key")
    if not base_url or not model:
        return None
    extra_in = data.get("extra") or {}
    extra = tuple((str(k), str(v)) for k, v in extra_in.items()) if isinstance(extra_in, dict) else ()
    return LLMEndpoint(
        provider=provider, base_url=base_url, model=model,
        api_key=(api_key if isinstance(api_key, str) and api_key else None),
        extra=extra,
    )


def get_llm_endpoint(workspace_id: str | None) -> LLMEndpoint:
    """Return cached endpoint config for workspace, or platform default on miss."""
    if not workspace_id:
        return _platform_default()
    now = time.time()
    with _cache_lock:
        cached = _cache.get(workspace_id)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]
    endpoint = fetch_endpoint(workspace_id) or _platform_default()
    with _cache_lock:
        _cache[workspace_id] = (now, endpoint)
    return endpoint


def invalidate_cache(workspace_id: str | None = None) -> None:
    """Drop cached endpoint for one workspace, or all if None."""
    with _cache_lock:
        if workspace_id is None:
            _cache.clear()
        else:
            _cache.pop(workspace_id, None)


def _endpoint_from_claims(
    token_info: "TokenInfoLike",
    user_jwt: str | None,
) -> LLMEndpoint:
    """Build an LLMEndpoint from JWT provider-claims. May call key_callback."""
    from mayring_core.llm.key_callback import LlmKeyUnavailableError, fetch_user_key

    provider = _JWT_PROVIDER_MAP.get(token_info.llm_provider, "platform")
    model = (token_info.llm_model or "").strip()
    base_url = (token_info.llm_endpoint or "").strip().rstrip("/")

    api_key: str | None = None
    if token_info.llm_requires_key:
        if not user_jwt or token_info.sub is None or token_info.iat is None:
            raise LlmKeyUnavailableError(
                "JWT declares llm_requires_key=true but sub/iat/raw-jwt are missing."
            )
        api_key = fetch_user_key(token_info.sub, token_info.iat, user_jwt)
        if not api_key:
            raise LlmKeyUnavailableError(
                f"User {token_info.sub} has no api_key configured — "
                "set one in /settings/ai-model or switch provider back to platform."
            )

    # anthropic-byo has no user-supplied base_url → fall back to api.anthropic.com.
    if provider == "anthropic" and not base_url:
        base_url = "https://api.anthropic.com"
    # openai-compatible requires an endpoint — without one, we can't route anywhere.
    if provider == "openai" and not base_url:
        raise LlmKeyUnavailableError(
            "openai-compatible provider requires llm_endpoint in JWT but none set."
        )

    if not model:
        raise LlmKeyUnavailableError(
            f"No model configured for provider={provider}. "
            "Set llm_custom_model in /settings/ai-model."
        )

    return LLMEndpoint(provider=provider, base_url=base_url, model=model, api_key=api_key)


def get_endpoint_for_request(
    token_info: "TokenInfoLike | None",
    workspace_id: str | None,
    user_jwt: str | None = None,
) -> LLMEndpoint:
    """Resolve the right LLMEndpoint for a request.

    Precedence:
      1. User-JWT-Claims (per-User BYO-Provider via /settings/ai-model)
      2. Workspace-LLM-Endpoints (Admin-Override via /einstellungen/llm-endpoints)
      3. Platform default (OLLAMA_URL, Modell via ModelRouter)

    Raises:
      LlmKeyUnavailableError — wenn der User einen Key hinterlegen müsste, aber
      keiner geliefert werden kann. Hardfail statt Silent-Fallback (Entscheidung
      des Produktteams 2026-04-21: silent billing-fallback ist kontraintuitiv).
    """
    if token_info is not None and token_info.uses_custom_provider:
        return _endpoint_from_claims(token_info, user_jwt)
    return get_llm_endpoint(workspace_id)
