"""Centralized model routing for MayringCoder tasks.

Maps task names to Ollama model names with fallback and availability caching.
Config loaded from config/model_routes.yaml (optional — silent defaults if missing).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from mayring_core.config import BASE_DIR as _ROOT  # depth-robust repo-root resolution (#267/#270)
_CONFIG_PATH = _ROOT / "config" / "model_routes.yaml"

# Model names live ONLY in config/model_routes.yaml (or the central model-config) —
# NEVER hardcoded here. A model literal goes stale on the next model swap and silently
# overrides the config (that was the nomic-embed VRAM bug: a leftover 768d default
# beside the live bge-m3 store). _DEFAULTS carries STRUCTURE/timeouts only; resolve()
# returns "" for a task with no configured model and embedding callers fail loud.
_DEFAULTS: dict[str, dict] = {
    "text":      {"model": "", "fallback": "", "timeout": 240},
    "vision":    {"model": "", "fallback": "", "timeout": 120},
    "embedding": {"model": "", "fallback": "", "timeout": 60},
}

# Zentrale Modell-Konfig (app.linn.games /api/mcp/model-config) — Single-Source-of-
# Truth für alle MCP-Server. Opt-in per MAYRING_MODEL_CONFIG_URL; ohne env null
# Verhaltensänderung (Fallback auf text_model.txt → model_routes.yaml). Fail-soft.
_CENTRAL_CACHE: dict[str, tuple[dict, float]] = {}
_CENTRAL_TTL = 60.0


def _fetch_central_config(server: str) -> dict:
    """{task: {model, options}} vom zentralen Endpoint, 60s gecached. {} bei
    fehlendem env/Fehler — der Consumer fällt dann transparent auf seine lokale
    Auflösung zurück (text_model.txt → yaml). Niemals blockieren."""
    base = os.getenv("MAYRING_MODEL_CONFIG_URL", "").rstrip("/")
    if not base or _httpx is None:
        return {}
    now = time.monotonic()
    cached = _CENTRAL_CACHE.get(server)
    if cached is not None and now - cached[1] < _CENTRAL_TTL:
        return cached[0]
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = _httpx.get(
            f"{base}/api/mcp/model-config",
            params={"server": server}, headers=headers, timeout=3.0,
        )
        cfg = (resp.json() or {}).get("config", {}) or {} if resp.status_code == 200 else {}
    except Exception:
        cfg = {}
    _CENTRAL_CACHE[server] = (cfg, now)
    return cfg


@dataclass
class RouteConfig:
    model: str
    fallback: str = ""
    timeout: int = 240
    # WHY(#183, T4): per-job-class overrides für model + timeout. mini (kleine
    # Tasks <500 chars) routet auf günstigere Modelle bei kürzerem timeout;
    # standard nutzt den outer route. Unbekannte job_class fällt auf outer.
    classes: dict[str, "RouteConfig"] = field(default_factory=dict)


class ModelRouter:
    """Routes MayringCoder tasks to specific Ollama models.

    Usage:
        router = ModelRouter(ollama_url="http://localhost:11434")
        model = router.resolve("text")      # all Mayring categorization + analysis
        if router.is_available("vision"):
            caption = caption_image(path, router.resolve("vision"))

    Tasks are intentionally minimal: text/vision/embedding cover all real model
    interface differences. Mayring mode (deduktiv/induktiv/hybrid) is a prompt-
    template choice, not a model choice — the codebook drives what gets labeled,
    the model just executes the categorization.
    """

    TASKS: list[str] = ["text", "vision", "embedding"]

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self._ollama_url = ollama_url.rstrip("/")
        # Name dieses Consumers im zentralen Config-Schema (server-Spalte).
        self._central_server = os.getenv("MAYRING_MODEL_CONFIG_SERVER", "mayringcoder")
        self._routes: dict[str, RouteConfig] = {
            task: RouteConfig(**cfg) for task, cfg in _DEFAULTS.items()
        }
        self._availability_cache: dict[str, tuple[bool, float]] = {}  # model → (available, timestamp)
        self._cache_ttl = 30.0  # seconds

        if _CONFIG_PATH.exists():
            self.load_config(_CONFIG_PATH)

    def load_config(self, path: Path) -> None:
        """Load routes from YAML file. Unknown tasks are ignored."""
        if not _HAS_YAML:
            return
        try:
            with path.open(encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
            for task, cfg in data.items():
                if not isinstance(cfg, dict):
                    continue
                existing = self._routes.get(task)
                if existing:
                    if "model" in cfg:
                        existing.model = str(cfg["model"])
                    if "fallback" in cfg:
                        existing.fallback = str(cfg["fallback"])
                    if "timeout" in cfg:
                        existing.timeout = int(cfg["timeout"])
                else:
                    existing = RouteConfig(
                        model=str(cfg.get("model", "")),
                        fallback=str(cfg.get("fallback", "")),
                        timeout=int(cfg.get("timeout", 240)),
                    )
                    self._routes[task] = existing
                # T4: per-class overrides
                classes_cfg = cfg.get("classes") or {}
                if isinstance(classes_cfg, dict):
                    existing.classes = {
                        cls: RouteConfig(
                            model=str(c.get("model", existing.model)),
                            fallback=str(c.get("fallback", existing.fallback)),
                            timeout=int(c.get("timeout", existing.timeout)),
                        )
                        for cls, c in classes_cfg.items()
                        if isinstance(c, dict)
                    }
        except Exception as e:
            import logging
            logging.warning("model_routes.yaml konnte nicht geladen werden: %s — Defaults aktiv", e)

    def save_config(self, path: Path | None = None) -> None:
        """Persist current routes to YAML."""
        target = path or _CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _HAS_YAML:
            return
        data = {
            task: {"model": cfg.model, "fallback": cfg.fallback, "timeout": cfg.timeout}
            for task, cfg in self._routes.items()
        }
        with target.open("w", encoding="utf-8") as f:
            _yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def _read_text_override() -> str:
        """Read CACHE_DIR/text_model.txt for a runtime model override.

        # WHY(2026-06-06): text_model.txt ermöglicht Laufzeit-Wechsel des
        # Text-Modells ohne YAML-Edit oder Neustart. Cache-Dir wird per-Call
        # frisch aufgelöst (nicht Modul-Level-Konstante) damit MAYRING_CACHE_DIR-
        # env in Tests und Prod gleich wirkt — identisch zu reranker_v2._cache_dir()
        # die ebenfalls _resolve_cache_dir() zur Laufzeit aufruft.
        """
        try:
            from mayring_core.config import _resolve_cache_dir, BASE_DIR
            cache_dir = _resolve_cache_dir(BASE_DIR)
            override = (cache_dir / "text_model.txt").read_text(encoding="utf-8").strip()
            return override
        except (FileNotFoundError, OSError):
            return ""

    def resolve(self, task: str, job_class: str | None = None) -> str:
        """Return model name for task, optional job_class override.

        Priority: zentrale Config (app.linn.games) → text_model.txt →
        config/model_routes.yaml::classes[job_class] → outer route → hardcoded
        default → "". Unknown job_class silently falls back to outer (no error —
        opaque per spec #183).

        Never reads OLLAMA_MODEL env — change models via zentrale UI, Web-UI oder
        model_routes.yaml.
        """
        central = _fetch_central_config(self._central_server).get(task) or {}
        if central.get("model"):
            return str(central["model"])
        if task == "text":
            override = self._read_text_override()
            if override:
                return override
        cfg = self._routes.get(task)
        if cfg is None:
            return ""
        if job_class and cfg.classes and job_class in cfg.classes:
            cls_cfg = cfg.classes[job_class]
            return cls_cfg.model or cls_cfg.fallback or cfg.model or cfg.fallback or ""
        return cfg.model or cfg.fallback or ""

    def timeout_for(self, task: str, job_class: str | None = None) -> int:
        """Return timeout (seconds) for task + optional job_class.

        Issue #183 T4: mini-Klasse hat strengeren timeout (z.B. 30s) damit
        kleine Tasks nicht 240s auf langsame standard-Modelle warten.
        """
        cfg = self._routes.get(task)
        if cfg is None:
            return 240
        if job_class and cfg.classes and job_class in cfg.classes:
            return cfg.classes[job_class].timeout
        return cfg.timeout

    def options_for(self, task: str) -> dict:
        """Zentrale Generate-Optionen für task ({think, num_ctx, …}), leer wenn
        keine zentrale Config. Caller reichen das an ollama_client.generate durch —
        so wird z.B. think=False für categorize zentral steuerbar."""
        return _fetch_central_config(self._central_server).get(task, {}).get("options") or {}

    def resolve_with_fallback(self, task: str) -> str:
        """Return model name, falling back to fallback if primary not available."""
        cfg = self._routes.get(task)
        if cfg is None:
            return ""

        primary = cfg.model or cfg.fallback or ""
        if primary and self._check_model_available(primary):
            return primary
        return cfg.fallback or ""

    def is_available(self, task: str) -> bool:
        """True if the model for this task is available in Ollama (cached 30s TTL).

        Returns False if:
        - task has no model configured
        - model name is empty string
        - httpx not installed
        - Ollama not reachable
        - model not in Ollama model list
        """
        cfg = self._routes.get(task)
        if cfg is None:
            return False
        model = cfg.model or cfg.fallback or ""
        if not model:
            return False
        return self._check_model_available(model)

    def _check_model_available(self, model: str) -> bool:
        """Check if model is in Ollama, with 30s cache."""
        now = time.monotonic()
        cached = self._availability_cache.get(model)
        if cached is not None:
            available, ts = cached
            if now - ts < self._cache_ttl:
                return available

        available = self._query_ollama_for_model(model)
        self._availability_cache[model] = (available, now)
        return available

    def _query_ollama_for_model(self, model: str) -> bool:
        """GET /api/tags and check if model is present."""
        if _httpx is None:
            return False
        try:
            resp = _httpx.get(f"{self._ollama_url}/api/tags", timeout=2.0)
            if resp.status_code != 200:
                return False
            tags = resp.json()
            names = {m["name"] for m in tags.get("models", [])}
            # Match exact name or name without tag
            model_base = model.split(":")[0]
            return model in names or any(
                n == model or n.split(":")[0] == model_base for n in names
            )
        except Exception:
            return False

    def set_route(self, task: str, model: str, fallback: str = "", timeout: int = 240) -> None:
        """Update a route at runtime (and invalidate availability cache)."""
        self._routes[task] = RouteConfig(model=model, fallback=fallback, timeout=timeout)
        # Invalidate cache entries for old+new model
        self._availability_cache.clear()

    def to_dict(self) -> dict[str, dict]:
        """Serialize routes for WebUI or JSON responses."""
        return {
            task: {
                "model": cfg.model,
                "fallback": cfg.fallback,
                "timeout": cfg.timeout,
                "available": self._availability_cache.get(cfg.model, (None, 0))[0],
            }
            for task, cfg in self._routes.items()
        }

    def __repr__(self) -> str:
        routes_str = ", ".join(f"{t}={cfg.model!r}" for t, cfg in self._routes.items())
        return f"ModelRouter({routes_str})"
