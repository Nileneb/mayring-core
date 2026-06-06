"""Central configuration and token-budget constants."""

import os
import re
from pathlib import Path
from urllib.parse import urlparse


def _resolve_base_dir() -> Path:
    """Repo root of the consumer (MayringCoder) — the dir holding codebooks/,
    prompts/, config/, wiki/. mayring_core is vendored at varying depths
    (src/ → core/ → vendor/mayring-core/ across #267/#270); a fixed __file__
    depth broke on every relocation. Anchor to marker dirs instead.
    MAYRING_BASE_DIR overrides for standalone installs where these dirs are absent."""
    env = os.getenv("MAYRING_BASE_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "codebooks").is_dir() and (parent / "prompts").is_dir():
            return parent
    return here.parent.parent.parent  # fallback: historical core/mayring_core/ layout


BASE_DIR = _resolve_base_dir()


def _resolve_cache_dir(base: Path) -> Path:
    """Where memory.db / wiki_v2.db / chroma live.

    MAYRING_CACHE_DIR overrides. In a Claude Code *plugin* install BASE_DIR sits
    under .claude/plugins/ and is re-versioned + wiped on every `/plugin update`,
    which orphans the DB and splits it from CLI/workflow runs. Persist at the user
    level there so every client entrypoint (MCP server, CLI, ingest-workflows)
    shares ONE DB. Server / dev clones keep the repo-relative cache — prod mounts
    BASE_DIR/cache as a volume, so that path must not move."""
    env = os.getenv("MAYRING_CACHE_DIR")
    if env:
        return Path(env)
    parts = set(base.parts)
    if ".claude" in parts and "plugins" in parts:
        return Path.home() / ".cache" / "mayringcoder"
    return base / "cache"


CACHE_DIR = _resolve_cache_dir(BASE_DIR)
WIKI_DIR = BASE_DIR / "wiki"
REPORTS_DIR = BASE_DIR / "reports"
PROMPTS_DIR = BASE_DIR / "prompts"
CODEBOOKS_DIR: Path = BASE_DIR / "codebooks"
CODEBOOK_PATH = CODEBOOKS_DIR / "code.yaml"

DEFAULT_PROMPT = PROMPTS_DIR / "file_inspector.md"
EXPLAINER_PROMPT = PROMPTS_DIR / "explainer.md"
OVERVIEW_PROMPT = PROMPTS_DIR / "overview.md"

# Token / budget limits
MAX_CHARS_PER_FILE = 20000
MAX_FILES_PER_RUN = 0          # 0 = kein Limit
MAX_FINDINGS_PER_FILE = 10

# GPU batching — pause every BATCH_SIZE files to cool down
BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 10

# Project context budget (Phase 1: overview cache → prompt prefix)
MAX_CONTEXT_CHARS = 6000  # ~500 tokens

# RAG context (Phase 2: ChromaDB similarity search)
RAG_TOP_K = 5                          # Number of similar context entries to inject
EMBEDDING_MODEL = "nomic-embed-text"   # Ollama embedding model (offline)

# Ollama
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "240"))
# WHY(#343 api-wedge 2026-06-06): Embeds sind schnell (nomic ~1s). 240s ließ ein
# hängendes/überlastetes Ollama den (synchron im async-Loop laufenden) Embed-Call
# bis zu 240s blockieren → Event-Loop + Threadpool wedged → /health + alle Endpoints
# tot. Embeds failen jetzt fail-fast statt minutenlang zu blockieren.
OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "30"))
OLLAMA_SSL_VERIFY: bool = os.getenv("OLLAMA_SSL_VERIFY", "true").lower() not in ("false", "0", "no")

# Overview-Job Wallclock-Budget in Sekunden (600 war zu klein für große Repos)
ANALYSIS_TIME_BUDGET: int = int(os.getenv("ANALYSIS_TIME_BUDGET", "3600"))

# gitingest content separator (48 "=" characters, per gitingest source)
INGEST_SEPARATOR = "=" * 48

# Risk categories — prioritized in Top-K file selection
RISK_CATEGORIES: frozenset[str] = frozenset({"api", "data_access", "domain"})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def repo_slug(repo_url: str) -> str:
    """Normalize a repo URL to a safe filesystem slug, e.g. ``owner-repo``."""
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").lower()
    slug = re.sub(r"\.git(?:/)?$", "", slug)
    slug = slug.replace("/", "-")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    return slug or "repo"


# ---------------------------------------------------------------------------
# Runtime-overridable limits (set via CLI flags; read by analyzer)
# ---------------------------------------------------------------------------

_active_max_chars_per_file: int = MAX_CHARS_PER_FILE
_active_batch_size: int = BATCH_SIZE
_active_batch_delay: float = BATCH_DELAY_SECONDS


def set_max_chars_per_file(limit: int) -> None:
    """Override per-file truncation limit at runtime (called once from src/cli.py)."""
    global _active_max_chars_per_file
    _active_max_chars_per_file = max(1, int(limit))


def get_max_chars_per_file() -> int:
    """Return the active per-file char limit (default: MAX_CHARS_PER_FILE)."""
    return _active_max_chars_per_file


def set_batch_size(n: int) -> None:
    """Override GPU batch size at runtime (0 = no pause)."""
    global _active_batch_size
    _active_batch_size = max(0, int(n))


def get_batch_size() -> int:
    return _active_batch_size


def set_batch_delay(seconds: float) -> None:
    """Override GPU batch delay at runtime."""
    global _active_batch_delay
    _active_batch_delay = max(0.0, float(seconds))


def get_batch_delay() -> float:
    return _active_batch_delay
