"""Memory data model — Source, Chunk, MemoryKey, RetrievalRecord.

These dataclasses are the single source of truth for the Memory layer.
They are used by memory_store.py, memory_ingest.py, and memory_retrieval.py.

Key format:  memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}
Example:     memory:repo:auth:owner-name-src-user_service.py:9f3a1b2c
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

# WHY(#252): scope_key is ALWAYS type-prefixed. Allowed types — extend
# deliberately, never accept a bare/untyped value (that's how `repo` got
# overloaded). `project:` = a Recherche project's papers; `repo:` = a code
# repo; `campaign:` reserved (gaming layer maps 1:1 to project for now, so
# unused — listed so the validator doesn't need touching later).
SCOPE_KEY_RE = re.compile(r"^(repo|project|campaign):.+$")


def is_valid_scope_key(value: str | None) -> bool:
    """True iff value is None (workspace-global) or a typed `<type>:<id>` key."""
    return value is None or bool(value and SCOPE_KEY_RE.match(value))


def canonicalize_url(url: str) -> str:
    """Normalize any URL (or slug) so case-only differences collapse.

    A URL is a URL — GitHub, GitLab, an arxiv abstract, a wiki page, an
    SSH-form git remote, or a plain `owner/name` slug. They all share one
    rule: the host and path are case-insensitive at routing time, but the
    string the user typed preserves whatever casing happened. Without
    normalisation that yields parallel source_ids and a Chroma-vs-SQLite
    split where vector search returns 0.0 even though the chunks exist.

    Lowercases host and path; preserves scheme. SSH form `git@host:owner/repo`
    is handled because Git remotes are URLs in spirit. Bare slugs and
    workspace IDs fall through to a plain lowercase.
    """
    if not url:
        return url
    if "://" in url:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower().rstrip("/")
        scheme = parsed.scheme.lower() or "https"
        return f"{scheme}://{host}{path}"
    if url.startswith("git@"):
        # git@github.com:Owner/Repo.git  → git@github.com:owner/repo
        host_path = url[4:]
        if ":" in host_path:
            host, path = host_path.split(":", 1)
            return f"git@{host.lower()}:{path.lower().removesuffix('.git')}"
    return url.lower()


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """Represents a single ingested source (file, doc, snippet, …)."""

    source_id: str          # Canonical ID, e.g. "repo:owner/name:path/to/file.py"
    source_type: str        # "repo_file" | "doc" | "note" | "conversation"
    repo: str               # "owner/name" or empty for non-repo sources
    path: str               # Relative path within repo, or arbitrary label
    branch: str = "main"
    commit: str = ""
    content_hash: str = ""  # sha256 of raw source content
    captured_at: str = field(default_factory=lambda: _now_iso())
    visibility: str = "private"   # "private" | "org" | "user" | "public"
    org_id: str | None = None
    user_id: str | None = None    # JWT.sub from app.linn.games — currently the
                                  # Laravel User.id as a string ("2"), not a
                                  # UUID. Same value across all workspaces of
                                  # the same human user; required when
                                  # visibility="user".
    # WHY(#252): typed logical sub-bucket WITHIN a workspace — e.g.
    # "project:<uuid>" for a Recherche project's papers, "repo:<url>" for a
    # code repo. NULL = workspace-global. Required for source_type in
    # (paper, agent_result) — enforced at the /ingest boundary. Never store
    # an untyped value here; see SCOPE_KEY_RE.
    scope_key: str | None = None
    # WHY(reference-doc-layer): inclusion-policy axis, ORTHOGONAL to source_type.
    # "code" = own code/notes/conversations (default, always retrieved);
    # "reference" = external docs (e.g. Unity 6.3 = 3495 chunks) that drowned
    # every 3D/graphics query. Reference is DEFAULT-EXCLUDED in both retrieval
    # stages and only surfaces via include_reference / /reference/search / a
    # repo-scoped chunk_project_links eligibility. See spec
    # 2026-06-21-reference-doc-layer.
    source_class: str = "code"

    @staticmethod
    def make_id(repo: str, path: str) -> str:
        """Build a canonical source_id from repo and path.

        repo is canonicalized so case-only typo variants collapse into one
        source_id (and one Chroma entry). Without this, ``Nileneb/X`` and
        ``nileneb/X`` ingest twice and vector retrieval misses half the corpus.
        """
        return f"repo:{canonicalize_url(repo)}:{path}"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "repo": self.repo,
            "path": self.path,
            "branch": self.branch,
            "commit": self.commit,
            "content_hash": self.content_hash,
            "captured_at": self.captured_at,
            "visibility": self.visibility,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "scope_key": self.scope_key,
            "source_class": self.source_class,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        data = {k: d.get(k, "") for k in cls.__dataclass_fields__}
        data["org_id"] = d.get("org_id")  # None when absent or NULL
        data["user_id"] = d.get("user_id")
        data["scope_key"] = d.get("scope_key")  # None when absent or NULL
        data["source_class"] = d.get("source_class") or "code"
        return cls(**data)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A segmented, versioned unit of memory derived from a Source."""

    chunk_id: str
    source_id: str
    parent_chunk_id: str | None = None
    chunk_level: str = "file"     # "file" | "class" | "function" | "section" | "block"
    ordinal: int = 0              # Position within source (0-based)
    start_offset: int = 0
    end_offset: int = 0
    text: str = ""
    text_hash: str = ""           # sha256 of text (dedup key for exact match)
    summary: str = ""
    category_labels: list[str] = field(default_factory=list)
    category_version: str = "mayring-inductive-v1"
    embedding_model: str = field(default_factory=lambda: _default_embed_model())
    embedding_id: str = ""
    quality_score: float = 0.0
    dedup_key: str = ""           # sha256 of normalized text (for near-dedup)
    category_source: str = ""     # "deductive"|"inductive"|"hybrid"|"fallback"|"manual"|""
    category_confidence: float = 0.0
    created_at: str = field(default_factory=lambda: _now_iso())
    workspace_id: str = "default"
    superseded_by: str | None = None
    is_active: bool = True

    @staticmethod
    def make_id(source_id: str, ordinal: int, chunk_level: str) -> str:
        raw = f"{source_id}:{chunk_level}:{ordinal}"
        return "chk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "parent_chunk_id": self.parent_chunk_id,
            "chunk_level": self.chunk_level,
            "ordinal": self.ordinal,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "text": self.text,
            "text_hash": self.text_hash,
            "summary": self.summary,
            "category_labels": self.category_labels,
            "category_version": self.category_version,
            "embedding_model": self.embedding_model,
            "embedding_id": self.embedding_id,
            "quality_score": self.quality_score,
            "dedup_key": self.dedup_key,
            "category_source": self.category_source,
            "category_confidence": self.category_confidence,
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
            "is_active": self.is_active,
            "workspace_id": self.workspace_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        data = dict(d)
        # category_labels is stored as comma-separated string in SQLite
        if isinstance(data.get("category_labels"), str):
            raw = data["category_labels"]
            data["category_labels"] = [c for c in raw.split(",") if c] if raw else []
        if "is_active" in data:
            data["is_active"] = bool(data["is_active"])
        return cls(**{k: data.get(k, v) for k, v in cls.__dataclass_fields__.items()
                      if k not in ("category_labels",)} | {
            "category_labels": data.get("category_labels", []),
        })


# ---------------------------------------------------------------------------
# MemoryKey
# ---------------------------------------------------------------------------

def make_memory_key(scope: str, category: str, source_fingerprint: str, chunk_hash_prefix: str) -> str:
    """Build the canonical memory key.

    Format: memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}
    Example: memory:repo:auth:owner-name-src-user_service.py:9f3a1b2c
    """
    return f"memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}"


def source_fingerprint(source_id: str) -> str:
    """Derive a short filesystem-safe fingerprint from a source_id."""
    # e.g. "repo:owner/name:src/user_service.py" → "owner-name-src-user_service.py"
    parts = source_id.split(":", 2)
    path = parts[2] if len(parts) == 3 else source_id
    slug = path.replace("/", "-").replace("\\", "-").replace(":", "-")
    # Keep it short
    if len(slug) > 48:
        slug = slug[:40] + "-" + hashlib.sha256(slug.encode()).hexdigest()[:7]
    return slug


# ---------------------------------------------------------------------------
# RetrievalRecord
# ---------------------------------------------------------------------------

@dataclass
class RetrievalRecord:
    """One ranked result from a hybrid memory search."""

    chunk_id: str
    score_vector: float = 0.0
    score_symbolic: float = 0.0
    score_recency: float = 0.0
    score_source_affinity: float = 0.0
    # Memory-Injection v2.0 — explicit logging of feedback + LLM signals
    # so the trained reranker can use them as features (avoids the
    # multi-collinearity that arises from feeding score_final back in).
    # NB: 0.5 here means "no signal yet" (chunk has zero feedback events
    # and no LLM-advisor score), NOT "user said maybe". User feedback
    # itself stays strictly binary in chunk_feedback (positive|negative)
    # — the 422-on-neutral check in /memory/feedback enforces that.
    score_feedback: float = 0.5     # [0,1]; 0.5 = no feedback events recorded
    score_llm: float = 0.5          # [0,1]; 0.5 = LLM advisor not run for this chunk
    # Issue #184: Markov-Chain-Vorhersage. predict_next_topics_for_query()
    # mappt die User-Query auf das wahrscheinlichste Topic, fragt dann das
    # transitions-DB nach Folgethemen und vergleicht diese gegen
    # category_labels des chunks. 1.0 wenn ein Folgethema in den Labels
    # auftaucht, 0.0 sonst. Im Reranker mit kleinem positiven Gewicht
    # (analog zu source_affinity) verrechnet — soll Recall-Erweiterung
    # sein, kein Hauptsignal.
    score_predicted_topic: float = 0.0
    # Reranker-v3 (#270): 1.0 wenn der Chunk über chunk_categories an eine
    # Kategorie der Query geknüpft ist (sonst 0.0). Geloggt → der tägliche
    # v2-Trainer kann das Gewicht lernen (Phase B), nicht nur der
    # deterministische Boost (Phase A).
    score_cat_match: float = 0.0
    # C3 v18 (project-scoped memory): 1.0 wenn der Chunk über chunk_project_links
    # an das Session-Projekt geknüpft ist (sonst 0.0). Triggert einen
    # DETERMINISTISCHEN Boost (_PROJECT_MATCH_BOOST) — KEIN harter Filter,
    # global/unverlinktes Wissen bleibt immer sichtbar. Bewusst NICHT als
    # gelerntes Reranker-Feature (cat_match-Lehre: trainiert negativ → schädlich).
    score_project_match: float = 0.0
    score_final: float = 0.0
    # Issue #185/#182 follow-up: rationale-edges aus wiki_v2 für diesen chunk.
    # Liste von dicts {target, context, why} — wird im compress_for_prompt
    # als '**Rationale:**' Block gerendert. Empty wenn kein wiki-Match.
    rationale_edges: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    source_id: str = ""
    text: str = ""
    summary: str = ""
    category_labels: list[str] = field(default_factory=list)
    also_in_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score_vector": round(self.score_vector, 4),
            "score_symbolic": round(self.score_symbolic, 4),
            "score_recency": round(self.score_recency, 4),
            "score_source_affinity": round(self.score_source_affinity, 4),
            "score_feedback": round(self.score_feedback, 4),
            "score_llm": round(self.score_llm, 4),
            "score_predicted_topic": round(self.score_predicted_topic, 4),
            "score_cat_match": round(self.score_cat_match, 4),
            "score_project_match": round(self.score_project_match, 4),
            "score_final": round(self.score_final, 4),
            "rationale_edges": self.rationale_edges,
            "reasons": self.reasons,
            "source_id": self.source_id,
            "text": self.text,
            "summary": self.summary,
            "category_labels": self.category_labels,
            "also_in_sources": self.also_in_sources,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_embed_model() -> str:
    """Record the embedder a chunk was made with from the single source of truth
    (config.EMBEDDING_MODEL), never a nomic literal. Lazy import avoids an import
    cycle (config ← model_router ← config)."""
    from mayring_core.config import EMBEDDING_MODEL
    return EMBEDDING_MODEL
