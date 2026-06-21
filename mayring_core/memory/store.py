"""Memory Storage Layer — SQLite + ChromaDB singletons.

Tables:
    sources, chunks, chunk_feedback, ingestion_log

ChromaDB:
    get_chroma_collection(name, path) — process-scoped singleton
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mayring_core.memory.db_adapter import DBAdapter

from mayring_core.config import CACHE_DIR
from mayring_core.memory.store_batch import batch_context, _maybe_commit, _batch_depth

# ---------------------------------------------------------------------------
# ChromaDB process-scoped singleton (replaces chroma_factory.py)
# ---------------------------------------------------------------------------

_chroma_clients: dict[str, Any] = {}
_chroma_collections: dict[str, Any] = {}


def get_chroma_collection(
    name: str = "memory_chunks",
    path: str | Path | None = None,
) -> Any:
    """Return a process-scoped ChromaDB collection singleton.

    The same (name, path) pair always returns the same object, preventing
    multiple PersistentClient instances pointing at the same directory.
    Returns None if chromadb is not installed.

    WHY(api-concurrency-capacity): when ``MAYRING_CHROMA_HOST`` is set (prod
    multi-worker), connect to a Chroma *server* via ``HttpClient`` instead of an
    embedded ``PersistentClient``. The embedded client is not multi-process-safe
    and blocks the ``uvicorn --workers`` fix (concurrent workers opening the same
    files → corruption risk). The server serialises access internally, so N API
    workers share one Chroma safely. ``path`` is ignored in server mode (the
    server owns the persist dir); it still selects the dir for the embedded
    fallback used by tests/standalone (``MAYRING_CHROMA_HOST`` unset).
    """
    # WHY(bge-m3-migration): blue-green cutover switches the live read/write
    # collection via env without editing the ~6 hardcoded call-sites. Only the
    # default "memory_chunks" name is remapped; codebook_categories/overview
    # pass through unchanged (re-embedded in-place at flip time).
    if name == "memory_chunks":
        name = os.getenv("MEMORY_CHUNKS_COLLECTION", "memory_chunks")

    try:
        import chromadb
    except ImportError:
        return None

    host = os.getenv("MAYRING_CHROMA_HOST")
    if host:
        port = int(os.getenv("MAYRING_CHROMA_PORT", "8000"))
        ckey = f"http://{host}:{port}"
        if ckey not in _chroma_clients:
            _chroma_clients[ckey] = chromadb.HttpClient(host=host, port=port)
        key = f"{ckey}::{name}"
        if key not in _chroma_collections:
            _chroma_collections[key] = _chroma_clients[ckey].get_or_create_collection(name)
        return _chroma_collections[key]

    # back-compat (tests, standalone) — embedded PersistentClient
    chroma_path = str(path or CACHE_DIR / "memory_chroma")
    key = f"{chroma_path}::{name}"
    if key not in _chroma_collections:
        if chroma_path not in _chroma_clients:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            _chroma_clients[chroma_path] = chromadb.PersistentClient(path=chroma_path)
        _chroma_collections[key] = _chroma_clients[chroma_path].get_or_create_collection(name)
    return _chroma_collections[key]
def reset_chroma_collection(name: str, path: "str | Path | None" = None):
    """Drop + recreate a Chroma collection and clear the singleton caches.

    WHY(bge-m3-migration dim-mismatch 2026-06-08): a collection is dim-locked at the
    embedding size it was created with. After the store moved nomic(768)→bge-m3(1024),
    every re-embed into an old 768d collection raised "Collection expecting embedding
    with dimension of 768, got 1024" — it bit memory_sync (local), reembed-categories
    (codebook_categories) and the migration. This is the one reusable self-heal: callers
    catch the dim error, call this, and retry the upsert against the fresh collection.
    Honours the same MAYRING_CHROMA_HOST / embedded-path resolution as
    get_chroma_collection and evicts the cached client/collection entries."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return None
    if name == "memory_chunks":
        name = os.getenv("MEMORY_CHUNKS_COLLECTION", "memory_chunks")
    host = os.getenv("MAYRING_CHROMA_HOST")
    if host:
        port = int(os.getenv("MAYRING_CHROMA_PORT", "8000"))
        ckey = f"http://{host}:{port}"
        client = _chroma_clients.get(ckey)
        if client is None:
            client = _chroma_clients.setdefault(
                ckey, chromadb.HttpClient(host=host, port=port))
        try:
            client.delete_collection(name)
        except Exception:
            pass  # not-found is fine — we're about to (re)create it
        _chroma_collections.pop(f"{ckey}::{name}", None)
    else:
        chroma_path = str(path or CACHE_DIR / "memory_chroma")
        client = _chroma_clients.get(chroma_path)
        if client is not None:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        _chroma_collections.pop(f"{chroma_path}::{name}", None)
    return get_chroma_collection(name, path=path)


from mayring_core.memory.schema import Chunk, Source

MEMORY_DB_PATH: Path = CACHE_DIR / "memory.db"

# ---------------------------------------------------------------------------
# In-memory KV cache (session-scoped, evicted on process exit)
# ---------------------------------------------------------------------------
_KV: dict[str, dict] = {}


def kv_get(chunk_id: str) -> dict | None:
    return _KV.get(chunk_id)


def kv_put(chunk_id: str, chunk_dict: dict) -> None:
    _KV[chunk_id] = chunk_dict


def kv_invalidate_by_ids(chunk_ids: list[str]) -> None:
    for cid in chunk_ids:
        _KV.pop(cid, None)


# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------

# Bump when _init_schema gains new DDL/migrations so existing DBs re-run it.
#   v2 (#252): sources.scope_key column + index.
#   v3 (task-tracker): tasks table + workspace/status/due indexes.
#   v4 (todo-dedup): tasks.derive_embedding column for prompt-vs-prompt dedup.
#   v7 (#workspace-uuid-sot): projects table + chunks/sources.project_id —
#       Projekt-Trennung als Dimension UNTER dem einen Workspace (User-Diagramm),
#       statt fake-separate workspace_ids.
#   v8 (#workspace-uuid-sot): projects.{owner_id, source_type, source_ref} —
#       Projekt-Diagramm: ein Projekt hat einen Owner + eine Source (github-Repo
#       ODER papers/research). Goals↔Project (M:M) = Backend-Designfrage offen.
#   v9 (#workspace-uuid-sot v2.0 Phase 1): codebook-in-DB (codebooks,
#       codebook_categories, codebook_proposals, chunk_categories) — Codebook aus
#       YAML → SQLite, Category-Embeddings → Chroma-Collection "codebook_categories".
#   v12 (smoke-fix): index on context_feedback_log.captured_at so the 24h
#       COUNT in /stats/feedback-log is a range-scan, not a full table scan —
#       lets that live-movement endpoint run UNcached (per-poll) without load.
#   v13 (smoke-fix): index on ingestion_log.created_at so /stats/recent-ops
#       (ORDER BY created_at DESC LIMIT) stays cheap UNcached — its 15s cache
#       served stale on write-then-read (watcher_hook_fires saw the ingest only
#       after the TTL; same class as the feedback-log fix).
#   v15 (tenancy phase A): visibility axis 4 values -> 3 values. Re-sorts
#       existing rows (user->private, org/team-workspace private->org with
#       org_id stamped, personal-workspace private gets user_id stamped) and
#       tightens the sources.visibility CHECK to ('private','org','public').
#       Runs at boot via migrate_visibility_axis() before the new scope_filter
#       ships -> no search blackout.
#   v16 (tenancy phase B): workspace_role_permissions table — per-workspace
#       role permission overrides (user/editor/admin × permission → allowed).
#       Absence of a row means the DEFAULT_MATRIX from authz.py applies.
#   v17 (C1 project-groups): project_groups table (named, colored groups per
#       workspace) + projects.group_id FK column.
#   v18 (C3 project-scoped memory): chunk_project_links table — 1-to-many
#       chunk↔project links (origin_ref is nested-repo-aware). Retires the
#       unpopulated hard chunks.project_id filter in retrieval._scope_filter in
#       favour of a deterministic project_match boost (the chunks.project_id
#       column stays DORMANT, not dropped). No backfill.
#   v25 (igio-removal 2026-06-12): DROP legacy IGIO columns.
#   v26 (#365): embed-pool device cols.
#   v27 (#365 house-audit): embed_jobs.is_audit.
#   v28 (reference-doc-layer): sources.source_class column ('code'|'reference').
#       Reference corpora (Unity docs) are default-excluded from retrieval. The
#       ALTER backfills every existing row to 'code'; a follow-up UPDATE flips
#       known reference prefixes (unity-docs:) to 'reference'. See spec
#       2026-06-21-reference-doc-layer + 2026-06-21-retrieval-repo-scoping-hardfilter.
CURRENT_SCHEMA_VERSION = 28  # v28: sources.source_class (reference-doc-layer)

# Source-id prefixes whose chunks are external reference docs, not own code.
# The v28 migration flips these to source_class='reference'. Extend when a new
# reference corpus is ingested without the --reference flag.
_REFERENCE_SOURCE_PREFIXES = ("unity-docs:",)

# C1 (project-groups): kuratierte, dark-mode-taugliche Palette. EINE Definition —
# Auto-Vergabe + Validierung (API) lesen hier; spätere Statusline (C2) liest die
# pro Gruppe gespeicherte Hex-Farbe, nicht diese Liste.
PROJECT_GROUP_PALETTE = [
    "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ec4899",
    "#14b8a6", "#ef4444", "#6366f1", "#84cc16",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_memory_db(db_path: Path | None = None) -> DBAdapter:
    """Create or open memory.db and ensure all tables + indexes exist.

    check_same_thread=False ist nötig weil die cachende Verbindung in web_ui.py
    und mcp.py über Gradio/FastMCP-Worker-Threads hinweg benutzt wird. WAL-Mode
    schützt vor Korruption bei konkurrierenden Readern; Writes sind im MayringCoder-
    Stack immer sequenziell (ein Ingest nach dem anderen), kein Lock nötig.

    ``busy_timeout=10000`` verhindert SQLITE_BUSY, wenn mehrere Prozesse
    (mayring-api, mayring-pi, mayring-mcp) gleichzeitig auf die DB zugreifen
    — SQLite wartet dann bis zu 10 s auf eine Writer-Lock statt sofort zu
    failen (Default 5 s war knapp bei parallelen Ingest+Retrieval-Calls).
    """
    path = db_path or MEMORY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter = DBAdapter.create(path, check_same_thread=False)
    _init_schema(adapter)

    # ATTACH wiki_v2.db als 'wikidb' für Cross-DB-Joins (rationale-edges,
    # cluster-lookups). Spec: docs/superpowers/specs/2026-05-09-rationale-edges-design.md.
    # Cold-start ohne wiki_v2.db: Flag = False, alle JOINs silent skip.
    wiki_path = path.parent / "wiki_v2.db"
    adapter._wiki_attached = False  # type: ignore[attr-defined]
    if wiki_path.exists():
        try:
            adapter.execute("ATTACH DATABASE ? AS wikidb", (str(wiki_path),))
            adapter._wiki_attached = True  # type: ignore[attr-defined]
        except sqlite3.OperationalError as e:
            import logging
            logging.getLogger(__name__).warning(
                "wiki_v2.db ATTACH failed (silent skip): %s", e,
            )

    return adapter


def _migrate_schema(conn: DBAdapter) -> None:
    """Add missing columns to existing DBs (idempotent migrations)."""
    migrations = {
        "sources": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
            ("visibility", "TEXT NOT NULL DEFAULT 'private'"),
            ("org_id", "TEXT DEFAULT NULL"),
            # user_id: User-scoped sharing across workspaces. Same user, multiple
            # workspace_ids (claude.ai web vs claude-cli) → set visibility='user'
            # so all sessions of the same user can see each other's memory
            # without falling back to 'public'.
            ("user_id", "TEXT DEFAULT NULL"),
            # scope_key (#252): typed logical sub-bucket within a workspace —
            # "project:<uuid>" / "repo:<url>" / NULL=workspace-global. Existing
            # rows get NULL (they were workspace-global before this concept).
            ("scope_key", "TEXT DEFAULT NULL"),
            # project_id (#workspace-uuid-sot, v7): FK→projects.id. The Project-ID
            # dimension UNTER dem einen Workspace (User-Diagramm). NULL = kein
            # Projekt (workspace-global). Ersetzt fake-separate workspace_ids.
            ("project_id", "TEXT DEFAULT NULL"),
            # source_class (v25, reference-doc-layer): inclusion-policy axis.
            # ALTER ... DEFAULT 'code' backfills every existing row to 'code'
            # automatically → SQL _scope_filter is consistent the moment this
            # migration runs (no NULL/absent window). Reference rows get flipped
            # by _migrate_reference_source_class below.
            ("source_class", "TEXT NOT NULL DEFAULT 'code'"),
        ],
        "chunks": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
            ("project_id", "TEXT DEFAULT NULL"),
            ("category_source", "TEXT NOT NULL DEFAULT ''"),
            ("category_confidence", "REAL NOT NULL DEFAULT 0.0"),
            ("user_id", "TEXT DEFAULT NULL"),
        ],
        # pi_jobs Phase 2: cloud-routable jobs need a worker_id, capability,
        # and a scope marker so local-only and cloud-routable rows can coexist.
        # pi_jobs T1 (#183): in-process queue extension (kind, job_class,
        # system_prompt, model_used, latency_ms).
        "pi_jobs": [
            ("scope", "TEXT NOT NULL DEFAULT 'local'"),
            ("capability_required", "TEXT NOT NULL DEFAULT ''"),
            ("claimed_by", "TEXT NOT NULL DEFAULT ''"),
            ("claimed_at", "TEXT NOT NULL DEFAULT ''"),
            ("kind", "TEXT NOT NULL DEFAULT 'pi-task'"),
            ("job_class", "TEXT NOT NULL DEFAULT 'standard'"),
            ("system_prompt", "TEXT NOT NULL DEFAULT ''"),
            ("model_used", "TEXT NOT NULL DEFAULT ''"),
            ("latency_ms", "INTEGER NOT NULL DEFAULT 0"),
        ],
        # Memory-Injection v2.0 — retrieval metrics + reranker training data.
        # The hook-injection path already wrote (trigger_ids, context_text,
        # captured_at) here. Adding query + stage_scores + workspace_id makes
        # the row a complete join key for chunk_feedback so we can compute
        # precision@K / NDCG@K and export training triples.
        "context_feedback_log": [
            ("query", "TEXT NOT NULL DEFAULT ''"),
            ("stage_scores", "TEXT NOT NULL DEFAULT '{}'"),
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
            # v2.0 reranker A/B logging — which formula scored this row.
            ("reranker_version", "TEXT NOT NULL DEFAULT 'v1'"),
        ],
        # Email-slug Workspace-IDs ab 2026-05-09 (statt user-N).
        # email-Spalte für display-name + multi-tenant-vorbereitete
        # Lookups.
        "workspaces": [
            ("email", "TEXT DEFAULT NULL"),
        ],
        # WHY(v2-stufe1, multi-tenant): Schema-Test
        # `assert_no_unworkspaced_table` failt für jede user-data-Tabelle
        # ohne workspace_id. Vor V2 hatten topic_transitions/chunk_feedback/
        # wiki_paper_cache/ingestion_log keinen Tenant-Filter — d.h. alice
        # konnte bene's Markov-Edges, Feedback-Signale, Paper-Cache und
        # Ingest-Events sehen. Idempotente Migrations: bestehende DBs
        # bekommen die Spalte mit DEFAULT 'default' (legacy-Zeilen sammeln
        # sich da, gehören dem 'default'-Bucket).
        "topic_transitions": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
        ],
        "chunk_feedback": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
        ],
        "wiki_paper_cache": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
        ],
        "ingestion_log": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
        ],
        "tasks": [
            ("derive_embedding", "TEXT"),
            ("external_id", "TEXT"),
        ],
        # #workspace-uuid-sot v8 (User-Diagramm): Projekt-Metadaten.
        "projects": [
            ("owner_id", "TEXT"),
            ("source_type", "TEXT"),
            ("source_ref", "TEXT"),
            # C1 (project-groups v17): FK→project_groups.id, NULL = keine Gruppe.
            ("group_id", "TEXT DEFAULT NULL"),
        ],
        # v2 Phase 3.2 (project-scoped codebook): NULL = geteilte Profil-Basis
        # (imported), gesetzt = projekt-spezifisch induziert. Der aktive
        # Kategorien-Satz einer Session = Basis ∪ active_project.categories.
        # NOTE: "categories" is the post-v19 name; "codebook_categories" is kept
        # here so the generic column-migration can still back-fill project_id on
        # DBs that are at v18 and haven't been collapsed yet. The migration guard
        # in _migrate_codebook_collapse short-circuits when "categories" already
        # exists, so this entry is a no-op on fully-migrated DBs.
        "codebook_categories": [
            ("project_id", "TEXT DEFAULT NULL"),
        ],
        # v21 goal-first-class: Goal-Anker auf der Kategorie. Additiv+nullable
        # (FK→goals.id); existierende Kategorien bekommen NULL (kein Goal bekannt)
        # bis der breite Re-Ingest sie goal-anchored neu bildet. Die goals-Tabelle
        # wird im DDL-Block VOR _migrate_schema angelegt → FK-Ziel existiert.
        "categories": [
            ("goal_id", "INTEGER REFERENCES goals(id)"),
        ],
    }
    existing_tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for table, columns in migrations.items():
        if table not in existing_tables:
            continue
        existing = conn.get_columns(table)
        for col_name, col_def in columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")

    _migrate_rename_research_questions(conn)
    _migrate_visibility_check(conn)
    _drop_legacy_igio_columns(conn)
    _migrate_reference_source_class(conn)


def _drop_legacy_igio_columns(conn: DBAdapter) -> None:
    # WHY(igio-removal 2026-06-12): IGIO-Achse abgeschafft — Spalten auf
    # Bestands-DBs wegräumen statt tote Daten zu schleppen. DROP COLUMN
    # braucht SQLite >=3.35; ältere DBs behalten die Spalte (harmlos, kein Reader).
    for table, col in (("chunks", "igio_axis"), ("chunks", "igio_confidence"),
                       ("chunks", "igio_classified_at"), ("categories", "igio_axis")):
        try:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        except Exception:
            pass  # Spalte existiert nicht (neue DB) oder SQLite <3.35


def _migrate_reference_source_class(conn: DBAdapter) -> None:
    """Flip known external reference corpora to source_class='reference'.

    The generic ALTER above defaults every row to 'code'. This idempotent
    UPDATE marks the reference prefixes (Unity docs) so they get
    default-excluded from retrieval. Runs only when the column exists.
    """
    if "source_class" not in conn.get_columns("sources"):
        return
    for prefix in _REFERENCE_SOURCE_PREFIXES:
        conn.execute(
            "UPDATE sources SET source_class = 'reference' "
            "WHERE source_id LIKE ? AND source_class != 'reference'",
            (prefix + "%",),
        )
    _maybe_commit(conn)


def _migrate_rename_research_questions(conn: DBAdapter) -> None:
    """Rename the legacy task_categories layer to research_questions.

    WHY(task-tracker): 'task' was overloaded — a new first-class `tasks`
    tracker now owns that word, so the derived-research-question storage is
    renamed to disambiguate. Idempotent: only fires when the old tables exist.
    """
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "task_categories" in tables and "research_questions" not in tables:
        conn.execute("ALTER TABLE task_categories RENAME TO research_questions")
        conn.execute("ALTER TABLE research_questions RENAME COLUMN task_id TO research_question_id")
    if "task_chunk_links" in tables and "research_question_chunk_links" not in tables:
        conn.execute("ALTER TABLE task_chunk_links RENAME TO research_question_chunk_links")
        conn.execute("ALTER TABLE research_question_chunk_links RENAME COLUMN task_id TO research_question_id")
    conn.commit()


def _migrate_visibility_check(conn: DBAdapter) -> None:
    """Extend sources.visibility CHECK constraint to allow 'user'.

    SQLite can't ALTER a CHECK constraint in place — the only way is to rebuild
    the table. Idempotent: detects via PRAGMA whether the new constraint is
    already in effect and no-ops when it is.
    """
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sources'"
        ).fetchall()
    except Exception:
        return
    if not rows:
        return
    sql = rows[0][0] or ""
    if "'user'" in sql:
        return  # already extended
    if "CHECK(visibility IN" not in sql and "CHECK (visibility IN" not in sql:
        return  # no CHECK at all (very old DB) — leave alone
    try:
        conn.execute("ALTER TABLE sources RENAME TO sources_legacy_visibility")
        conn.executescript("""
            CREATE TABLE sources (
                source_id       TEXT PRIMARY KEY,
                source_type     TEXT NOT NULL DEFAULT 'repo_file',
                repo            TEXT NOT NULL DEFAULT '',
                path            TEXT NOT NULL DEFAULT '',
                branch          TEXT NOT NULL DEFAULT 'main',
                "commit"        TEXT NOT NULL DEFAULT '',
                content_hash    TEXT NOT NULL DEFAULT '',
                captured_at     TEXT NOT NULL,
                workspace_id    TEXT NOT NULL DEFAULT 'default',
                visibility      TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'org', 'public', 'user')),
                org_id          TEXT DEFAULT NULL,
                user_id         TEXT DEFAULT NULL,
                scope_key       TEXT DEFAULT NULL,
                source_class    TEXT NOT NULL DEFAULT 'code'
            );
        """)
        legacy_cols = conn.get_columns("sources_legacy_visibility")
        select_cols = ", ".join(
            f'"{c}"' if c == "commit" else c
            for c in legacy_cols
        )
        # Insert preserving every column the legacy table had — new columns
        # (user_id) get DEFAULT NULL automatically.
        conn.execute(
            f"INSERT INTO sources ({select_cols}) "
            f"SELECT {select_cols} FROM sources_legacy_visibility"
        )
        conn.execute("DROP TABLE sources_legacy_visibility")
        _maybe_commit(conn)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "visibility-check migration failed: %s — DB still on old constraint", exc
        )


# Tenancy Phase A (v15): the sources table after migrations carries
# workspace_id + project_id (added by the generic column-migration on existing
# DBs, and never present in the legacy CREATE TABLE block). This DDL is the
# 3-value-CHECK rebuild target — an exact copy of the live column set, only the
# visibility CHECK is tightened to ('private', 'org', 'public').
_SOURCES_DDL_V15 = """
    CREATE TABLE sources (
        source_id       TEXT PRIMARY KEY,
        source_type     TEXT NOT NULL DEFAULT 'repo_file',
        repo            TEXT NOT NULL DEFAULT '',
        path            TEXT NOT NULL DEFAULT '',
        branch          TEXT NOT NULL DEFAULT 'main',
        "commit"        TEXT NOT NULL DEFAULT '',
        content_hash    TEXT NOT NULL DEFAULT '',
        captured_at     TEXT NOT NULL,
        visibility      TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'org', 'public')),
        org_id          TEXT DEFAULT NULL,
        user_id         TEXT DEFAULT NULL,
        scope_key       TEXT DEFAULT NULL,
        workspace_id    TEXT NOT NULL DEFAULT 'default',
        project_id      TEXT DEFAULT NULL,
        source_class    TEXT NOT NULL DEFAULT 'code'
    );
"""


def _personal_owner_map(conn: DBAdapter) -> dict:
    """Owner lookup for personal workspaces, from env MAYRING_PERSONAL_OWNERS.

    JSON object ws_id -> user_id. Returns {} when unset/empty/malformed — the
    migration then simply skips the personal-owner stamping step (rows keep
    their existing user_id, which is the safe no-op).
    """
    raw = os.environ.get("MAYRING_PERSONAL_OWNERS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import logging
        logging.getLogger(__name__).warning(
            "MAYRING_PERSONAL_OWNERS is not valid JSON — ignoring"
        )
        return {}
    return parsed if isinstance(parsed, dict) else {}


# WHY(tenancy-v15, BLOCKER 1): the sources indexes are CREATEd in _init_schema
# BEFORE this migration runs, and the table rebuild (RENAME->CREATE->...->DROP)
# drops them with the legacy table. The short-circuit `if current_version >=
# CURRENT_SCHEMA_VERSION: return` then prevents _init_schema from ever
# re-creating them -> full-table-scan on the retrieval hot path. Re-create them
# in the same transaction as the rebuild. DDLs copied verbatim from
# _init_schema (grep idx_sources).
_SOURCES_INDEX_DDL_V15 = (
    "CREATE INDEX IF NOT EXISTS idx_sources_workspace_id ON sources(workspace_id)",
    "CREATE INDEX IF NOT EXISTS idx_sources_scope_key ON sources(scope_key)",
)


def _copy_legacy_into_sources(conn: DBAdapter) -> None:
    """INSERT...SELECT every legacy column into the freshly-built `sources`."""
    legacy_cols = conn.get_columns("sources_legacy_v15")
    select_cols = ", ".join(
        f'"{c}"' if c == "commit" else c
        for c in legacy_cols
    )
    conn.execute(
        f"INSERT INTO sources ({select_cols}) "
        f"SELECT {select_cols} FROM sources_legacy_v15"
    )


def _rebuild_visibility_check_3values(conn: DBAdapter) -> None:
    """Rebuild sources with the tightened 3-value visibility CHECK (v15).

    SQLite can't ALTER a CHECK in place — rebuild via RENAME -> CREATE ->
    INSERT SELECT -> DROP, mirroring _migrate_visibility_check. Idempotent:
    no-ops once 'user' is gone from the live sources DDL in sqlite_master.

    WHY(tenancy-v15, BLOCKER 2): the rebuild must be atomic. `executescript`
    forces an implicit COMMIT, so RENAME+CREATE would land BEFORE the
    INSERT...SELECT — a crash in between (OOM/cutover) would strand all rows in
    sources_legacy_v15. We therefore build the new table with a single
    `conn.execute` (no executescript), keep RENAME+CREATE+INSERT+DROP+indexes
    in ONE transaction and commit only at the very end. On error we roll back
    so a half-rebuild never persists. A recovery path handles the case where a
    previous crash already swapped to the 3-value table but left legacy data.
    """
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sources'"
    ).fetchall()
    if not rows:
        return
    sql = rows[0][0] or ""

    legacy_exists = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sources_legacy_v15'"
    ).fetchone())

    if "'user'" not in sql:
        # Already on the 3-value CHECK. BLOCKER 2 (c): if a prior crash left a
        # populated legacy table (RENAME+CREATE committed, INSERT did not), the
        # new `sources` is empty and the rows live only in the legacy table.
        # Recover by completing the copy instead of blindly skipping.
        if not legacy_exists:
            return
        new_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM sources_legacy_v15"
        ).fetchone()[0]
        if new_count == 0 and legacy_count > 0:
            # WHY(tenancy-v15): toggle FK off so the DROP doesn't dangle the
            # chunks FK (same reasoning as the main rebuild below).
            conn.commit()
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                _copy_legacy_into_sources(conn)
                conn.execute("DROP TABLE sources_legacy_v15")
                for ddl in _SOURCES_INDEX_DDL_V15:
                    conn.execute(ddl)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
        else:
            # New table already populated — the legacy table is stale residue.
            conn.execute("DROP TABLE IF EXISTS sources_legacy_v15")
            conn.commit()
        return

    # WHY(tenancy-v15): with foreign_keys=ON, ALTER TABLE ... RENAME rewrites
    # chunks.source_id's REFERENCES to the renamed table, so DROPping it would
    # leave the chunks FK dangling -> next insert_chunk fails ("no such table
    # sources_legacy_v15"). legacy_alter_table=ON suppresses that rewrite so
    # the FK re-binds to the freshly CREATEd `sources`. foreign_keys can only
    # be toggled outside a transaction, so commit first and restore after.
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        # BLOCKER 2 (b): a stale legacy table from an aborted earlier run would
        # make the RENAME fail ("table sources_legacy_v15 already exists").
        # Drop it defensively first. (Reached only when 'user' is still in the
        # live CHECK, i.e. the rebuild genuinely hasn't completed.)
        conn.execute("DROP TABLE IF EXISTS sources_legacy_v15")
        conn.execute("ALTER TABLE sources RENAME TO sources_legacy_v15")
        # BLOCKER 2 (a): single execute, NOT executescript — no implicit commit
        # splits the rebuild. RENAME+CREATE+INSERT+DROP+indexes stay in one
        # transaction, committed only at the end.
        conn.execute(_SOURCES_DDL_V15)
        _copy_legacy_into_sources(conn)
        conn.execute("DROP TABLE sources_legacy_v15")
        for ddl in _SOURCES_INDEX_DDL_V15:
            conn.execute(ddl)
        conn.commit()
    except Exception:
        # BLOCKER (rollback): drop the half-built state so the next boot retries
        # from a consistent point instead of stranding data.
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table = OFF")
        conn.execute("PRAGMA foreign_keys = ON")


def migrate_visibility_axis(conn: DBAdapter, personal_owner: dict | None = None) -> None:
    """Re-sort sources rows onto the 3-value visibility axis (v15).

    1. visibility='user' -> 'private' (user_id is already cross-device-me).
    2. visibility='private' in an org/team workspace -> 'org', org_id stamped
       from workspace_id.
    3. visibility='private' in a personal workspace -> stamp user_id = owner
       (from `personal_owner` ws_id->user_id) where it is NULL/empty.
    Then tighten the CHECK to ('private', 'org', 'public') via table rebuild.

    Run BEFORE the new scope_filter ships so retrieval never goes dark.
    """
    personal_owner = personal_owner or {}

    # 1. legacy 'user' is now plain user_id-scoped 'private'.
    conn.execute(
        "UPDATE sources SET visibility = 'private' WHERE visibility = 'user'"
    )

    # 2. workspace-scoped 'private' rows in org/team workspaces become 'org'.
    # workspaces.kind CHECK ships ('user','team','project','system'); 'team'
    # and 'project' are the shared/org buckets. ('organization' is included
    # defensively for any externally-seeded row that used the long form.)
    #
    # WHY(tenancy-v15, ordering): step 1 above just converted 'user' rows to
    # 'private' WITH a user_id set. Those are user-owned, not workspace-shared —
    # promoting them to 'org' would leak a personal chunk into the whole team.
    # Restrict the org promotion to genuinely workspace-scoped rows (no owner),
    # so ex-'user' rows stay 'private'.
    conn.execute(
        "UPDATE sources SET visibility = 'org', org_id = workspace_id "
        "WHERE visibility = 'private' AND (user_id IS NULL OR user_id = '') "
        "AND workspace_id IN ("
        "  SELECT id FROM workspaces WHERE kind IN ('team', 'project', 'organization')"
        ")"
    )

    # 3. personal-workspace 'private' rows get their owner stamped (where blank).
    for ws_id, user_id in personal_owner.items():
        if not user_id:
            continue
        conn.execute(
            "UPDATE sources SET user_id = ? "
            "WHERE workspace_id = ? AND visibility = 'private' "
            "AND (user_id IS NULL OR user_id = '')",
            (user_id, ws_id),
        )

    _rebuild_visibility_check_3values(conn)
    conn.commit()


def _migrate_codebook_collapse(conn: DBAdapter) -> None:
    """v19: Codebook-Konzept entfernen. codebook_categories → categories (ohne codebook_id,
    UNIQUE(name)), dependent FKs (chunk_categories, codebook_proposals) auf categories
    umgebogen, codebooks-Tabelle gedroppt. Idempotent + geguardet (v14-Lehre: darf init
    NIE bricken, sonst jeder get_conn 500tet).

    Zwei-Phasen-Rebuild (analog _migrate_visibility_check / _migrate_devices_composite_pk):
    Phase A: codebook_categories in-place rebuilden (codebook_id weg, UNIQUE(name)) unter
    GLEICHEM Namen → dependent FKs (chunk_categories.category_id, codebook_proposals.
    category_id/parent_hint_id, self parent_id) binden danach wieder an codebook_categories.
    Phase B: RENAME zu "categories" mit legacy_alter_table=OFF → SQLite rewritet alle
    FK-Refs auf "categories".
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "categories" in tables or "codebook_categories" not in tables:
        return  # frische DB oder schon migriert
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        # Phase A: codebook_categories in-place rebuilden (codebook_id weg,
        # UNIQUE(name)). Gleicher Name → dependent FKs binden danach wieder.
        conn.execute("DROP TABLE IF EXISTS codebook_categories_old_v19")
        conn.execute("ALTER TABLE codebook_categories RENAME TO codebook_categories_old_v19")
        conn.execute(
            """
            CREATE TABLE codebook_categories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                parent_id       INTEGER REFERENCES codebook_categories(id),
                description     TEXT NOT NULL DEFAULT '',
                examples        TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active','proposed','deprecated')),
                source          TEXT NOT NULL DEFAULT 'imported'
                                CHECK (source IN ('manual','induced','imported')),
                evidence_count  INTEGER NOT NULL DEFAULT 0,
                embedding_id    TEXT NOT NULL DEFAULT '',
                risk_level      TEXT NOT NULL DEFAULT '',
                languages       TEXT NOT NULL DEFAULT '[]',
                patterns        TEXT NOT NULL DEFAULT '[]',
                promoted_at     TEXT,
                project_id      TEXT DEFAULT NULL,
                UNIQUE (name)
            )
            """
        )
        conn.execute(
            "INSERT INTO codebook_categories "
            "(id,name,parent_id,description,examples,status,source,"
            "evidence_count,embedding_id,risk_level,languages,patterns,promoted_at,project_id) "
            "SELECT id,name,parent_id,description,examples,status,source,"
            "evidence_count,embedding_id,risk_level,languages,patterns,promoted_at,project_id "
            "FROM codebook_categories_old_v19"
        )
        conn.execute("DROP TABLE codebook_categories_old_v19")
        conn.execute("PRAGMA legacy_alter_table = OFF")
        # Phase B: RENAME zu "categories" — legacy OFF rewritet dependent FK-Refs
        # (chunk_categories.category_id, codebook_proposals.category_id/parent_hint_id,
        # self parent_id) auf den neuen Namen "categories".
        conn.execute("ALTER TABLE codebook_categories RENAME TO categories")
        conn.execute("DROP TABLE IF EXISTS codebooks")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table = OFF")
        conn.execute("PRAGMA foreign_keys = ON")


def _migrate_categories_goal_unique(conn: DBAdapter) -> None:
    """v22: categories UNIQUE(name) → UNIQUE(goal_id, name) (goal-gescopt).

    WHY: goal-anchored Mayring — derselbe Kategoriename darf unter verschiedenen
    Goals existieren. UNIQUE(name) erzwang Cross-Goal-Merge. SQLite kann UNIQUE
    nicht in-place ändern → in-place-Rebuild unter GLEICHEM Namen "categories", damit
    die dependent FKs (chunk_categories.category_id, self parent_id) wieder binden
    (analog _migrate_codebook_collapse). MUSS NACH _migrate_schema laufen (das
    goal_id per ALTER anlegt). Idempotent + geguardet (v14-Lehre: darf init NIE
    bricken). Die Basis-Eindeutigkeit (goal_id IS NULL) sichert danach der partielle
    Index ux_categories_base_name (init legt ihn nach dieser Migration an)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "categories" not in tables:
        return  # frische DB: DDL legt direkt UNIQUE(goal_id,name) an
    ddl_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='categories'"
    ).fetchone()
    ddl = (ddl_row[0] if ddl_row else "") or ""
    if "goal_id" not in ddl:
        return  # goal_id noch nicht da (sollte nach _migrate_schema nie passieren)
    if "UNIQUE(goal_id,name)" in ddl.replace(" ", ""):
        return  # schon goal-gescopt
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        # legacy ON: dependents behalten ihre Referenz auf "categories"; die
        # frisch erzeugte gleichnamige Tabelle erbt sie. Kein Rename-back nötig
        # (Name bleibt "categories").
        conn.execute("DROP TABLE IF EXISTS categories_old_v22")
        conn.execute("ALTER TABLE categories RENAME TO categories_old_v22")
        conn.execute(
            """
            CREATE TABLE categories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                parent_id       INTEGER REFERENCES categories(id),
                description     TEXT NOT NULL DEFAULT '',
                examples        TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active','proposed','deprecated')),
                source          TEXT NOT NULL DEFAULT 'imported'
                                CHECK (source IN ('manual','induced','imported')),
                evidence_count  INTEGER NOT NULL DEFAULT 0,
                embedding_id    TEXT NOT NULL DEFAULT '',
                risk_level      TEXT NOT NULL DEFAULT '',
                languages       TEXT NOT NULL DEFAULT '[]',
                patterns        TEXT NOT NULL DEFAULT '[]',
                promoted_at     TEXT,
                project_id      TEXT DEFAULT NULL,
                goal_id         INTEGER REFERENCES goals(id),
                UNIQUE (goal_id, name)
            )
            """
        )
        conn.execute(
            "INSERT INTO categories "
            "(id,name,parent_id,description,examples,status,source,"
            "evidence_count,embedding_id,risk_level,languages,patterns,promoted_at,"
            "project_id,goal_id) "
            "SELECT id,name,parent_id,description,examples,status,source,"
            "evidence_count,embedding_id,risk_level,languages,patterns,promoted_at,"
            "project_id,goal_id FROM categories_old_v22"
        )
        conn.execute("DROP TABLE categories_old_v22")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table = OFF")
        conn.execute("PRAGMA foreign_keys = ON")


def _migrate_devices_composite_pk(conn: DBAdapter) -> None:
    """Rebuild `devices` with PK (device_id, workspace_id) — migration v5→v6.

    WHY(#274 review): v5 shipped `devices` with a sole `device_id` PK, but every
    lookup is workspace-scoped, so a re-register in another workspace would
    silently move the row (ON CONFLICT(device_id)). Composite PK lets the same
    device_id coexist per workspace. SQLite can't ALTER a PK in place → rebuild.
    Idempotent: short-circuits once the PK is already composite. Rows preserved
    (old single-PK rows have unique device_id → no composite collision)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "devices" not in tables:
        return
    # PRAGMA table_info row: (cid, name, type, notnull, dflt_value, pk).
    pk_cols = {r[1] for r in conn.execute("PRAGMA table_info(devices)").fetchall()
               if r[5] > 0}
    if pk_cols == {"device_id", "workspace_id"}:
        return  # already migrated
    conn.executescript(
        """
        CREATE TABLE devices_v6 (
            device_id    TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            name         TEXT NOT NULL DEFAULT '',
            os           TEXT NOT NULL DEFAULT '',
            capabilities TEXT NOT NULL DEFAULT '',
            last_seen    TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'active',
            created_at   TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (device_id, workspace_id)
        );
        INSERT INTO devices_v6
            SELECT device_id, workspace_id, name, os, capabilities,
                   last_seen, status, created_at FROM devices;
        DROP TABLE devices;
        ALTER TABLE devices_v6 RENAME TO devices;
        CREATE INDEX IF NOT EXISTS idx_devices_workspace ON devices(workspace_id);
        """
    )
    conn.commit()


def _migrate_devices_embed_pool_cols(conn: DBAdapter) -> None:
    """Add embed-pool columns to an existing `devices` table — migration v25→v26.
    Idempotent: ADD COLUMN can't be IF NOT EXISTS in SQLite, so gate on PRAGMA table_info."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "devices" not in tables:
        return
    have = {r[1] for r in conn.execute("PRAGMA table_info(devices)").fetchall()}
    adds = {
        "trust_score": "REAL NOT NULL DEFAULT 0",
        "embed_verified": "INTEGER NOT NULL DEFAULT 0",
        "embed_divergences": "INTEGER NOT NULL DEFAULT 0",
        "quarantined_until": "TEXT NOT NULL DEFAULT ''",
    }
    for col, decl in adds.items():
        if col not in have:
            conn.execute(f"ALTER TABLE devices ADD COLUMN {col} {decl}")
    conn.commit()


def _migrate_embed_jobs_audit_col(conn: DBAdapter) -> None:
    """Add embed_jobs.is_audit — migration v26->v27. Idempotent (PRAGMA table_info gate)."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "embed_jobs" not in tables:
        return
    have = {r[1] for r in conn.execute("PRAGMA table_info(embed_jobs)").fetchall()}
    if "is_audit" not in have:
        conn.execute("ALTER TABLE embed_jobs ADD COLUMN is_audit INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def _init_schema(conn: DBAdapter) -> None:
    """Ensure all tables and indexes exist, with schema versioning to skip DDL when current.

    Uses PRAGMA user_version to track schema version and avoid lock contention on
    repeated CLI invocations.
    """
    # Read current schema version from PRAGMA user_version (default 0 for new DBs)
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]

    # WHY(v14-Lehre): _migrate_codebook_collapse MUSS VOR dem Early-Return laufen
    # (analog _migrate_rename_research_questions). Auf v18-DBs ist user_version < 19,
    # also liegt kein Early-Return vor — aber die Prüfung hier macht es explizit
    # und sichert den Upgrade-Pfad für alle Zwischen-Versionen.
    _migrate_codebook_collapse(conn)

    # If schema is already at current version, skip all DDL and migrations (no-op)
    if current_version >= CURRENT_SCHEMA_VERSION:
        return

    # WHY(task-tracker): rename must run BEFORE CREATE TABLE IF NOT EXISTS
    # research_questions below. On a legacy DB that still has task_categories,
    # the executescript would create an empty research_questions first, making
    # the rename guard ("research_questions" not in tables) False — the real
    # data would stay stranded in task_categories. Running it here first means
    # the subsequent CREATE TABLE IF NOT EXISTS is a no-op on legacy DBs.
    _migrate_rename_research_questions(conn)

    # Schema is behind (or uninitialized) — run full DDL + migrations
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id       TEXT PRIMARY KEY,
            source_type     TEXT NOT NULL DEFAULT 'repo_file',
            repo            TEXT NOT NULL DEFAULT '',
            path            TEXT NOT NULL DEFAULT '',
            branch          TEXT NOT NULL DEFAULT 'main',
            "commit"        TEXT NOT NULL DEFAULT '',
            content_hash    TEXT NOT NULL DEFAULT '',
            captured_at     TEXT NOT NULL,
            visibility      TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'org', 'public', 'user')),
            org_id          TEXT DEFAULT NULL,
            user_id         TEXT DEFAULT NULL,
            scope_key       TEXT DEFAULT NULL,
            source_class    TEXT NOT NULL DEFAULT 'code'
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id            TEXT PRIMARY KEY,
            source_id           TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
            parent_chunk_id     TEXT,
            chunk_level         TEXT NOT NULL DEFAULT 'file',
            ordinal             INTEGER NOT NULL DEFAULT 0,
            start_offset        INTEGER NOT NULL DEFAULT 0,
            end_offset          INTEGER NOT NULL DEFAULT 0,
            text                TEXT NOT NULL DEFAULT '',
            text_hash           TEXT NOT NULL DEFAULT '',
            summary             TEXT NOT NULL DEFAULT '',
            category_labels     TEXT NOT NULL DEFAULT '',
            category_version    TEXT NOT NULL DEFAULT 'mayring-inductive-v1',
            embedding_model     TEXT NOT NULL DEFAULT '',
            embedding_id        TEXT NOT NULL DEFAULT '',
            quality_score       REAL NOT NULL DEFAULT 0.0,
            dedup_key           TEXT NOT NULL DEFAULT '',
            created_at          TEXT NOT NULL,
            superseded_by       TEXT,
            is_active           INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source_id
            ON chunks(source_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_text_hash
            ON chunks(text_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_dedup_key
            ON chunks(dedup_key);
        CREATE INDEX IF NOT EXISTS idx_chunks_is_active
            ON chunks(is_active);

        -- #workspace-uuid-sot (v7): Project = Sub-Dimension eines Workspace
        -- (User-Diagramm: User → Workspace(UUID, owner) → Project(Project-ID)).
        -- Ersetzt die alten fake-separaten Projekt-workspace_ids.
        CREATE TABLE IF NOT EXISTS projects (
            id            TEXT PRIMARY KEY,
            workspace_id  TEXT NOT NULL,
            name          TEXT NOT NULL DEFAULT '',
            -- #workspace-uuid-sot v8 (User-Diagramm): ein Projekt hat einen Owner
            -- + eine Source (github-Repo ODER papers/research). owner_id =
            -- app.linn.games user id; source_type ∈ {github, papers, research, …};
            -- source_ref = Repo-URL / Paper-IDs / freie Referenz.
            owner_id      TEXT,
            source_type   TEXT,
            source_ref    TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_projects_workspace
            ON projects(workspace_id);
        -- NOTE: idx_projects_source (workspace_id, source_type, source_ref) is
        -- created AFTER _migrate_schema() below — on a pre-v8 DB those columns
        -- are back-filled there via ALTER, so an inline CREATE INDEX here crashed
        -- legacy DBs with "no such column: source_type".

        -- C1 (project-groups v17): named, colored project groups per workspace.
        -- group_id on projects is FK→project_groups.id (NULL = no group).
        -- NOTE: idx_project_groups_workspace is inline-safe here because this
        -- table is entirely new in v17 — no pre-v17 DB has it, so CREATE TABLE
        -- IF NOT EXISTS is a real create, not a no-op.
        CREATE TABLE IF NOT EXISTS project_groups (
            id            TEXT PRIMARY KEY,
            workspace_id  TEXT NOT NULL,
            name          TEXT NOT NULL,
            color         TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_project_groups_workspace
            ON project_groups(workspace_id);

        -- C3 (project-scoped memory v18): 1-to-many chunk↔project links.
        -- origin_ref is nested-repo-aware (e.g. a submodule path that pins
        -- WHICH project a chunk belongs to inside a monorepo). source records
        -- HOW the link was made (ingest, manual, …). Inject priorises the
        -- session project via a DETERMINISTIC boost (retrieval._PROJECT_MATCH_
        -- BOOST), never a hard filter — global/unlinked knowledge stays visible.
        -- NOTE: indexes are inline-safe here because the table is entirely new
        -- in v18 — no pre-v18 DB has it, so CREATE TABLE is a real create.
        CREATE TABLE IF NOT EXISTS chunk_project_links (
            chunk_id      TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            project_id    TEXT NOT NULL,
            origin_ref    TEXT NOT NULL DEFAULT '',
            source        TEXT NOT NULL DEFAULT '',
            workspace_id  TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            PRIMARY KEY (chunk_id, project_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_project_links_ws_project
            ON chunk_project_links(workspace_id, project_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_project_links_chunk
            ON chunk_project_links(chunk_id);

        -- v19 (drop-codebook): categories (flach, kein codebook_id, UNIQUE(name)).
        -- codebooks-Tabelle entfernt. Bestehende DBs werden per _migrate_codebook_collapse
        -- migriert (codebook_categories → categories); frische DBs erhalten direkt die neue DDL.
        -- Category-Embeddings leben in der Chroma-Collection "codebook_categories" (Name
        -- bleibt für Back-Compat, da Chroma-Collections nicht atomar umbenannt werden können).
        CREATE TABLE IF NOT EXISTS categories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            parent_id       INTEGER REFERENCES categories(id),
            description     TEXT NOT NULL DEFAULT '',
            examples        TEXT NOT NULL DEFAULT '[]',
            status          TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active','proposed','deprecated')),
            source          TEXT NOT NULL DEFAULT 'imported'
                            CHECK (source IN ('manual','induced','imported')),
            evidence_count  INTEGER NOT NULL DEFAULT 0,
            embedding_id    TEXT NOT NULL DEFAULT '',
            risk_level      TEXT NOT NULL DEFAULT '',
            languages       TEXT NOT NULL DEFAULT '[]',
            patterns        TEXT NOT NULL DEFAULT '[]',
            promoted_at     TEXT,
            project_id      TEXT DEFAULT NULL,
            -- v21: Goal-Anker (Mayring goal-anchored). NULL = Legacy/geteilte Basis
            -- (vor v21 induziert, kein Goal bekannt).
            goal_id         INTEGER REFERENCES goals(id),
            -- v22: goal-gescopte Eindeutigkeit — derselbe Kategoriename darf unter
            -- verschiedenen Goals existieren (kein Cross-Goal-Merge mehr). Für die
            -- Legacy/Basis (goal_id IS NULL) erzwingt der partielle Unique-Index
            -- ux_categories_base_name weiter globale Eindeutigkeit (SQLite behandelt
            -- NULL in UNIQUE als distinct → ohne den Index keine Basis-Dedup).
            UNIQUE (goal_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_categories_status
            ON categories(status);
        -- idx_categories_goal wird NACH _migrate_schema angelegt (v21): auf einer
        -- Legacy-DB existiert categories.goal_id erst nach dem ALTER. Index hier im
        -- DDL-Block würde auf pre-v21-DBs "no such column: goal_id" werfen
        -- (v14/v19-Lehre — Index nie vor der spaltenanlegenden Migration).
        CREATE TABLE IF NOT EXISTS codebook_proposals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id     INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            pi_job_id       TEXT NOT NULL DEFAULT '',
            chunk_id        TEXT,
            paraphrase      TEXT NOT NULL DEFAULT '',
            parent_hint_id  INTEGER REFERENCES categories(id),
            proposed_at     TEXT NOT NULL,
            reviewed_by     TEXT,
            decision        TEXT CHECK (decision IN ('promote','merge','reject') OR decision IS NULL)
        );
        CREATE TABLE IF NOT EXISTS chunk_categories (
            chunk_id          TEXT NOT NULL,
            category_id       INTEGER NOT NULL REFERENCES categories(id),
            codebook_version  INTEGER NOT NULL DEFAULT 1,
            confidence        REAL NOT NULL DEFAULT 0.0,
            source            TEXT NOT NULL DEFAULT 'deductive'
                              CHECK (source IN ('deductive','inductive','hybrid-merge')),
            span_start        INTEGER,
            span_end          INTEGER,
            PRIMARY KEY (chunk_id, category_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_cat_cat ON chunk_categories(category_id);

        CREATE TABLE IF NOT EXISTS chunk_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            signal          TEXT NOT NULL,
            metadata        TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_chunk_id
            ON chunk_feedback(chunk_id);

        CREATE TABLE IF NOT EXISTS chunk_source_refs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_chunk_id  TEXT NOT NULL,
            source_id       TEXT NOT NULL,
            workspace_id    TEXT NOT NULL DEFAULT 'default',
            created_at      TEXT NOT NULL,
            UNIQUE(canonical_chunk_id, source_id)
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_canonical
            ON chunk_source_refs(canonical_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_source
            ON chunk_source_refs(source_id);

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id       TEXT NOT NULL DEFAULT '',
            event_type      TEXT NOT NULL,
            payload         TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ingestion_log_source_id
            ON ingestion_log(source_id);

        CREATE TABLE IF NOT EXISTS wiki_paper_cache (
            source_id    TEXT NOT NULL,
            rule_name    TEXT NOT NULL,
            extracted    TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (source_id, rule_name)
        );

        CREATE TABLE IF NOT EXISTS trigger_stats (
            trigger_id   TEXT PRIMARY KEY,
            fire_count   INTEGER NOT NULL DEFAULT 0,
            ref_count    INTEGER NOT NULL DEFAULT 0,
            is_active    INTEGER NOT NULL DEFAULT 1,
            last_fired   TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS pi_jobs (
            job_id              TEXT PRIMARY KEY,
            task_text           TEXT NOT NULL,
            repo_slug           TEXT NOT NULL DEFAULT '',
            workspace_id        TEXT NOT NULL DEFAULT 'default',
            status              TEXT NOT NULL DEFAULT 'queued',
            prefer              TEXT NOT NULL DEFAULT 'auto',
            ollama_url          TEXT NOT NULL DEFAULT '',
            model               TEXT NOT NULL DEFAULT '',
            result_json         TEXT NOT NULL DEFAULT '',
            error               TEXT NOT NULL DEFAULT '',
            timeout_s           REAL NOT NULL DEFAULT 180.0,
            scope               TEXT NOT NULL DEFAULT 'local',
            capability_required TEXT NOT NULL DEFAULT '',
            claimed_by          TEXT NOT NULL DEFAULT '',
            claimed_at          TEXT NOT NULL DEFAULT '',
            created_at          TEXT NOT NULL,
            started_at          TEXT NOT NULL DEFAULT '',
            finished_at         TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_pi_jobs_status ON pi_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_pi_jobs_scope ON pi_jobs(scope);
        CREATE INDEX IF NOT EXISTS idx_pi_jobs_created ON pi_jobs(created_at);

        CREATE TABLE IF NOT EXISTS context_feedback_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_ids      TEXT NOT NULL,
            context_text     TEXT NOT NULL,
            was_referenced   INTEGER NOT NULL,
            led_to_retrieval INTEGER NOT NULL,
            relevance_score  REAL NOT NULL,
            captured_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topic_transitions (
            from_topic  TEXT NOT NULL,
            to_topic    TEXT NOT NULL,
            count       INTEGER NOT NULL DEFAULT 1,
            last_seen   TEXT NOT NULL,
            PRIMARY KEY (from_topic, to_topic)
        );

        CREATE TABLE IF NOT EXISTS llm_calls_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            call_type    TEXT NOT NULL DEFAULT 'pi_task',
            model        TEXT NOT NULL DEFAULT '',
            prompt       TEXT NOT NULL DEFAULT '',
            response     TEXT NOT NULL DEFAULT '',
            tool_calls   INTEGER NOT NULL DEFAULT 0,
            duration_ms  INTEGER NOT NULL DEFAULT 0,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            created_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_llm_calls_created
            ON llm_calls_log(created_at);

        -- source_goals (v20): der goal-anchor (Mayring Selektionskriterium), gegen
        -- den die Chunks einer Source kategorisiert wurden. Die kanonische Methode
        -- ist goal-anchored, aber das goal war bisher transient (nur lose in
        -- llm_calls_log) → treues Re-Labeling unmöglich, Fallback-anchored Chunks
        -- nicht auditierbar. Eigene Tabelle statt sources-Spalte: die sources-Tabelle
        -- wird vom v15-visibility-Rebuild umgebaut (fragil) — entkoppeln. Ein
        -- "Inhalte aus …"-goal markiert die schwachen Fallback-Anker.
        CREATE TABLE IF NOT EXISTS source_goals (
            source_id   TEXT PRIMARY KEY,
            goal        TEXT NOT NULL DEFAULT '',
            updated_at  TEXT NOT NULL DEFAULT ''
        );

        -- goals (v21): Goal als FIRST-CLASS Entity = das Mayring-Selektionskriterium.
        -- Bisher lag das Goal nur als Freitext pro Source (source_goals) → derselbe
        -- konzeptuelle Goal mehrfach erfasst, und Kategorien hatten KEINEN Goal-Anker
        -- (global UNIQUE(name), project_id nie gesetzt) → goal→category-Verknüpfung
        -- nur transitiv+degeneriert rekonstruierbar. goals dedupliziert kanonisch per
        -- Embedding-Cosine (wie research_questions/Kategorien); categories.goal_id
        -- ankert jede Kategorie an ihr Goal → goal-anchored Mayring-Methode.
        CREATE TABLE IF NOT EXISTS goals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            text             TEXT NOT NULL,
            workspace_id     TEXT NOT NULL DEFAULT 'default',
            project_id       TEXT DEFAULT NULL,
            embedding_id     TEXT NOT NULL DEFAULT '',
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT NOT NULL DEFAULT '',
            updated_at       TEXT NOT NULL DEFAULT '',
            UNIQUE (workspace_id, text)
        );
        CREATE INDEX IF NOT EXISTS idx_goals_workspace ON goals(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_goals_project ON goals(project_id);

        -- Workspace-Identity-Foundation. Bisher war workspace_id ein
        -- freier String den jeder Code-Pfad anders auflöste:
        --   * JWT-Pfad: f"user-{sub}" (sub = app.linn.games User.id)
        --   * CLI/workflow-Pfad: getattr(args, "workspace_id", "default")
        -- → derselbe Mensch landete in zwei verschiedenen Buckets je nach
        -- Eingangsweg. Diese Tabelle macht workspace_id zu einem
        -- typisierten First-Class-Citizen (kind=user|team|project),
        -- erlaubt Sub-Workspaces über parent_id (z.B. user-2:mayringcoder
        -- als Projekt-Sub-Bucket unter user-2) und ist damit die
        -- Foundation für Multi-Tenant + Multi-User.
        CREATE TABLE IF NOT EXISTS workspaces (
            id               TEXT PRIMARY KEY,
            kind             TEXT NOT NULL CHECK(kind IN ('user', 'team', 'project', 'system')),
            parent_id        TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
            owner_user_id    INTEGER,
            email            TEXT,
            display_name     TEXT NOT NULL DEFAULT '',
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_workspaces_parent
            ON workspaces(parent_id);
        CREATE INDEX IF NOT EXISTS idx_workspaces_owner
            ON workspaces(owner_user_id);

        -- Aliases ermöglichen Migration ohne Datenverlust: alte Aufrufe
        -- mit workspace_id="default" oder "nileneb-mayringcoder" werden
        -- über diese Tabelle auf den kanonischen Workspace abgebildet.
        -- WHY(2026-05-11, task-categorization): Mayring-konforme
        -- forschungsfragen-basierte Kategorisierung. Vor diesem schema
        -- haben chunks tech-labels (api, ui, data_access) bekommen die
        -- weder retrieval-relevant noch self-organisierend waren — LLM
        -- erfand täglich neue [neu]label, kein cluster-effekt.
        -- research_questions: 1 row per durch derive_research_question() identifizierte
        -- forschungsfrage. Aus user-prompts emergierend, semantisch
        -- geclustert via embedding-similarity (>0.85 = same question).
        CREATE TABLE IF NOT EXISTS research_questions (
            research_question_id   TEXT PRIMARY KEY,
            title                  TEXT NOT NULL,
            embedding_id           TEXT NOT NULL DEFAULT '',
            parent_research_question_id TEXT REFERENCES research_questions(research_question_id) ON DELETE SET NULL,
            occurrence_count       INTEGER NOT NULL DEFAULT 1,
            first_seen_at          TEXT NOT NULL,
            last_used_at           TEXT NOT NULL,
            workspace_id           TEXT NOT NULL DEFAULT 'default'
        );

        CREATE INDEX IF NOT EXISTS idx_research_questions_workspace
            ON research_questions(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_research_questions_last_used
            ON research_questions(last_used_at);

        -- research_question_chunk_links: gibt einem chunk ein relevance-score für eine
        -- research_question (geboostet wenn der chunk bei einer ähnlichen frage schon
        -- positive feedback bekam). Pro research_question & chunk genau eine row.
        CREATE TABLE IF NOT EXISTS research_question_chunk_links (
            research_question_id  TEXT NOT NULL REFERENCES research_questions(research_question_id) ON DELETE CASCADE,
            chunk_id              TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            relevance_score       REAL NOT NULL DEFAULT 1.0,
            created_at            TEXT NOT NULL,
            PRIMARY KEY (research_question_id, chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_rq_chunk_links_chunk
            ON research_question_chunk_links(chunk_id);

        CREATE TABLE IF NOT EXISTS workspace_aliases (
            alias            TEXT PRIMARY KEY,
            workspace_id     TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_workspace_aliases_target
            ON workspace_aliases(workspace_id);

        CREATE TABLE IF NOT EXISTS tasks (
            task_id        TEXT PRIMARY KEY,
            workspace_id   TEXT NOT NULL,
            title          TEXT NOT NULL,
            description    TEXT NOT NULL DEFAULT '',
            status         TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','in_progress','done')),
            priority       TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low','medium','high')),
            due_date       TEXT,
            tags           TEXT NOT NULL DEFAULT '',
            created_by     TEXT,
            linked_chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
            scope_key      TEXT,
            derive_embedding TEXT,
            external_id    TEXT,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            completed_at   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_workspace_status ON tasks(workspace_id, status);
        CREATE INDEX IF NOT EXISTS idx_tasks_workspace_due ON tasks(workspace_id, due_date);
        -- NOTE(incident 2026-05-25): idx_tasks_external_id is created in the
        -- version-gated v14 block below (after a defensive ALTER), NOT inline
        -- here — on an EXISTING DB this CREATE TABLE is a no-op so external_id
        -- isn't present yet, and an inline index on it aborted the whole
        -- executescript ("no such column") → init raised → every request 500'd.

        -- #274: Device-Registry. Cloud-seitiges Gegenstück zum Plugin-Device-
        -- Kanal (mayring-claude-plugin#5). Das Plugin/der Pi-Worker schickt eine
        -- stabile device_id (X-Device-Id-Header); die Cloud persistiert hier
        -- Gerät + Capabilities. `capabilities` ist comma-separated (z.B.
        -- "local-gpu,write"). Diese Tabelle ist die SoT für Write-Job-Routing:
        -- ein capability_required='write'-Job wird nur an ein hier mit 'write'
        -- registriertes Gerät zugewiesen (join pi_jobs.claimed_by ↔ device_id).
        -- PK ist (device_id, workspace_id): dasselbe Gerät kann pro Workspace
        -- eine eigene Registrierung haben, und ein re-register in workspace B
        -- verschiebt NICHT stillschweigend die workspace-A-Zeile (alle Lookups
        -- sind workspace-scoped). Migration v5→v6: _migrate_devices_composite_pk.
        CREATE TABLE IF NOT EXISTS devices (
            device_id    TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            name         TEXT NOT NULL DEFAULT '',
            os           TEXT NOT NULL DEFAULT '',
            capabilities TEXT NOT NULL DEFAULT '',
            last_seen    TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'active',
            created_at   TEXT NOT NULL DEFAULT '',
            trust_score       REAL NOT NULL DEFAULT 0,
            embed_verified    INTEGER NOT NULL DEFAULT 0,
            embed_divergences INTEGER NOT NULL DEFAULT 0,
            quarantined_until TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (device_id, workspace_id)
        );
        CREATE INDEX IF NOT EXISTS idx_devices_workspace ON devices(workspace_id);

        -- #274: Hook-Firings pro Gerät (Observability-Dashboard "Plugin").
        -- best-effort insert vom Stop/UserPromptSubmit-Hook — payload ist ein
        -- kleines JSON-summary, KEIN voller Transcript-Dump.
        CREATE TABLE IF NOT EXISTS hook_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            device_id    TEXT NOT NULL DEFAULT '',
            hook_type    TEXT NOT NULL DEFAULT '',
            fired_at     TEXT NOT NULL DEFAULT '',
            payload      TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_hook_events_ws_fired
            ON hook_events(workspace_id, fired_at);

        -- #365 Schicht 3: Distributed embedding pool. One row per embed task with
        -- two claim slots (A/B); two DISTINCT devices each fill a slot and the
        -- vectors get cross-checked. Canonical schema; the lazy ensure_tables in
        -- embed_pool.py mirrors this (raw-conn tests + safety net for v26 DBs that
        -- predate this table, since _init_schema early-returns at user_version>=26).
        CREATE TABLE IF NOT EXISTS embed_jobs (
            embed_id     TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            projekt_id   TEXT NOT NULL DEFAULT '',
            text         TEXT NOT NULL,
            chunk_ref    TEXT NOT NULL DEFAULT '',
            model        TEXT NOT NULL DEFAULT '',
            is_golden    INTEGER NOT NULL DEFAULT 0,
            golden_ref   TEXT NOT NULL DEFAULT '',
            is_audit     INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL DEFAULT 'queued',
            device_a     TEXT NOT NULL DEFAULT '',
            result_a     TEXT NOT NULL DEFAULT '',
            device_b     TEXT NOT NULL DEFAULT '',
            result_b     TEXT NOT NULL DEFAULT '',
            cosine       REAL,
            verdict      TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL DEFAULT '',
            verified_at  TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_embed_jobs_ws_status
            ON embed_jobs(workspace_id, status, created_at);

        -- notification_state (v23): Seen/Ack-Status je hook_event für das
        -- Ampel-Notification-Center. DRY: hook_events bleibt Source-of-Truth
        -- (Klassifikation + repo/project-Zuordnung deterministisch on-read), hier
        -- liegt NUR der user-facing Triage-State. Ein fehlender Eintrag = ungelesen.
        CREATE TABLE IF NOT EXISTS notification_state (
            hook_event_id INTEGER PRIMARY KEY REFERENCES hook_events(id) ON DELETE CASCADE,
            workspace_id  TEXT NOT NULL DEFAULT 'default',
            seen          INTEGER NOT NULL DEFAULT 0,
            acked         INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_notification_state_ws
            ON notification_state(workspace_id, acked);
    """)

    # v16 (tenancy phase B): per-workspace role permission overrides.
    # Absence of a row means the DEFAULT_MATRIX from authz.py applies.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_role_permissions (
            workspace_id TEXT NOT NULL,
            role         TEXT NOT NULL CHECK(role IN ('user','editor','admin')),
            permission   TEXT NOT NULL,
            allowed      INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT,
            PRIMARY KEY (workspace_id, role, permission)
        )
        """
    )

    # Migration: add missing columns to existing DBs
    _migrate_schema(conn)

    # Migration v5→v6: devices PK device_id → (device_id, workspace_id) (#274 review)
    _migrate_devices_composite_pk(conn)

    # Migration v25→v26: embed-pool trust/quarantine cols (#365). MUSS nach dem
    # composite-PK-Rebuild laufen — der rebuildet devices ohne diese Spalten.
    _migrate_devices_embed_pool_cols(conn)

    # Migration v26→v27: embed_jobs.is_audit (#365 house-audit).
    _migrate_embed_jobs_audit_col(conn)

    # v22: categories UNIQUE(name) → UNIQUE(goal_id, name). MUSS nach _migrate_schema
    # (goal_id-ALTER) laufen.
    _migrate_categories_goal_unique(conn)

    # v21/v22: categories-Indizes — NACH _migrate_schema/-rebuild, damit goal_id auf
    # Legacy-DBs existiert (v14/v19-Lehre). Der partielle Unique-Index hält die
    # Basis (goal_id IS NULL) global eindeutig (SQLite-NULL-Distinctness umgehen).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_categories_goal ON categories(goal_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_categories_base_name "
        "ON categories(name) WHERE goal_id IS NULL"
    )

    # Indexes for workspace_id filtering
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id ON chunks(workspace_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sources_workspace_id ON sources(workspace_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sources_scope_key ON sources(scope_key)"
    )
    # Project Router (Slice 1): match by cwd-git-remote → source_ref. Created
    # here (not in the DDL block) because _migrate_schema back-fills projects'
    # source_type/source_ref on pre-v8 DBs — same migration-order class as v14.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_projects_source "
        "ON projects(workspace_id, source_type, source_ref)"
    )
    # v12: range-scan the 24h injections COUNT in /stats/feedback-log.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_context_feedback_captured "
        "ON context_feedback_log(captured_at)"
    )
    # v13: index recent-ops' ORDER BY created_at DESC LIMIT (uncached live feed).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_log_created "
        "ON ingestion_log(created_at)"
    )
    # v14: tasks.external_id for idempotent agent-task capture (PostToolUse hook).
    # WHY(migration-order incident 2026-05-25): on an EXISTING (v13) DB the
    # CREATE TABLE above is a no-op, so external_id isn't present here yet — the
    # generic column-migration adds it elsewhere. Indexing it first raised
    # "no such column: external_id", which aborted the WHOLE migration → prod
    # stuck at v13 → init_memory_db raised → every get_conn 500'd (search down).
    # Fresh-DB tests missed it (CREATE TABLE adds the column). Add it defensively
    # right before indexing so this block is self-contained for both paths.
    if "external_id" not in {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}:
        conn.execute("ALTER TABLE tasks ADD COLUMN external_id TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_external_id ON tasks(workspace_id, external_id)"
    )

    # v24 (#270-Cleanup): das alte integration_notifications-Subsystem (MayringCoder
    # src/api/integrations/notifications_store.py) ist entfernt — kein Producer hat je
    # darauf geschrieben (lebende Pipeline = GitHub-Action → /repo-events → hook_events).
    # Die leere Tabelle bleibt sonst inert in jeder Prod-DB liegen. DROP IF EXISTS ist
    # idempotent (no-op auf Fresh-DBs, die sie nie hatten).
    conn.execute("DROP TABLE IF EXISTS integration_notifications")

    # System-Workspace seedet sich beim DB-init. Verwendet von Service-
    # Tokens (src/api/auth.py:37), post-deploy-ingest, conversation_watcher,
    # ambient-Snapshots. Ohne diese Row würde der workspace_resolver mit
    # UnknownWorkspaceError werfen, sobald irgendein Cron-Job läuft.
    _now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, display_name,
                                   created_at, updated_at)
           VALUES ('system', 'system', NULL,
                   'System (Service-Token / Cron)', ?, ?)
           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
        (_now, _now),
    )

    # v15 (tenancy phase A): re-sort the visibility axis (user->private,
    # org/team-workspace private->org, personal-workspace private gets owner
    # stamped) and tighten the sources.visibility CHECK to 3 values. Runs here
    # — before the PRAGMA bump — so a half-migrated DB re-runs it on next boot.
    if current_version < 15:
        migrate_visibility_axis(conn, personal_owner=_personal_owner_map(conn))

    # Update schema version in PRAGMA user_version to mark migration as complete
    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")

    conn.commit()


# ---------------------------------------------------------------------------
# Role permission overrides (tenancy phase B)
# ---------------------------------------------------------------------------

def get_role_permissions(conn, workspace_id: str) -> "dict[tuple[str, str], bool]":
    """All per-workspace overrides. Empty dict = use authz.DEFAULT_MATRIX."""
    rows = conn.execute(
        "SELECT role, permission, allowed FROM workspace_role_permissions WHERE workspace_id = ?",
        (workspace_id,),
    ).fetchall()
    out: "dict[tuple[str, str], bool]" = {}
    for r in rows:
        role = r["role"] if hasattr(r, "keys") else r[0]
        perm = r["permission"] if hasattr(r, "keys") else r[1]
        allowed = r["allowed"] if hasattr(r, "keys") else r[2]
        out[(role, perm)] = bool(allowed)
    return out


def set_role_permission(conn, workspace_id: str, role: str, permission: str, allowed: bool) -> None:
    conn.execute(
        "INSERT INTO workspace_role_permissions (workspace_id, role, permission, allowed, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT(workspace_id, role, permission) DO UPDATE SET allowed = excluded.allowed, updated_at = excluded.updated_at",
        (workspace_id, role, permission, 1 if allowed else 0),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Source operations
# ---------------------------------------------------------------------------

def upsert_source(
    conn: DBAdapter, source: Source, workspace_id: str = "default",
    visibility: str | None = None, org_id: str | None = None,
    user_id: str | None = None, scope_key: str | None = None,
) -> None:
    """Insert/update a source row.

    visibility/org_id/user_id/scope_key default to the values on the Source
    dataclass when not explicitly passed; this lets callers attach them at
    Source construction time and have them flow all the way to the DB.
    """
    eff_visibility = visibility if visibility is not None else source.visibility
    eff_org_id = org_id if org_id is not None else source.org_id
    eff_user_id = user_id if user_id is not None else source.user_id
    eff_scope_key = scope_key if scope_key is not None else source.scope_key
    # WHY(owner-attribution 2026-06-08): a private source with no user_id is
    # unreachable — build_chroma_where / _scope_filter require user_id = caller, so it
    # matches nobody (dead, ungoverned data). Service-token ingests (populate, ambient,
    # repo-events; info.sub=None) used to write private+user_id=NULL here. Fall back to
    # the workspace owner at the single persist chokepoint so EVERY path is covered, not
    # just the endpoints that remembered to stamp it (/memory/put, /conversation).
    if eff_visibility == "private" and not eff_user_id:
        try:
            from mayring_core.identity.workspace_resolver import workspace_owner
            eff_user_id = workspace_owner(conn, workspace_id) or eff_user_id
        except Exception:  # noqa: BLE001 — owner lookup must never block ingest
            pass
    eff_source_class = source.source_class or "code"
    conn.execute(
        """
        INSERT INTO sources
            (source_id, source_type, repo, path, branch, "commit", content_hash,
             captured_at, workspace_id, visibility, org_id, user_id, scope_key,
             source_class)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            source_type  = excluded.source_type,
            repo         = excluded.repo,
            path         = excluded.path,
            branch       = excluded.branch,
            "commit"     = excluded."commit",
            content_hash = excluded.content_hash,
            captured_at  = excluded.captured_at,
            workspace_id = excluded.workspace_id,
            visibility   = excluded.visibility,
            org_id       = excluded.org_id,
            user_id      = excluded.user_id,
            scope_key    = excluded.scope_key,
            source_class = excluded.source_class
        """,
        (
            source.source_id, source.source_type, source.repo, source.path,
            source.branch, source.commit, source.content_hash, source.captured_at,
            workspace_id, eff_visibility, eff_org_id, eff_user_id, eff_scope_key,
            eff_source_class,
        ),
    )
    _maybe_commit(conn)


def get_source(conn: DBAdapter, source_id: str) -> Source | None:
    row = conn.execute(
        "SELECT * FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    return Source(
        source_id=d["source_id"],
        source_type=d["source_type"],
        repo=d["repo"],
        path=d["path"],
        branch=d["branch"],
        commit=d["commit"],
        content_hash=d["content_hash"],
        captured_at=d["captured_at"],
        visibility=d.get("visibility") or "private",
        org_id=d.get("org_id"),
        user_id=d.get("user_id"),
        scope_key=d.get("scope_key"),
        source_class=d.get("source_class") or "code",
    )


# ---------------------------------------------------------------------------
# Chunk operations
# ---------------------------------------------------------------------------

def insert_chunk(
    conn: DBAdapter, chunk: Chunk, workspace_id: str | None = None,
) -> None:
    """Insert a new chunk. category_labels stored as comma-joined string."""
    _ws = workspace_id if workspace_id is not None else chunk.workspace_id
    labels_str = ",".join(chunk.category_labels)
    conn.execute(
        """
        INSERT INTO chunks
            (chunk_id, source_id, parent_chunk_id, chunk_level, ordinal,
             start_offset, end_offset, text, text_hash, summary, category_labels,
             category_version, embedding_model, embedding_id, quality_score,
             dedup_key, category_source, category_confidence,
             created_at, superseded_by, is_active, workspace_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            text=excluded.text, text_hash=excluded.text_hash,
            summary=excluded.summary, category_labels=excluded.category_labels,
            category_source=excluded.category_source,
            category_confidence=excluded.category_confidence,
            is_active=1, superseded_by=NULL, created_at=excluded.created_at,
            workspace_id=excluded.workspace_id
        """,
        (
            chunk.chunk_id, chunk.source_id, chunk.parent_chunk_id,
            chunk.chunk_level, chunk.ordinal, chunk.start_offset, chunk.end_offset,
            chunk.text, chunk.text_hash, chunk.summary, labels_str,
            chunk.category_version, chunk.embedding_model, chunk.embedding_id,
            chunk.quality_score, chunk.dedup_key,
            chunk.category_source, chunk.category_confidence,
            chunk.created_at, chunk.superseded_by, int(chunk.is_active), _ws,
        ),
    )
    _maybe_commit(conn)


def link_chunk_to_project(
    conn: DBAdapter,
    chunk_id: str,
    project_id: str,
    *,
    origin_ref: str = "",
    source: str = "",
    workspace_id: str,
    created_at: str | None = None,
) -> None:
    """Link a chunk to a project (C3 v18, 1-to-many via chunk_project_links).

    Idempotent: the PK is (chunk_id, project_id), so a repeated link is a
    no-op via INSERT OR IGNORE. origin_ref is nested-repo-aware (which
    sub-project a chunk belongs to inside a monorepo); source records HOW the
    link was made (e.g. 'ingest', 'manual').
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO chunk_project_links
            (chunk_id, project_id, origin_ref, source, workspace_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, project_id, origin_ref, source, workspace_id,
         created_at or _now_iso()),
    )
    _maybe_commit(conn)


def project_linked_chunk_ids(
    conn: DBAdapter,
    chunk_ids: list[str],
    project_id: str,
    workspace_id: str,
) -> set[str]:
    """Return the subset of chunk_ids linked to project_id within workspace_id.

    One bulk query (workspace-scoped). Empty chunk_ids or empty project_id
    short-circuit to an empty set. Used by retrieval for the deterministic
    project_match boost — NOT a hard filter.
    """
    if not chunk_ids or not project_id:
        return set()
    out: set[str] = set()
    BATCH = 900  # stay under SQLite's 999 variable limit (+2 fixed params)
    for i in range(0, len(chunk_ids), BATCH):
        batch = chunk_ids[i:i + BATCH]
        ph = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT chunk_id FROM chunk_project_links "
            f"WHERE project_id = ? AND workspace_id = ? AND chunk_id IN ({ph})",
            (project_id, workspace_id, *batch),
        ).fetchall()
        out.update(r[0] for r in rows)
    return out


def get_chunk(conn: DBAdapter, chunk_id: str, active_only: bool = True) -> Chunk | None:
    query = "SELECT * FROM chunks WHERE chunk_id = ?"
    if active_only:
        query += " AND is_active = 1"
    row = conn.execute(query, (chunk_id,)).fetchone()
    if row is None:
        return None
    return Chunk.from_dict(dict(row))


def get_chunks_bulk(
    conn: DBAdapter, chunk_ids: list[str], active_only: bool = True
) -> list[Chunk]:
    """Load many chunks in ONE batched query instead of N single reads.

    WHY(#workspace-uuid-sot perf): the retrieval candidate-load looped get_chunk()
    per chunk_id — with a whole-workspace candidate set (4000+) that was ~11s of
    SQLite roundtrips per search → hook timeouts. Bulk IN-query, batched to stay
    under SQLite's variable limit (999)."""
    if not chunk_ids:
        return []
    out: list[Chunk] = []
    BATCH = 900
    for i in range(0, len(chunk_ids), BATCH):
        batch = chunk_ids[i:i + BATCH]
        ph = ",".join("?" * len(batch))
        q = f"SELECT * FROM chunks WHERE chunk_id IN ({ph})"
        if active_only:
            q += " AND is_active = 1"
        rows = conn.execute(q, batch).fetchall()
        out.extend(Chunk.from_dict(dict(r)) for r in rows)
    return out


def get_chunks_by_source(
    conn: DBAdapter, source_id: str, active_only: bool = True
) -> list[Chunk]:
    query = "SELECT * FROM chunks WHERE source_id = ?"
    params: list = [source_id]
    if active_only:
        query += " AND is_active = 1"
    query += " ORDER BY ordinal"
    rows = conn.execute(query, params).fetchall()
    return [Chunk.from_dict(dict(r)) for r in rows]


def find_by_text_hash(
    conn: DBAdapter, text_hash: str, workspace_id: str = "default"
) -> Chunk | None:
    """Exact dedup: return first active chunk with this text_hash in workspace, or None."""
    row = conn.execute(
        "SELECT * FROM chunks WHERE text_hash = ? AND is_active = 1 AND workspace_id = ? LIMIT 1",
        (text_hash, workspace_id),
    ).fetchone()
    return Chunk.from_dict(dict(row)) if row else None


def supersede_chunk(
    conn: DBAdapter, old_chunk_id: str, new_chunk_id: str
) -> None:
    """Mark old chunk as inactive, point superseded_by to new chunk."""
    conn.execute(
        "UPDATE chunks SET is_active = 0, superseded_by = ? WHERE chunk_id = ?",
        (new_chunk_id, old_chunk_id),
    )
    _maybe_commit(conn)


def deactivate_chunks_by_source(
    conn: DBAdapter, source_id: str
) -> int:
    """Set is_active=0 for all active chunks of source_id. Returns count."""
    conn.execute(
        "UPDATE chunks SET is_active = 0 WHERE source_id = ? AND is_active = 1",
        (source_id,),
    )
    _maybe_commit(conn)
    return conn.changes()


def update_chunk_category_labels(
    conn: Any,
    chunk_ids: list[str],
    mapping: "dict[str, str]",
) -> int:
    """Apply S7 old→canonical label mapping to stored chunks.

    WHY: S7 operates on committed chunks after the ingest transaction. A targeted
    UPDATE is correct here — insert_chunk() would do a full upsert including
    text/hash fields which must not be touched for a label-only patch.

    Strips [neu] prefix before lookup so "[neu]auth-check" resolves via "auth-check".
    Sets category_source='s7-reduced' for auditability.
    Returns count of actually-modified rows.
    """
    if not chunk_ids or not mapping:
        return 0
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, category_labels FROM chunks "
        f"WHERE chunk_id IN ({placeholders}) AND is_active = 1",
        chunk_ids,
    ).fetchall()
    updated = 0
    for chunk_id, label_csv in rows:
        old_labels = [lbl.strip() for lbl in (label_csv or "").split(",") if lbl.strip()]
        new_labels = []
        for lbl in old_labels:
            key = lbl.lower().removeprefix("[neu]")
            new_labels.append(mapping.get(key, key))
        new_labels = list(dict.fromkeys(new_labels))
        new_csv = ",".join(new_labels)
        if new_csv != label_csv:
            conn.execute(
                "UPDATE chunks SET category_labels = ?, category_source = 's7-reduced' "
                "WHERE chunk_id = ?",
                (new_csv, chunk_id),
            )
            updated += 1
    _maybe_commit(conn)
    return updated


def log_context_injection(
    conn: Any,
    *,
    trigger_ids: Any,
    context_text: str,
    workspace_id: str = "default",
    query: str = "",
    stage_scores: Any = None,
    reranker_version: str = "v1",
    was_referenced: int = 0,
    led_to_retrieval: int = 0,
    relevance_score: float = 0.0,
) -> None:
    """DIE EINE Telemetrie-Funktion für context_feedback_log.

    ALLE Inject-Pfade (REST /memory/search, MCP search_memory, ambient) rufen NUR diese —
    mit dem VOLLEN Spaltensatz inkl. workspace_id + query + stage_scores + reranker_version.
    Vorher drei divergierte INSERTs: zwei (MCP, ambient) ließen workspace_id/query/stage_scores
    weg → mis-scopete, feature-lose Rows, die die workspace-gescopten Dashboards (Memory-
    Effizienz) UNTERZÄHLTEN (alle MCP-Connector-Injektionen unsichtbar) + fürs Reranker-Training
    nutzlos waren. Telemetrie: DB-Fehler (locked / Spalte fehlt) werden geloggt, nie
    hot-path-blockierend. trigger_ids/stage_scores werden zu JSON serialisiert wenn nötig."""
    ids = trigger_ids if isinstance(trigger_ids, str) else json.dumps(list(trigger_ids or []))
    stage = stage_scores if isinstance(stage_scores, str) else json.dumps(stage_scores or {})
    try:
        conn.execute(
            "INSERT INTO context_feedback_log "
            "(trigger_ids, context_text, was_referenced, led_to_retrieval, relevance_score, "
            " captured_at, query, stage_scores, workspace_id, reranker_version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ids, (context_text or "")[:2000], int(was_referenced), int(led_to_retrieval),
             float(relevance_score), datetime.now(timezone.utc).isoformat(),
             (query or "")[:1000], stage, workspace_id, reranker_version),
        )
        _maybe_commit(conn)
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as exc:
        import logging
        logging.getLogger(__name__).warning("context_feedback_log insert skipped: %s", exc)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def add_feedback(
    conn: DBAdapter,
    chunk_id: str,
    signal: str,
    metadata: dict | None = None,
) -> None:
    """Insert a feedback row. Rating-only — '1'..'5' string.

    WHY(2026-05-10 rating-migration): binary positive/negative komplett
    rausgenommen. Skalares rating gibt reranker echten gradient + erlaubt
    "wichtig aber nicht primär" auszudrücken.

    Raises ``ValueError`` für jeden anderen wert.
    """
    if signal not in ("1", "2", "3", "4", "5"):
        raise ValueError(
            f"add_feedback: signal must be a rating '1'..'5', got {signal!r}"
        )
    conn.execute(
        """
        INSERT INTO chunk_feedback (chunk_id, signal, metadata, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (chunk_id, signal, json.dumps(metadata or {}), _now_iso()),
    )

    # WHY(2026-05-11, task-categorization): bei rating>=4 die chunk-zu-task
    # link verstärken, sodass das beim nächsten ähnlichen prompt geboostet
    # wird. metadata muss task_id enthalten (vom memory_inject hook gesetzt).
    rating = int(signal)
    task_id = (metadata or {}).get("task_id") if metadata else None
    if rating >= 4 and isinstance(task_id, str) and task_id:
        try:
            from mayring_core.memory.task_derivation import link_chunk_to_research_question
            # Score: rating 4 = +0.5, rating 5 = +1.0. Verstärkt sich bei
            # wiederholtem positiv-feedback (clipped bei 5.0 im link).
            score = (rating - 3) * 0.5
            link_chunk_to_research_question(conn, task_id, chunk_id, relevance_score=score)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("task-chunk-link failed: %s", e)

    _maybe_commit(conn)


def get_feedback_score(conn: DBAdapter, chunk_id: str) -> float:
    """Average rating, normalized to [0..1]. No feedback → 0.5.

    Mapping: rating 1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0. Used by reranker
    feature `sf`. Default 0.5 (no rating) ≡ rating 3 (neutral), so chunks
    without history score the same as explicitly-neutral ones.
    """
    rows = conn.execute(
        "SELECT signal FROM chunk_feedback WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchall()
    if not rows:
        return 0.5
    ratings: list[int] = []
    for (s,) in rows:
        try:
            r = int(s)
            if 1 <= r <= 5:
                ratings.append(r)
        except (TypeError, ValueError):
            continue
    if not ratings:
        return 0.5
    avg = sum(ratings) / len(ratings)
    return max(0.0, min(1.0, (avg - 1.0) / 4.0))


# ---------------------------------------------------------------------------
# Ingestion log
# ---------------------------------------------------------------------------

def log_ingestion_event(
    conn: DBAdapter,
    source_id: str,
    event_type: str,
    payload: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_log (source_id, event_type, payload, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (source_id, event_type, json.dumps(payload), _now_iso()),
    )
    _maybe_commit(conn)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_active_chunk_count(conn: DBAdapter) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE is_active = 1"
    ).fetchone()[0]


def get_source_count(conn: DBAdapter) -> int:
    return conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]


# ---------------------------------------------------------------------------
# Cross-source references (dedup co-occurrence)
# ---------------------------------------------------------------------------

def add_source_ref(
    conn: DBAdapter,
    canonical_chunk_id: str,
    source_id: str,
    workspace_id: str = "default",
) -> None:
    """Record that source_id contains the same text as canonical_chunk_id."""
    conn.execute(
        """
        INSERT OR IGNORE INTO chunk_source_refs
            (canonical_chunk_id, source_id, workspace_id, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (canonical_chunk_id, source_id, workspace_id, _now_iso()),
    )
    _maybe_commit(conn)


def get_source_refs(
    conn: DBAdapter,
    canonical_chunk_id: str,
) -> list[str]:
    """Return all source_ids that share the same text as canonical_chunk_id."""
    rows = conn.execute(
        "SELECT source_id FROM chunk_source_refs WHERE canonical_chunk_id = ?",
        (canonical_chunk_id,),
    ).fetchall()
    return [r[0] for r in rows]


def log_llm_call(
    conn: DBAdapter,
    call_type: str,
    model: str,
    prompt: str,
    response: str,
    tool_calls: int = 0,
    duration_ms: int = 0,
    workspace_id: str = "default",
) -> None:
    from datetime import datetime, timezone
    conn.execute(
        "INSERT INTO llm_calls_log"
        " (call_type, model, prompt, response, tool_calls, duration_ms, workspace_id, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (call_type, model, prompt[:500], response[:800],
         tool_calls, duration_ms, workspace_id,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def log_llm_call_async(
    call_type: str,
    model: str,
    prompt: str,
    response: str,
    tool_calls: int = 0,
    duration_ms: int = 0,
    workspace_id: str = "default",
    db_path: "Path | None" = None,
) -> None:
    """Fire-and-forget telemetry write on a daemon thread with its OWN connection.

    WHY(write-contention 2026-06-07): the synchronous log_llm_call(conn, ...) on the
    request's connection holds the HTTP worker on the single SQLite write lock (WAL
    serializes writers; busy_timeout makes them WAIT, not error). Under sustained
    pi-claim/ingest load that turned a read-mostly /memory/search into an 8-14s block —
    worker blocked on the lock, CPU ~0%. Backgrounding the telemetry insert keeps the
    hot path off the lock entirely: search returns as soon as its real work (embed +
    Chroma + rerank) is done; the log row lands whenever the writer queue drains.
    SQLite connections are not thread-safe, so the thread opens a fresh one."""
    import threading

    def _write() -> None:
        try:
            conn = init_memory_db(db_path or MEMORY_DB_PATH)
            try:
                log_llm_call(conn, call_type, model, prompt, response,
                             tool_calls, duration_ms, workspace_id)
            finally:
                conn.close()
        except Exception:
            pass  # best-effort telemetry — never surface from a background thread

    threading.Thread(target=_write, daemon=True).start()


# ---------------------------------------------------------------------------
# wiki_paper_cache helpers
# ---------------------------------------------------------------------------

def get_paper_cache(conn: DBAdapter, source_id: str, rule_name: str) -> list | None:
    row = conn.execute(
        "SELECT extracted FROM wiki_paper_cache WHERE source_id = ? AND rule_name = ?",
        (source_id, rule_name),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["extracted"])


def set_paper_cache(conn: DBAdapter, source_id: str, rule_name: str, extracted: list) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO wiki_paper_cache(source_id, rule_name, extracted, created_at)"
        " VALUES (?, ?, ?, ?)",
        (source_id, rule_name, json.dumps(extracted, ensure_ascii=False), _now_iso()),
    )
    conn.commit()
