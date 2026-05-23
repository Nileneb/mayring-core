"""Memory Storage Layer — SQLite + ChromaDB singletons.

Tables:
    sources, chunks, chunk_feedback, ingestion_log

ChromaDB:
    get_chroma_collection(name, path) — process-scoped singleton
"""

from __future__ import annotations

import json
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
    """
    try:
        import chromadb
    except ImportError:
        return None
    chroma_path = str(path or CACHE_DIR / "memory_chroma")
    key = f"{chroma_path}::{name}"
    if key not in _chroma_collections:
        if chroma_path not in _chroma_clients:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            _chroma_clients[chroma_path] = chromadb.PersistentClient(path=chroma_path)
        _chroma_collections[key] = _chroma_clients[chroma_path].get_or_create_collection(name)
    return _chroma_collections[key]
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
#       ODER papers/research). Goals↔Project (M:M, IGIO) = Backend-Designfrage offen.
#   v9 (#workspace-uuid-sot v2.0 Phase 1): codebook-in-DB (codebooks,
#       codebook_categories, codebook_proposals, chunk_categories) — Codebook aus
#       YAML → SQLite, Category-Embeddings → Chroma-Collection "codebook_categories".
CURRENT_SCHEMA_VERSION = 11


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
        ],
        "chunks": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
            ("project_id", "TEXT DEFAULT NULL"),
            ("category_source", "TEXT NOT NULL DEFAULT ''"),
            ("category_confidence", "REAL NOT NULL DEFAULT 0.0"),
            ("igio_axis", "TEXT NOT NULL DEFAULT ''"),
            ("igio_confidence", "REAL NOT NULL DEFAULT 0.0"),
            ("igio_classified_at", "TEXT NOT NULL DEFAULT ''"),
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
        ],
        # #workspace-uuid-sot v8 (User-Diagramm): Projekt-Metadaten.
        "projects": [
            ("owner_id", "TEXT"),
            ("source_type", "TEXT"),
            ("source_ref", "TEXT"),
        ],
        # v2 Phase 3.2 (project-scoped codebook): NULL = geteilte Profil-Basis
        # (imported), gesetzt = projekt-spezifisch induziert. Der aktive
        # Kategorien-Satz einer Session = Basis ∪ active_project.categories.
        "codebook_categories": [
            ("project_id", "TEXT DEFAULT NULL"),
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
                scope_key       TEXT DEFAULT NULL
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


def _init_schema(conn: DBAdapter) -> None:
    """Ensure all tables and indexes exist, with schema versioning to skip DDL when current.

    Uses PRAGMA user_version to track schema version and avoid lock contention on
    repeated CLI invocations.
    """
    # Read current schema version from PRAGMA user_version (default 0 for new DBs)
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]

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
            scope_key       TEXT DEFAULT NULL
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
            embedding_model     TEXT NOT NULL DEFAULT 'nomic-embed-text',
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
        -- Project Router (Slice 1): match by cwd-git-remote → source_ref.
        CREATE INDEX IF NOT EXISTS idx_projects_source
            ON projects(workspace_id, source_type, source_ref);

        -- #workspace-uuid-sot v2.0 Phase 1: Codebook aus YAML → DB (DB = SoT).
        -- Category-Embeddings leben in der Chroma-Collection "codebook_categories"
        -- (embedding_id = Chroma-Doc-ID), NICHT pgvector. Single-Workspace → kein
        -- workspace_id auf den Codebook-Tabellen.
        CREATE TABLE IF NOT EXISTS codebooks (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            slug                    TEXT UNIQUE NOT NULL,
            description             TEXT NOT NULL DEFAULT '',
            version                 INTEGER NOT NULL DEFAULT 1,
            auto_promote_threshold  INTEGER NOT NULL DEFAULT 3,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS codebook_categories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            codebook_id     INTEGER NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
            name            TEXT NOT NULL,
            igio_axis       TEXT CHECK (igio_axis IN ('I','G','V','O') OR igio_axis IS NULL),
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
            UNIQUE (codebook_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_codebook_cat_codebook
            ON codebook_categories(codebook_id, status);
        CREATE TABLE IF NOT EXISTS codebook_proposals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id     INTEGER NOT NULL REFERENCES codebook_categories(id) ON DELETE CASCADE,
            pi_job_id       TEXT NOT NULL DEFAULT '',
            chunk_id        TEXT,
            paraphrase      TEXT NOT NULL DEFAULT '',
            parent_hint_id  INTEGER REFERENCES codebook_categories(id),
            proposed_at     TEXT NOT NULL,
            reviewed_by     TEXT,
            decision        TEXT CHECK (decision IN ('promote','merge','reject') OR decision IS NULL)
        );
        CREATE TABLE IF NOT EXISTS chunk_categories (
            chunk_id          TEXT NOT NULL,
            category_id       INTEGER NOT NULL REFERENCES codebook_categories(id),
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
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            completed_at   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_workspace_status ON tasks(workspace_id, status);
        CREATE INDEX IF NOT EXISTS idx_tasks_workspace_due ON tasks(workspace_id, due_date);

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
    """)

    # Migration: add missing columns to existing DBs
    _migrate_schema(conn)

    # Migration v5→v6: devices PK device_id → (device_id, workspace_id) (#274 review)
    _migrate_devices_composite_pk(conn)

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

    # Update schema version in PRAGMA user_version to mark migration as complete
    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")

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
    conn.execute(
        """
        INSERT INTO sources
            (source_id, source_type, repo, path, branch, "commit", content_hash,
             captured_at, workspace_id, visibility, org_id, user_id, scope_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            scope_key    = excluded.scope_key
        """,
        (
            source.source_id, source.source_type, source.repo, source.path,
            source.branch, source.commit, source.content_hash, source.captured_at,
            workspace_id, eff_visibility, eff_org_id, eff_user_id, eff_scope_key,
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


def update_chunk_igio_axis(
    conn: Any,
    chunk_id: str,
    axis: str,
    confidence: float = 0.85,
) -> bool:
    """Tag a single chunk with an IGIO axis without touching other fields.

    WHY(igio-pipeline-2026-05-15): stop_hook sends a fast-hint axis from the
    user prompt (regex, no LLM). Writing it here skips the async IGIO cron
    and makes the axis immediately visible in /memory/goals and retrieval
    scoring. Only overwrites when the existing axis is empty so the LLM-cron
    verdict (higher confidence) is never clobbered.
    """
    import datetime as _dt
    row = conn.execute(
        "SELECT igio_axis FROM chunks WHERE chunk_id = ? AND is_active = 1",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return False
    existing = (row[0] or "").strip()
    if existing:  # already classified by LLM-cron → don't overwrite
        return False
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    conn.execute(
        "UPDATE chunks SET igio_axis = ?, igio_confidence = ?, igio_classified_at = ? "
        "WHERE chunk_id = ?",
        (axis, confidence, now, chunk_id),
    )
    _maybe_commit(conn)
    return True


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
