"""reference-doc-layer — v24→v25 source_class migration.

On an existing DB (no source_class column), the migration must:
  - add sources.source_class with DEFAULT 'code' (every legacy row → 'code'),
  - flip known reference prefixes (unity-docs:) to 'reference'.
Runs at container boot (_migrate_schema) before the default-exclude ships, so
there is no retrieval blackout. Spec 2026-06-21-reference-doc-layer.
"""
from pathlib import Path

from mayring_core.memory.store import init_memory_db

# Legacy v24 sources shape — no source_class column yet.
_SOURCES_DDL_V24 = """
    CREATE TABLE sources (
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
        workspace_id    TEXT NOT NULL DEFAULT 'default',
        project_id      TEXT DEFAULT NULL
    );
"""


def test_v25_adds_source_class_and_flips_reference(tmp_path: Path):
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)
    # Reset sources to the legacy v24 shape (no source_class) and rewind version.
    conn.execute("DROP TABLE sources")
    conn.executescript(_SOURCES_DDL_V24)
    now = "2026-06-21T00:00:00+00:00"
    conn.execute(
        "INSERT INTO sources (source_id, captured_at) VALUES ('repo:owner/app:a.py', ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO sources (source_id, captured_at) VALUES ('unity-docs:webgl', ?)",
        (now,),
    )
    conn.execute("PRAGMA user_version = 24")
    conn.commit()
    conn.close()

    # Re-open → _migrate_schema runs the v25 migration.
    conn2 = init_memory_db(db)
    cols = conn2.get_columns("sources")
    assert "source_class" in cols

    rows = dict(conn2.execute(
        "SELECT source_id, source_class FROM sources").fetchall())
    assert rows["repo:owner/app:a.py"] == "code"        # legacy default
    assert rows["unity-docs:webgl"] == "reference"        # flipped by prefix
