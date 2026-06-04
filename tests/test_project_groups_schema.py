"""Schema v16→v17: project_groups table + projects.group_id (C1)."""
import tempfile
from pathlib import Path

from mayring_core.memory import store
from mayring_core.memory.store import init_memory_db, PROJECT_GROUP_PALETTE


def _cols(conn, table):
    return conn.get_columns(table)


def test_fresh_db_has_project_groups_and_group_id():
    with tempfile.TemporaryDirectory() as d:
        db = init_memory_db(Path(d) / "m.db")
        conn = db
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "project_groups" in tables
        assert "group_id" in _cols(conn, "projects")


def test_existing_db_without_group_id_upgrades_without_raise():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.db"
        db = init_memory_db(path)
        # Simulate a v16 DB: drop project_groups and strip group_id from projects,
        # then reset PRAGMA user_version so _init_schema re-runs migration.
        db.execute("DROP TABLE IF EXISTS project_groups")
        db.execute("ALTER TABLE projects RENAME TO projects_old")
        db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, "
                   "name TEXT NOT NULL DEFAULT '', owner_id TEXT, source_type TEXT, "
                   "source_ref TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        db.execute("INSERT INTO projects(id,workspace_id,name,created_at,updated_at) "
                   "SELECT id,workspace_id,name,created_at,updated_at FROM projects_old")
        db.execute("DROP TABLE projects_old")
        # WHY: PRAGMA user_version cannot be set via parameter binding; literal is safe here.
        db.execute("PRAGMA user_version = 16")
        db.commit()

        db2 = init_memory_db(path)
        tables = {r[0] for r in db2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "project_groups" in tables
        assert "group_id" in _cols(db2, "projects")


def test_palette_is_distinct_hex():
    assert len(PROJECT_GROUP_PALETTE) >= 8
    assert all(c.startswith("#") and len(c) == 7 for c in PROJECT_GROUP_PALETTE)
    assert len(set(PROJECT_GROUP_PALETTE)) == len(PROJECT_GROUP_PALETTE)
