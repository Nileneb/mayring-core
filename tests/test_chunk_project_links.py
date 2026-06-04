"""Schema v17→v18: chunk_project_links table (C3 project-scoped memory).

Mirrors test_project_groups_schema.py: fresh-DB create + existing-DB upgrade
path (no raise on a v17 DB that lacks the table).
"""
import tempfile
from pathlib import Path

from mayring_core.memory.store import init_memory_db


def _cols(conn, table):
    return conn.get_columns(table)


def test_fresh_db_has_chunk_project_links():
    with tempfile.TemporaryDirectory() as d:
        conn = init_memory_db(Path(d) / "m.db")
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "chunk_project_links" in tables
        cols = set(_cols(conn, "chunk_project_links"))
        assert {"chunk_id", "project_id", "origin_ref",
                "source", "workspace_id", "created_at"} <= cols
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 18


def test_existing_v17_db_upgrades_without_raise():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.db"
        conn = init_memory_db(path)
        # Simulate a v17 DB: drop the new table and reset user_version so
        # _init_schema re-runs the (idempotent) DDL block.
        conn.execute("DROP TABLE IF EXISTS chunk_project_links")
        # WHY: PRAGMA user_version cannot be parameter-bound; literal is safe.
        conn.execute("PRAGMA user_version = 17")
        conn.commit()

        conn2 = init_memory_db(path)  # must NOT raise
        tables = {r[0] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "chunk_project_links" in tables
        assert conn2.execute("PRAGMA user_version").fetchone()[0] == 18
