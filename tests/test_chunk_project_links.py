"""Schema v17→v18: chunk_project_links table (C3 project-scoped memory).

Mirrors test_project_groups_schema.py: fresh-DB create + existing-DB upgrade
path (no raise on a v17 DB that lacks the table).
"""
import tempfile
from pathlib import Path

from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import (
    init_memory_db,
    insert_chunk,
    upsert_source,
    link_chunk_to_project,
    project_linked_chunk_ids,
)


def _cols(conn, table):
    return conn.get_columns(table)


def _seed_chunk(conn, source_id, *, workspace_id):
    upsert_source(
        conn,
        Source(source_id=source_id, source_type="note", repo="", path=source_id),
        workspace_id=workspace_id,
    )
    chunk_id = Chunk.make_id(source_id, 0, "block")
    insert_chunk(
        conn,
        Chunk(chunk_id=chunk_id, source_id=source_id, chunk_level="block",
              text=f"text {source_id}", workspace_id=workspace_id),
        workspace_id=workspace_id,
    )
    return chunk_id


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


def test_link_and_bulk_lookup_idempotent_and_ws_isolated():
    with tempfile.TemporaryDirectory() as d:
        conn = init_memory_db(Path(d) / "m.db")
        ws = "ws-a"
        c1 = _seed_chunk(conn, "src:1", workspace_id=ws)
        c2 = _seed_chunk(conn, "src:2", workspace_id=ws)

        link_chunk_to_project(conn, c1, "projX", workspace_id=ws,
                              origin_ref="repo/sub", source="ingest")
        link_chunk_to_project(conn, c2, "projX", workspace_id=ws)
        # Idempotent second call must not raise (PK conflict → IGNORE).
        link_chunk_to_project(conn, c1, "projX", workspace_id=ws)

        linked = project_linked_chunk_ids(conn, [c1, c2], "projX", ws)
        assert linked == {c1, c2}

        # Workspace isolation: a different ws sees nothing for projX.
        assert project_linked_chunk_ids(conn, [c1, c2], "projX", "ws-other") == set()

        # Empty inputs short-circuit.
        assert project_linked_chunk_ids(conn, [], "projX", ws) == set()
        assert project_linked_chunk_ids(conn, [c1], "", ws) == set()
