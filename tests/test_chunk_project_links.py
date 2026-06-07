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
        assert conn.execute("PRAGMA user_version").fetchone()[0] >= 20


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
        assert conn2.execute("PRAGMA user_version").fetchone()[0] >= 20


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


def test_rerank_project_boost_is_additive_not_a_wall():
    """A linked candidate gets +_PROJECT_MATCH_BOOST; an unlinked one stays in
    the result (global/unlinked knowledge is NEVER hidden) with project_match 0."""
    from mayring_core.memory.retrieval import _rerank, _PROJECT_MATCH_BOOST
    with tempfile.TemporaryDirectory() as d:
        conn = init_memory_db(Path(d) / "m.db")
        ws = "ws-a"
        linked_id = _seed_chunk(conn, "src:linked", workspace_id=ws)
        global_id = _seed_chunk(conn, "src:global", workspace_id=ws)
        link_chunk_to_project(conn, linked_id, "projX", workspace_id=ws)

        from mayring_core.memory.store import get_chunk
        candidates = [get_chunk(conn, linked_id), get_chunk(conn, global_id)]
        # Identical base signals so the ONLY differentiator is the project boost.
        vec = {linked_id: 0.4, global_id: 0.4}
        sym = {linked_id: 0.4, global_id: 0.4}

        records = _rerank(
            candidates, vec, sym, top_k=10, conn=conn,
            project_id="projX", workspace_id=ws,
        )
        by_id = {r.chunk_id: r for r in records}
        # Both present — boost is additive, not a filter wall.
        assert linked_id in by_id
        assert global_id in by_id

        assert by_id[linked_id].score_project_match > 0.0
        assert by_id[global_id].score_project_match == 0.0
        assert by_id[linked_id].score_final > by_id[global_id].score_final
        # The delta is exactly the deterministic boost (same base signals).
        delta = by_id[linked_id].score_final - by_id[global_id].score_final
        assert abs(delta - _PROJECT_MATCH_BOOST) < 1e-9


def test_rerank_no_project_id_no_boost():
    from mayring_core.memory.retrieval import _rerank
    with tempfile.TemporaryDirectory() as d:
        conn = init_memory_db(Path(d) / "m.db")
        ws = "ws-a"
        cid = _seed_chunk(conn, "src:1", workspace_id=ws)
        link_chunk_to_project(conn, cid, "projX", workspace_id=ws)
        from mayring_core.memory.store import get_chunk
        records = _rerank([get_chunk(conn, cid)], {cid: 0.4}, {cid: 0.4},
                          top_k=10, conn=conn)
        assert records[0].score_project_match == 0.0
