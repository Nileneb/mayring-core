"""v29 FTS5-Keyword-Lane — Kandidaten-Cap ohne Recall- oder Isolationsverlust.

Vorher lud jede Suche den kompletten Scope (~8k Chunks samt Volltext) und
tokenisierte ihn für die symbolische Stufe. Jetzt liefert chunks_fts die besten
bm25-Treffer, die gegen scope_set geschnitten werden. Diese Tests halten die drei
Eigenschaften fest, die dabei brechen könnten: exakte Keyword-Treffer bleiben
auffindbar, fremde Workspaces werden nie Kandidat, reasons/score_* bleiben befüllt
(Bug-Historie: Query-Cache lieferte score_vector=0 / reasons=[]).
"""
import mayring_core.memory.retrieval as retrieval
from mayring_core.memory.retrieval import _fts_candidate_ids, invalidate_query_cache, search
from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import (
    deactivate_chunks_by_source, init_memory_db, insert_chunk, upsert_source,
)

_RARE = "zqxglorbfish"


def _seed(conn, source_id, text, *, workspace_id, user_id, summary=""):
    upsert_source(
        conn,
        Source(
            source_id=source_id,
            source_type="note",
            repo="",
            path=source_id,
            visibility="private",
            user_id=user_id,
        ),
        workspace_id=workspace_id,
    )
    chunk = Chunk(
        chunk_id=Chunk.make_id(source_id, 0, "block"),
        source_id=source_id,
        chunk_level="block",
        text=text,
        text_hash=Chunk.compute_text_hash(text),
        summary=summary,
        embedding_model="test-embed",
        workspace_id=workspace_id,
    )
    insert_chunk(conn, chunk, workspace_id=workspace_id)
    conn.commit()
    return chunk.chunk_id


def _seed_noise(conn, n, *, workspace_id, user_id):
    return [
        _seed(conn, f"src:noise-{i}", f"unrelated filler text number {i}",
              workspace_id=workspace_id, user_id=user_id)
        for i in range(n)
    ]


def _search(conn, query, **opts):
    invalidate_query_cache()
    return search(query, conn, None, "", opts={"workspace_id": "ws-a", "user_id": "u-me", **opts})


class TestFtsKeywordLane:

    def test_exact_keyword_hit_survives_the_candidate_cap(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        _seed_noise(conn, 40, workspace_id="ws-a", user_id="u-me")
        needle = _seed(conn, "src:needle", f"the {_RARE} identifier lives here",
                       workspace_id="ws-a", user_id="u-me")

        results = _search(conn, _RARE)

        assert needle in [r.chunk_id for r in results]

    def test_summary_only_hit_is_found(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        needle = _seed(conn, "src:summary", "body without the identifier",
                       summary=f"summary mentions {_RARE}",
                       workspace_id="ws-a", user_id="u-me")

        assert needle in [r.chunk_id for r in _search(conn, _RARE)]

    def test_foreign_workspace_chunk_never_becomes_candidate(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        _seed(conn, "src:mine", f"{_RARE} in my own note",
              workspace_id="ws-a", user_id="u-me")
        foreign = _seed(conn, "src:foreign", f"{_RARE} in someone else's note",
                        workspace_id="ws-b", user_id="u-other")

        # Die FTS-Lane selbst kennt keine Tenancy — der Schnitt mit scope_set muss isolieren.
        assert foreign in _fts_candidate_ids(conn, {_RARE})
        assert foreign not in [r.chunk_id for r in _search(conn, _RARE)]

    def test_reasons_and_stage_scores_stay_populated(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        _seed(conn, "src:needle", f"{_RARE} token overlap carrier",
              workspace_id="ws-a", user_id="u-me")

        top = _search(conn, _RARE)[0]

        assert top.reasons
        assert top.score_symbolic > 0.0

    def test_vector_hit_outside_the_keyword_lane_is_merged_but_stays_scoped(
        self, tmp_path, monkeypatch,
    ):
        conn = init_memory_db(tmp_path / "memory.db")
        semantic = _seed(conn, "src:semantic", "no shared tokens whatsoever",
                         workspace_id="ws-a", user_id="u-me")
        foreign = _seed(conn, "src:foreign", "another user's chunk",
                        workspace_id="ws-b", user_id="u-other")

        class _Collection:
            def count(self):
                return 2

            def query(self, **kwargs):
                return {"ids": [[semantic, foreign]], "distances": [[0.1, 0.1]]}

        monkeypatch.setattr(retrieval, "_embed_texts", lambda texts, url: [[0.1, 0.2]])
        monkeypatch.setattr(retrieval, "_HAS_EMBED", True)
        from mayring_core.memory.ingestion import mayring_process
        monkeypatch.setattr(mayring_process, "derive_query_category_ids",
                            lambda *a, **kw: set())
        invalidate_query_cache()

        results = search(_RARE, conn, _Collection(), "http://localhost:11434",
                         opts={"workspace_id": "ws-a", "user_id": "u-me"})

        ids = [r.chunk_id for r in results]
        assert semantic in ids          # Vektor-Treffer wird nachgeladen
        assert foreign not in ids       # bleibt am scope_set hängen


class TestMigrationV29:

    def _rewind_to_v28(self, conn):
        conn.executescript("""
            DROP TRIGGER IF EXISTS chunks_fts_ai;
            DROP TRIGGER IF EXISTS chunks_fts_ad;
            DROP TRIGGER IF EXISTS chunks_fts_au;
            DROP TABLE IF EXISTS chunks_fts;
        """)
        conn.execute("PRAGMA user_version = 28")
        conn.commit()

    def test_backfill_is_complete_on_an_existing_v28_db(self, tmp_path):
        db = tmp_path / "memory.db"
        conn = init_memory_db(db)
        for i in range(5):
            _seed(conn, f"src:pre-{i}", f"legacy chunk {i} with {_RARE}",
                  workspace_id="ws-a", user_id="u-me")
        self._rewind_to_v28(conn)
        conn.close()

        conn2 = init_memory_db(db)

        n_chunks = conn2.execute("SELECT count(*) FROM chunks").fetchone()[0]
        n_fts = conn2.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        assert n_chunks == 5
        assert n_fts == n_chunks
        assert len(_fts_candidate_ids(conn2, {_RARE})) == 5

    def test_triggers_keep_the_index_in_sync(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")

        cid = _seed(conn, "src:live", f"inserted with {_RARE}",
                    workspace_id="ws-a", user_id="u-me")
        assert _fts_candidate_ids(conn, {_RARE}) == [cid]

        conn.execute("UPDATE chunks SET text = 'rewritten without it' WHERE chunk_id = ?", (cid,))
        conn.commit()
        assert _fts_candidate_ids(conn, {_RARE}) == []
        assert _fts_candidate_ids(conn, {"rewritten"}) == [cid]

        conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (cid,))
        conn.commit()
        assert _fts_candidate_ids(conn, {"rewritten"}) == []

    def test_deactivate_still_reports_the_chunk_count(self, tmp_path):
        """changes() zählt nach dem Commit die Trigger-Writes — deshalb rowcount."""
        conn = init_memory_db(tmp_path / "memory.db")
        upsert_source(
            conn,
            Source(source_id="src:multi", source_type="note", repo="", path="src:multi",
                   visibility="private", user_id="u-me"),
            workspace_id="ws-a",
        )
        for i in range(3):
            insert_chunk(conn, Chunk(
                chunk_id=Chunk.make_id("src:multi", i, "block"),
                source_id="src:multi", chunk_level="block", ordinal=i,
                text=f"chunk {i}", embedding_model="test-embed", workspace_id="ws-a",
            ), workspace_id="ws-a")
        conn.commit()

        assert deactivate_chunks_by_source(conn, "src:multi") == 3
