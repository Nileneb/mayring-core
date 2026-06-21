"""reference-doc-layer + repo-scoping-hardfilter — _scope_filter behaviour.

Reference docs (source_class='reference') are DEFAULT-EXCLUDED from retrieval so
external corpora (Unity docs) never drown own code. They surface only via
include_reference, reference_only (/reference/search) or a chunk_project_links
eligibility for the active project ("Unity-Docs nur bei Battlefield"). Repo
scoping hard-filters the candidate pool to one repo.

Specs: 2026-06-21-reference-doc-layer, 2026-06-21-retrieval-repo-scoping-hardfilter.
"""
from mayring_core.memory.retrieval import _scope_filter
from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import (
    init_memory_db, insert_chunk, link_chunk_to_project, upsert_source,
)


def _seed(conn, source_id, *, workspace_id, user_id="u-me", repo="",
          source_class="code"):
    src = Source(
        source_id=source_id,
        source_type="note",
        repo=repo,
        path=source_id,
        visibility="private",
        user_id=user_id,
        source_class=source_class,
    )
    upsert_source(conn, src, workspace_id=workspace_id)
    chunk_id = Chunk.make_id(source_id, 0, "block")
    insert_chunk(
        conn,
        Chunk(chunk_id=chunk_id, source_id=source_id, chunk_level="block",
              text=f"text for {source_id}", workspace_id=workspace_id),
        workspace_id=workspace_id,
    )
    return chunk_id


def _seed_corpus(conn):
    return {
        "code": _seed(conn, "src:code", workspace_id="ws-a", repo="owner/app"),
        "code_other_repo": _seed(
            conn, "src:code-other", workspace_id="ws-a", repo="owner/mayring"),
        "ref": _seed(
            conn, "unity-docs:webgl", workspace_id="ws-a",
            source_class="reference"),
    }


class TestReferenceDocLayer:

    def test_default_excludes_reference(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_corpus(conn)
        visible = set(_scope_filter(conn, workspace_id="ws-a", user_id="u-me"))
        assert ids["code"] in visible
        assert ids["ref"] not in visible  # reference drowned no more

    def test_include_reference_surfaces_it(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_corpus(conn)
        visible = set(_scope_filter(
            conn, workspace_id="ws-a", user_id="u-me", include_reference=True))
        assert ids["ref"] in visible
        assert ids["code"] in visible

    def test_reference_only_returns_just_reference(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_corpus(conn)
        visible = set(_scope_filter(
            conn, workspace_id="ws-a", user_id="u-me", reference_only=True))
        assert visible == {ids["ref"]}

    def test_reference_eligible_when_linked_to_active_project(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_corpus(conn)
        link_chunk_to_project(
            conn, ids["ref"], "proj-battlefield", workspace_id="ws-a")
        # Active project = the one the reference corpus is linked to → eligible.
        visible = set(_scope_filter(
            conn, workspace_id="ws-a", user_id="u-me",
            project_id="proj-battlefield"))
        assert ids["ref"] in visible
        # A different active project must NOT see the reference corpus.
        visible_other = set(_scope_filter(
            conn, workspace_id="ws-a", user_id="u-me", project_id="proj-web"))
        assert ids["ref"] not in visible_other


class TestRepoScoping:

    def test_repo_scope_excludes_foreign_repos(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_corpus(conn)
        visible = set(_scope_filter(
            conn, workspace_id="ws-a", user_id="u-me", repo="owner/app"))
        assert ids["code"] in visible
        assert ids["code_other_repo"] not in visible
