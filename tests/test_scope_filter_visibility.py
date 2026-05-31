"""Tenancy Phase A (T4): _scope_filter visibility disjunction.

private is now user_id-scoped (same human across all devices/workspaces), NOT
workspace-scoped. A chunk is visible when ANY of:
  - visibility='public'
  - visibility='org'     AND s.org_id  IN caller_orgs
  - visibility='private' AND s.user_id = caller_user

Hard isolation invariant: another user's private chunks must NEVER match, and
org chunks must NEVER match a non-member.
"""
from mayring_core.memory.retrieval import _scope_filter
from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import init_memory_db, insert_chunk, upsert_source


def _seed_source(conn, source_id, visibility, *, workspace_id, org_id=None, user_id=None):
    src = Source(
        source_id=source_id,
        source_type="note",
        repo="",
        path=source_id,
        visibility=visibility,
        org_id=org_id,
        user_id=user_id,
    )
    upsert_source(conn, src, workspace_id=workspace_id)
    chunk_id = Chunk.make_id(source_id, 0, "block")
    chunk = Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        chunk_level="block",
        text=f"text for {source_id}",
        workspace_id=workspace_id,
    )
    insert_chunk(conn, chunk, workspace_id=workspace_id)
    return chunk_id


def _seed_four(conn):
    return {
        # NOTE: own private lives in ws-other (NOT the caller's ws-a) on purpose:
        # under the 3-value model 'private' is user_id-scoped, so the same human
        # must see it from a different workspace (claude.ai web vs CLI). The old
        # workspace-scoped clause would WRONGLY hide it.
        "own_private": _seed_source(
            conn, "src:own-private", "private", workspace_id="ws-other", user_id="u-me"
        ),
        "other_private": _seed_source(
            conn, "src:other-private", "private", workspace_id="ws-b", user_id="u-other"
        ),
        "org": _seed_source(
            conn, "src:org", "org", workspace_id="ws-b", org_id="org-x"
        ),
        "public": _seed_source(
            conn, "src:public", "public", workspace_id="ws-b"
        ),
    }


class TestScopeFilterVisibility:

    def test_member_sees_own_private_org_and_public_never_foreign_private(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_four(conn)

        visible = set(
            _scope_filter(
                conn,
                workspace_id="ws-a",
                user_id="u-me",
                org_ids=("org-x",),
            )
        )

        assert ids["own_private"] in visible
        assert ids["org"] in visible
        assert ids["public"] in visible
        # Hard isolation: another human's private chunk must NEVER leak.
        assert ids["other_private"] not in visible

    def test_non_member_does_not_see_org_but_sees_public(self, tmp_path):
        conn = init_memory_db(tmp_path / "memory.db")
        ids = _seed_four(conn)

        visible = set(
            _scope_filter(
                conn,
                workspace_id="ws-a",
                user_id="u-me",
                org_ids=(),
            )
        )

        assert ids["public"] in visible
        assert ids["own_private"] in visible
        # Not a member of org-x → org chunk must NOT be visible.
        assert ids["org"] not in visible
        assert ids["other_private"] not in visible
