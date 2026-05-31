"""Tenancy Phase A — Task 3: visibility-axis migration v14→v15.

The visibility axis moves from 4 values (private|user|org|public, where
private == workspace-scoped and user == cross-device-me) to 3 values
(private == user_id-scoped, org, public). This migration re-sorts EXISTING
rows and tightens the DB CHECK. It runs at container boot (_migrate_schema)
BEFORE the new scope_filter ships, so there is no search blackout.

Mapping:
  - visibility='user'                          -> 'private' (user_id stays)
  - visibility='private' in org/team workspace -> 'org' + org_id=workspace_id
  - visibility='private' in personal workspace -> stamp user_id = owner
  - new CHECK: visibility IN ('private', 'org', 'public')
"""
import tempfile
from pathlib import Path

import pytest

from mayring_core.memory import store
from mayring_core.memory.store import init_memory_db


# Legacy v14 sources DDL (4-value CHECK incl. 'user'). init_memory_db already
# migrates a fresh DB to v15 (3 values), so to exercise the v14->v15 upgrade
# path we rebuild sources back to the legacy shape and seed legacy rows.
_SOURCES_DDL_V14 = """
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


def _seed(tmp: Path):
    conn = init_memory_db(tmp)
    # Reset sources to the legacy v14 (4-value) shape so 'user' inserts pass.
    conn.execute("DROP TABLE sources")
    conn.executescript(_SOURCES_DDL_V14)
    # workspaces.kind CHECK allows ('user', 'team', 'project', 'system'):
    # 'user' == a personal workspace, 'team' == an org/team workspace.
    now = "2026-05-31T00:00:00+00:00"
    conn.execute(
        "INSERT INTO workspaces (id, kind, display_name, created_at, updated_at) "
        "VALUES ('ws-personal', 'user', 'Personal', ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO workspaces (id, kind, display_name, created_at, updated_at) "
        "VALUES ('ws-org', 'team', 'Org', ?, ?)",
        (now, now),
    )
    # s-p: personal-workspace 'private' -> should get user_id stamped, stays private
    conn.execute(
        "INSERT INTO sources (source_id, captured_at, workspace_id, visibility) "
        "VALUES ('s-p', ?, 'ws-personal', 'private')",
        (now,),
    )
    # s-o: org-workspace 'private' -> should become 'org'
    conn.execute(
        "INSERT INTO sources (source_id, captured_at, workspace_id, visibility) "
        "VALUES ('s-o', ?, 'ws-org', 'private')",
        (now,),
    )
    # s-u: legacy 'user' -> should become 'private', user_id preserved
    conn.execute(
        "INSERT INTO sources (source_id, captured_at, workspace_id, visibility, user_id) "
        "VALUES ('s-u', ?, 'ws-personal', 'user', 'u-1')",
        (now,),
    )
    conn.commit()
    return conn


def test_migration_resorts_visibility():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = _seed(tmp)
        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})

        def row(sid):
            return conn.execute(
                "SELECT visibility, org_id, user_id FROM sources WHERE source_id = ?",
                (sid,),
            ).fetchone()

        vis_p, org_p, uid_p = row("s-p")
        assert vis_p == "private"
        assert uid_p == "u-1"

        vis_o, org_o, uid_o = row("s-o")
        assert vis_o == "org"

        vis_u, org_u, uid_u = row("s-u")
        assert vis_u == "private"
        assert uid_u == "u-1"


def test_migration_check_rejects_legacy_values():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = _seed(tmp)
        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})

        now = "2026-05-31T00:00:00+00:00"
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO sources (source_id, captured_at, workspace_id, visibility) "
                "VALUES ('bad', ?, 'ws-org', 'workspace')",
                (now,),
            )
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO sources (source_id, captured_at, workspace_id, visibility) "
                "VALUES ('bad2', ?, 'ws-org', 'user')",
                (now,),
            )


def _index_names(conn) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='sources'"
        ).fetchall()
    }


def test_full_boot_path_preserves_sources_indexes():
    """BLOCKER 1: init_memory_db on a fresh DB migrates internally to v15.

    The table rebuild must not strand the sources hot-path indexes — they are
    created in _init_schema BEFORE the migration runs and dropped by the
    rebuild, and the version short-circuit prevents _init_schema from
    re-creating them. This test fails without the in-rebuild CREATE INDEX.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = init_memory_db(tmp)  # internally migrates fresh DB to v15
        names = _index_names(conn)
        assert "idx_sources_workspace_id" in names
        assert "idx_sources_scope_key" in names


def test_migration_recreates_indexes_after_v14_upgrade():
    """BLOCKER 1: the v14->v15 upgrade path also re-creates the indexes."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = _seed(tmp)
        # _seed drops/rebuilds `sources` to the legacy shape, losing indexes.
        assert "idx_sources_workspace_id" not in _index_names(conn)
        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})
        names = _index_names(conn)
        assert "idx_sources_workspace_id" in names
        assert "idx_sources_scope_key" in names


def test_migration_is_idempotent():
    """A second migrate call no-ops and leaves no legacy table behind."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = _seed(tmp)
        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})
        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})  # 2nd run

        legacy = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sources_legacy_v15'"
        ).fetchone()
        assert legacy is None
        # rows still intact
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 3


def test_fk_integrity_after_migration():
    """BLOCKER 2: chunks.source_id FK must still resolve to `sources`.

    A bungled rebuild (FK rewritten to the dropped legacy table) would make
    insert_chunk fail with "no such table". Insert a chunk against a migrated
    source and assert it lands.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = _seed(tmp)
        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})

        now = "2026-05-31T00:00:00+00:00"
        # s-p exists in sources; chunk references it via FK (chunks.source_id
        # REFERENCES sources(source_id)). text/created_at are the real columns.
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, text_hash, created_at) "
            "VALUES ('c-1', 's-p', 'hello', 'h1', ?)",
            (now,),
        )
        conn.commit()
        got = conn.execute(
            "SELECT source_id FROM chunks WHERE chunk_id = 'c-1'"
        ).fetchone()
        assert got[0] == "s-p"


def test_user_chunk_in_team_ws_stays_private():
    """WICHTIG: an ex-'user' row (user_id set) in a team WS must NOT become 'org'.

    Step-1 (user->private) runs before step-2 (private->org). Without the
    user_id guard in step 2, the converted row would be promoted to 'org',
    leaking a personal chunk to the whole team.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "memory.db"
        conn = _seed(tmp)
        now = "2026-05-31T00:00:00+00:00"
        # seed a legacy 'user' row owned by u-9 living in the TEAM workspace.
        conn.execute(
            "INSERT INTO sources (source_id, captured_at, workspace_id, visibility, user_id) "
            "VALUES ('s-ut', ?, 'ws-org', 'user', 'u-9')",
            (now,),
        )
        conn.commit()

        store.migrate_visibility_axis(conn, personal_owner={"ws-personal": "u-1"})

        vis, org_id, uid = conn.execute(
            "SELECT visibility, org_id, user_id FROM sources WHERE source_id = 's-ut'"
        ).fetchone()
        assert vis == "private"   # NOT 'org'
        assert uid == "u-9"
        assert org_id is None
