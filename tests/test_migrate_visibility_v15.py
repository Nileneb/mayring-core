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
