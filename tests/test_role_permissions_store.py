from mayring_core.memory import store
from mayring_core.memory.store import get_role_permissions, set_role_permission, CURRENT_SCHEMA_VERSION


def test_table_exists_and_version(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    assert CURRENT_SCHEMA_VERSION >= 16
    names = {(r[0] if not hasattr(r, "keys") else r["name"]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "workspace_role_permissions" in names


def test_set_and_get_roundtrip(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    assert get_role_permissions(conn, "ws-1") == {}
    set_role_permission(conn, "ws-1", "editor", "share_public", False)
    assert get_role_permissions(conn, "ws-1")[("editor", "share_public")] is False
    set_role_permission(conn, "ws-1", "editor", "share_public", True)
    assert get_role_permissions(conn, "ws-1")[("editor", "share_public")] is True
