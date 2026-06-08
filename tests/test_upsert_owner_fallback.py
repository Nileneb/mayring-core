import sqlite3

from mayring_core.memory.schema import Source
from mayring_core.memory.store import init_memory_db, upsert_source


def _owner(db, source_id):
    return sqlite3.connect(db).execute(
        "SELECT user_id, visibility FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()


def test_private_source_without_owner_falls_back_to_workspace_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_PERSONAL_OWNERS", '{"ws1": "1"}')
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(db)
    # service-token style write: private, no user_id (the orphan-producing case)
    src = Source(source_id="repo:x:a.py", source_type="repo_file", repo="x", path="a.py", visibility="private")
    upsert_source(conn, src, workspace_id="ws1")
    conn.commit()
    assert _owner(db, "repo:x:a.py") == ("1", "private")


def test_explicit_user_id_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_PERSONAL_OWNERS", '{"ws1": "1"}')
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(db)
    src = Source(source_id="repo:x:b.py", source_type="repo_file", repo="x", path="b.py", visibility="private")
    upsert_source(conn, src, workspace_id="ws1", user_id="42")
    conn.commit()
    assert _owner(db, "repo:x:b.py")[0] == "42"  # caller's owner wins


def test_public_source_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_PERSONAL_OWNERS", '{"ws1": "1"}')
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(db)
    src = Source(source_id="pub:y", source_type="note", repo="", path="y", visibility="public")
    upsert_source(conn, src, workspace_id="ws1")
    conn.commit()
    # public data needs no owner stamp (the where-clause public branch ignores user_id)
    assert (_owner(db, "pub:y")[0] or "") == ""


def test_no_owner_map_is_safe_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("MAYRING_PERSONAL_OWNERS", raising=False)
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(db)
    src = Source(source_id="repo:x:c.py", source_type="repo_file", repo="x", path="c.py", visibility="private")
    upsert_source(conn, src, workspace_id="ws1")
    conn.commit()
    assert (_owner(db, "repo:x:c.py")[0] or "") == ""  # no map → unchanged, no crash
