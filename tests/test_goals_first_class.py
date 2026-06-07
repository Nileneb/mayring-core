"""v21 goal-first-class: goals table + categories.goal_id anchor.

Critical (v14/v19-Lehre): the migration MUST work on the existing-DB upgrade
path, not only fresh DBs — that's where past schema changes broke prod."""
import sqlite3

from mayring_core.memory import store


def test_fresh_db_has_goals_and_goal_id(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "goals" in tables
    cat_cols = {r[1] for r in conn.execute("PRAGMA table_info(categories)").fetchall()}
    assert "goal_id" in cat_cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 21


def test_init_is_idempotent(tmp_path):
    db = tmp_path / "m.db"
    store.init_memory_db(db)
    conn = store.init_memory_db(db)  # second run must not raise
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "goals" in tables


def test_legacy_db_upgrade_adds_goal_id_and_goals(tmp_path):
    """Simulate a pre-v21 DB: a categories table WITHOUT goal_id and NO goals
    table, user_version=20. init must add the column + table without data loss."""
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    # mirror the real pre-v21 categories schema MINUS goal_id
    raw.executescript(
        """
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            igio_axis TEXT,
            parent_id INTEGER REFERENCES categories(id),
            description TEXT NOT NULL DEFAULT '',
            examples TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL DEFAULT 'imported',
            evidence_count INTEGER NOT NULL DEFAULT 0,
            embedding_id TEXT NOT NULL DEFAULT '',
            risk_level TEXT NOT NULL DEFAULT '',
            languages TEXT NOT NULL DEFAULT '[]',
            patterns TEXT NOT NULL DEFAULT '[]',
            promoted_at TEXT,
            project_id TEXT DEFAULT NULL,
            UNIQUE (name)
        );
        INSERT INTO categories(name) VALUES ('legacy-cat');
        PRAGMA user_version = 20;
        """
    )
    raw.commit()
    raw.close()

    conn = store.init_memory_db(db)

    cat_cols = {r[1] for r in conn.execute("PRAGMA table_info(categories)").fetchall()}
    assert "goal_id" in cat_cols
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "goals" in tables
    # legacy row survives, goal_id defaults to NULL (no goal known)
    row = conn.execute(
        "SELECT goal_id FROM categories WHERE name='legacy-cat'").fetchone()
    assert row[0] is None
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 21


def test_goals_table_roundtrip(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    conn.execute(
        "INSERT INTO goals(text, workspace_id, created_at, updated_at) "
        "VALUES (?,?,?,?)",
        ("Welche Faktoren fördern KI-Nutzung?", "ws-1", "2026-06-07", "2026-06-07"),
    )
    gid = conn.execute("SELECT id FROM goals WHERE workspace_id='ws-1'").fetchone()[0]
    conn.execute("UPDATE categories SET goal_id=? WHERE name IS NOT NULL", (gid,)) \
        if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] else None
    # goal is reusable as an anchor target
    assert conn.execute("SELECT text FROM goals WHERE id=?", (gid,)).fetchone()[0] \
        == "Welche Faktoren fördern KI-Nutzung?"
