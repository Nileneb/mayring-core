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


def _mk_goal(conn, text, ws="ws-1"):
    conn.execute(
        "INSERT INTO goals(text, workspace_id, created_at, updated_at) VALUES (?,?,?,?)",
        (text, ws, "2026-06-07", "2026-06-07"))
    return conn.execute(
        "SELECT id FROM goals WHERE workspace_id=? AND text=?", (ws, text)).fetchone()[0]


def test_v22_goal_scoped_uniqueness_fresh(tmp_path):
    import sqlite3
    conn = store.init_memory_db(tmp_path / "m.db")
    g1 = _mk_goal(conn, "Goal A")
    g2 = _mk_goal(conn, "Goal B")
    # same name under DIFFERENT goals is allowed (no cross-goal merge)
    conn.execute("INSERT INTO categories(name, goal_id) VALUES (?,?)", ("Motivation", g1))
    conn.execute("INSERT INTO categories(name, goal_id) VALUES (?,?)", ("Motivation", g2))
    assert conn.execute(
        "SELECT COUNT(*) FROM categories WHERE name='Motivation'").fetchone()[0] == 2
    # same name under the SAME goal is rejected
    try:
        conn.execute("INSERT INTO categories(name, goal_id) VALUES (?,?)", ("Motivation", g1))
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "UNIQUE(goal_id,name) must reject a duplicate within one goal"


def test_v22_base_partial_unique_index(tmp_path):
    """goal_id IS NULL (Legacy/Basis) bleibt global eindeutig — SQLite-NULL-
    Distinctness wird vom partiellen Unique-Index ux_categories_base_name gefangen."""
    import sqlite3
    conn = store.init_memory_db(tmp_path / "m.db")
    conn.execute("INSERT INTO categories(name) VALUES ('Basis')")  # goal_id NULL
    try:
        conn.execute("INSERT INTO categories(name) VALUES ('Basis')")  # second NULL dup
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "partial unique index must keep base (goal_id NULL) names unique"


def test_v21_to_v22_rebuild_preserves_data_and_fks(tmp_path):
    """Simuliere eine v21-DB (categories MIT goal_id aber UNIQUE(name)) + ein
    chunk_categories-FK. init muss auf UNIQUE(goal_id,name) rebuilden, Daten + FK
    erhalten (v14/v19-Lehre: kein Datenverlust, dependent FK bleibt gültig)."""
    import sqlite3
    db = tmp_path / "v21.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        """
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
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
            goal_id INTEGER REFERENCES goals(id),
            UNIQUE (name)
        );
        CREATE TABLE chunk_categories (
            chunk_id TEXT NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(id),
            codebook_version INTEGER NOT NULL DEFAULT 1,
            confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL DEFAULT 'deductive',
            span_start INTEGER, span_end INTEGER,
            PRIMARY KEY (chunk_id, category_id)
        );
        INSERT INTO categories(id, name) VALUES (7, 'Legacy-Cat');
        INSERT INTO chunk_categories(chunk_id, category_id) VALUES ('chunk-1', 7);
        PRAGMA user_version = 21;
        """
    )
    raw.commit()
    raw.close()

    conn = store.init_memory_db(db)

    cat_ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='categories'").fetchone()[0]
    assert "UNIQUE (goal_id, name)" in cat_ddl
    # data preserved (same id) + dependent FK row still joins
    assert conn.execute("SELECT name FROM categories WHERE id=7").fetchone()[0] == "Legacy-Cat"
    joined = conn.execute(
        "SELECT c.name FROM chunk_categories cc JOIN categories c ON c.id=cc.category_id "
        "WHERE cc.chunk_id='chunk-1'").fetchone()
    assert joined and joined[0] == "Legacy-Cat"


def test_upsert_canonical_goal_dedup(tmp_path):
    from mayring_core.memory.ingestion.mayring_process import upsert_canonical_goal
    conn = store.init_memory_db(tmp_path / "m.db")
    # no chroma → exact-text path only
    g1 = upsert_canonical_goal(conn, None, "Welche Faktoren fördern KI-Nutzung?", [], "ws-1")
    g2 = upsert_canonical_goal(conn, None, "Welche Faktoren fördern KI-Nutzung?", [], "ws-1")
    assert g1 == g2  # exact text reused, no duplicate goal
    assert conn.execute("SELECT occurrence_count FROM goals WHERE id=?", (g1,)).fetchone()[0] == 2
    g3 = upsert_canonical_goal(conn, None, "Ein ganz anderes Ziel", [], "ws-1")
    assert g3 != g1
    assert conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0] == 2
    assert upsert_canonical_goal(conn, None, "   ", [], "ws-1") is None


def test_upsert_canonical_goal_semantic_reuse(tmp_path):
    """Paraphrase wird per Embedding-Cosine auf denselben Goal gemappt (fake chroma)."""
    from mayring_core.memory.ingestion import mayring_process as mp
    conn = store.init_memory_db(tmp_path / "m.db")

    class FakeChroma:
        def __init__(self):
            self.store = []  # (goal_id, emb)
        def upsert(self, ids, embeddings, documents, metadatas):
            self.store.append((metadatas[0]["goal_id"], embeddings[0]))
        def query(self, query_embeddings, n_results, where):
            if not self.store:
                return {"distances": [[]], "metadatas": [[]]}
            gid, _ = self.store[-1]
            return {"distances": [[0.05]], "metadatas": [[{"goal_id": gid}]]}  # cosine 0.95

    fc = FakeChroma()
    g1 = mp.upsert_canonical_goal(conn, fc, "Goal X", [0.1, 0.2], "ws-1")
    g2 = mp.upsert_canonical_goal(conn, fc, "Goal X paraphrasiert", [0.1, 0.2], "ws-1")
    assert g1 == g2  # semantic reuse via cosine >= threshold
    assert conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0] == 1


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
