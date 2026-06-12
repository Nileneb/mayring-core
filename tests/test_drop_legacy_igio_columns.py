"""igio-removal 2026-06-12: the drop-migration must remove the 4 legacy IGIO
columns on an existing DB that still carries them. Critical (v14/v19-Lehre):
verify the upgrade path, not only fresh DBs."""
from mayring_core.memory import store


def test_drop_migration_removes_legacy_igio_columns(tmp_path):
    """Build a current DB, re-add the legacy igio columns + roll user_version back
    (simulating a pre-removal prod DB), then re-run init — the drop-migration must
    wipe them. Rolling user_version back is required because _init_schema
    early-returns on an up-to-date version (v14-Lehre: migrations gate on version)."""
    db = tmp_path / "igio.db"
    conn = store.init_memory_db(db)
    for stmt in (
        "ALTER TABLE chunks ADD COLUMN igio_axis TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE chunks ADD COLUMN igio_confidence REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE chunks ADD COLUMN igio_classified_at TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE categories ADD COLUMN igio_axis TEXT",
    ):
        conn.execute(stmt)
    conn.execute("PRAGMA user_version = 24")  # pre-igio-removal schema version
    conn.commit()
    # legacy columns are present before the re-init
    pre_chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "igio_axis" in pre_chunk_cols
    conn.close()

    # re-open → _migrate_schema runs again and drops the legacy columns
    conn = store.init_memory_db(db)

    chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    cat_cols = {r[1] for r in conn.execute("PRAGMA table_info(categories)").fetchall()}
    for col in ("igio_axis", "igio_confidence", "igio_classified_at"):
        assert col not in chunk_cols, f"chunks.{col} must be dropped"
    assert "igio_axis" not in cat_cols, "categories.igio_axis must be dropped"
    conn.close()


def test_drop_migration_is_idempotent_on_fresh_db(tmp_path):
    """Fresh DB has no igio columns → drop-migration must not raise, second
    init must not raise either."""
    db = tmp_path / "fresh.db"
    store.init_memory_db(db).close()
    conn = store.init_memory_db(db)  # second run hits the no-op DROP branch
    chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "igio_axis" not in chunk_cols
    conn.close()
