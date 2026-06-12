"""v19 Migration test: _migrate_codebook_collapse (v14-Lehre, PFLICHT).

Verifikation:
 - Upgrade-Pfad (v18-DB mit Daten) → categories-Tabelle, Daten erhalten,
   FKs korrekt umgebogen, codebooks + codebook_categories weg.
 - Idempotenz: zweiter _init_schema-Lauf wirft nicht.
 - Frische DB: categories direkt, kein codebooks, kein Fehler.
"""
import sqlite3
from pathlib import Path

import pytest

from mayring_core.memory.store import init_memory_db, _migrate_codebook_collapse


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _old_schema_db(path: Path) -> sqlite3.Connection:
    """Baut eine DB mit dem ALTEN v18-Schema (codebooks + codebook_categories + dependents)
    und befüllt sie mit Testdaten. user_version=18 (kein auto-migrate durch store.py)."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        PRAGMA user_version = 18;

        CREATE TABLE codebooks (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            slug                    TEXT UNIQUE NOT NULL,
            description             TEXT NOT NULL DEFAULT '',
            version                 INTEGER NOT NULL DEFAULT 1,
            auto_promote_threshold  INTEGER NOT NULL DEFAULT 3,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL
        );

        CREATE TABLE codebook_categories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            codebook_id     INTEGER NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
            name            TEXT NOT NULL,
            parent_id       INTEGER REFERENCES codebook_categories(id),
            description     TEXT NOT NULL DEFAULT '',
            examples        TEXT NOT NULL DEFAULT '[]',
            status          TEXT NOT NULL DEFAULT 'active',
            source          TEXT NOT NULL DEFAULT 'imported',
            evidence_count  INTEGER NOT NULL DEFAULT 0,
            embedding_id    TEXT NOT NULL DEFAULT '',
            risk_level      TEXT NOT NULL DEFAULT '',
            languages       TEXT NOT NULL DEFAULT '[]',
            patterns        TEXT NOT NULL DEFAULT '[]',
            promoted_at     TEXT,
            project_id      TEXT DEFAULT NULL,
            UNIQUE (codebook_id, name)
        );

        CREATE TABLE chunk_categories (
            chunk_id          TEXT NOT NULL,
            category_id       INTEGER NOT NULL REFERENCES codebook_categories(id),
            codebook_version  INTEGER NOT NULL DEFAULT 1,
            confidence        REAL NOT NULL DEFAULT 0.0,
            source            TEXT NOT NULL DEFAULT 'deductive',
            span_start        INTEGER,
            span_end          INTEGER,
            PRIMARY KEY (chunk_id, category_id)
        );

        CREATE TABLE codebook_proposals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id     INTEGER NOT NULL REFERENCES codebook_categories(id) ON DELETE CASCADE,
            pi_job_id       TEXT NOT NULL DEFAULT '',
            chunk_id        TEXT,
            paraphrase      TEXT NOT NULL DEFAULT '',
            parent_hint_id  INTEGER REFERENCES codebook_categories(id),
            proposed_at     TEXT NOT NULL,
            reviewed_by     TEXT,
            decision        TEXT
        );

        INSERT INTO codebooks(slug, description, auto_promote_threshold, created_at, updated_at)
        VALUES ('universal', 'Universal codebook', 3, '2026-01-01', '2026-01-01');

        INSERT INTO codebook_categories(id, codebook_id, name, status, evidence_count, embedding_id)
        VALUES (1, 1, 'auth', 'active', 5, 'cb:1'),
               (2, 1, 'api_design', 'active', 3, 'cb:2'),
               (3, 1, 'child_cat', 'proposed', 1, '');

        UPDATE codebook_categories SET parent_id = 1 WHERE id = 3;

        INSERT INTO chunk_categories(chunk_id, category_id, codebook_version, confidence, source)
        VALUES ('chk_a', 1, 1, 0.85, 'deductive'),
               ('chk_b', 2, 1, 0.72, 'deductive');

        INSERT INTO codebook_proposals(category_id, pi_job_id, chunk_id, paraphrase,
                                       parent_hint_id, proposed_at)
        VALUES (3, 'pi-1', 'chk_c', 'authentication flow', 1, '2026-01-02');
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMigrateCodebookCollapse:
    def test_upgrade_path_tables(self, tmp_path):
        """Nach Migration: categories vorhanden, codebooks + codebook_categories weg."""
        old = _old_schema_db(tmp_path / "old.db")
        old.close()

        conn = init_memory_db(tmp_path / "old.db")

        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "categories" in tables, "categories muss nach Migration existieren"
        assert "codebooks" not in tables, "codebooks muss nach Migration weg sein"
        assert "codebook_categories" not in tables, "codebook_categories muss weg sein"
        conn.close()

    def test_upgrade_path_data_preserved(self, tmp_path):
        """Daten (ids, names, evidence_count) bleiben nach Migration erhalten."""
        old = _old_schema_db(tmp_path / "data.db")
        old.close()

        conn = init_memory_db(tmp_path / "data.db")

        rows = conn.execute(
            "SELECT id, name, evidence_count FROM categories ORDER BY id"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][0] == 1 and rows[0][1] == "auth" and rows[0][2] == 5
        assert rows[1][0] == 2 and rows[1][1] == "api_design" and rows[1][2] == 3
        assert rows[2][0] == 3 and rows[2][1] == "child_cat"
        conn.close()

    def test_upgrade_path_parent_id_preserved(self, tmp_path):
        """self-FK parent_id bleibt nach Migration korrekt."""
        old = _old_schema_db(tmp_path / "parent.db")
        old.close()

        conn = init_memory_db(tmp_path / "parent.db")
        row = conn.execute(
            "SELECT parent_id FROM categories WHERE id = 3"
        ).fetchone()
        assert row is not None and row[0] == 1
        conn.close()

    def test_upgrade_path_chunk_categories_fk(self, tmp_path):
        """chunk_categories.category_id FK zeigt nach Migration auf categories."""
        old = _old_schema_db(tmp_path / "fk.db")
        old.close()

        conn = init_memory_db(tmp_path / "fk.db")

        # Vorhandene Links müssen noch da sein
        links = conn.execute(
            "SELECT chunk_id, category_id FROM chunk_categories ORDER BY chunk_id"
        ).fetchall()
        assert len(links) == 2
        chk_ids = {r[0] for r in links}
        assert "chk_a" in chk_ids and "chk_b" in chk_ids

        # FK aktiv: INSERT mit nicht-existenter category_id muss fehlschlagen
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chunk_categories(chunk_id, category_id, codebook_version, "
                "confidence, source) VALUES ('chk_x', 9999, 1, 0.5, 'deductive')"
            )
            conn.commit()

        # FK aktiv: INSERT mit existenter category_id muss klappen
        conn.execute(
            "INSERT INTO chunk_categories(chunk_id, category_id, codebook_version, "
            "confidence, source) VALUES ('chk_new', 1, 1, 0.9, 'deductive')"
        )
        conn.commit()
        row = conn.execute(
            "SELECT chunk_id FROM chunk_categories WHERE chunk_id = 'chk_new'"
        ).fetchone()
        assert row is not None
        conn.close()

    def test_upgrade_path_proposals_preserved(self, tmp_path):
        """codebook_proposals-Daten + FK auf categories bleiben erhalten."""
        old = _old_schema_db(tmp_path / "prop.db")
        old.close()

        conn = init_memory_db(tmp_path / "prop.db")
        rows = conn.execute("SELECT category_id, parent_hint_id FROM codebook_proposals").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 3 and rows[0][1] == 1
        conn.close()

    def test_idempotency(self, tmp_path):
        """Zweiter _init_schema-Lauf auf migrierter DB wirft nicht."""
        old = _old_schema_db(tmp_path / "idem.db")
        old.close()

        conn1 = init_memory_db(tmp_path / "idem.db")
        conn1.close()

        # Zweiter Lauf — darf keinen Fehler werfen
        conn2 = init_memory_db(tmp_path / "idem.db")
        tables = {r[0] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "categories" in tables
        assert "codebook_categories" not in tables
        conn2.close()

    def test_fresh_db_has_categories_not_codebooks(self, tmp_path):
        """Frische DB (kein altes Schema): categories vorhanden, codebooks nie erstellt."""
        conn = init_memory_db(tmp_path / "fresh.db")
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "categories" in tables, "frische DB muss categories-Tabelle haben"
        assert "codebooks" not in tables, "frische DB darf keine codebooks-Tabelle haben"
        assert "codebook_categories" not in tables, "frische DB darf keine codebook_categories haben"
        conn.close()

    def test_migrate_codebook_collapse_direct_idempotent_on_fresh(self, tmp_path):
        """_migrate_codebook_collapse auf frischer DB (kein codebook_categories): kein Fehler."""
        conn = init_memory_db(tmp_path / "direct.db")
        # Direktaufruf auf bereits migrierter DB — darf nicht werfen
        _migrate_codebook_collapse(conn)
        conn.close()
