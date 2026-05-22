from __future__ import annotations

import sqlite3
from pathlib import Path


class DBAdapter:
    """Thin wrapper around sqlite3.Connection.

    Centralises all SQLite-specific code: PRAGMA setup, PRAGMA table_info,
    SELECT changes(). All other files hold plain SQL — no sqlite3 imports needed.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def create(cls, path: str | Path, *, check_same_thread: bool = False) -> "DBAdapter":
        conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return cls(conn)

    @classmethod
    def memory(cls) -> "DBAdapter":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return cls(conn)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params) -> sqlite3.Cursor:
        return self._conn.executemany(sql, params)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        return self._conn.executescript(sql)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def get_columns(self, table: str) -> set[str]:
        """Return column names via PRAGMA table_info (SQLite-specific)."""
        if not table.isidentifier():
            raise ValueError(f"Invalid table name: {table!r}")
        return {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def changes(self) -> int:
        """Return row count affected by last INSERT/UPDATE/DELETE (SQLite-specific)."""
        return self._conn.execute("SELECT changes()").fetchone()[0]
