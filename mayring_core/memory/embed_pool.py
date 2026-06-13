"""Distributed embedding pool (#365 Schicht 3) — one row per embed task with two
claim slots (A/B). Two DISTINCT devices each fill a slot; on the second result the
vectors are compared (see submit_result, Task 4). Mirrors pi_jobs' atomic-claim
pattern (SELECT candidates → guarded UPDATE) but with dual slots so 'two distinct
devices' is a single UPDATE guard (device_a != claimer)."""
from __future__ import annotations

import hashlib
import secrets
import time
from datetime import datetime, timezone
from typing import Any

VALID_STATUS = ("queued", "claimed_one", "claimed_two", "verified", "diverged", "failed")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    raw = f"{time.time_ns()}:{secrets.token_hex(4)}"
    return "emb_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def ensure_tables(conn: Any) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS embed_jobs (
            embed_id     TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            projekt_id   TEXT NOT NULL DEFAULT '',
            text         TEXT NOT NULL,
            chunk_ref    TEXT NOT NULL DEFAULT '',
            model        TEXT NOT NULL DEFAULT 'bge-m3',
            is_golden    INTEGER NOT NULL DEFAULT 0,
            golden_ref   TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'queued',
            device_a     TEXT NOT NULL DEFAULT '',
            result_a     TEXT NOT NULL DEFAULT '',
            device_b     TEXT NOT NULL DEFAULT '',
            result_b     TEXT NOT NULL DEFAULT '',
            cosine       REAL,
            verdict      TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL DEFAULT '',
            verified_at  TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_embed_jobs_ws_status
            ON embed_jobs(workspace_id, status, created_at);
        """
    )
    conn.commit()


def _row_to_dict(row: Any) -> dict:
    keys = ("embed_id", "workspace_id", "projekt_id", "text", "chunk_ref", "model",
            "is_golden", "golden_ref", "status", "device_a", "result_a", "device_b",
            "result_b", "cosine", "verdict", "created_at", "verified_at")
    return {k: row[k] for k in keys}


def enqueue(conn: Any, *, workspace_id: str, projekt_id: str, text: str,
            chunk_ref: str, model: str = "bge-m3") -> str:
    ensure_tables(conn)
    eid = _new_id()
    conn.execute(
        "INSERT INTO embed_jobs (embed_id, workspace_id, projekt_id, text, "
        "chunk_ref, model, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)",
        (eid, workspace_id, projekt_id, text, chunk_ref, model, _now_iso()),
    )
    conn.commit()
    return eid


def get(conn: Any, embed_id: str) -> dict | None:
    ensure_tables(conn)
    row = conn.execute("SELECT * FROM embed_jobs WHERE embed_id = ?", (embed_id,)).fetchone()
    return _row_to_dict(row) if row else None


def claim_replica(conn: Any, *, device_id: str, workspace_id: str) -> dict | None:
    """Atomically take one open slot for this device. Picks the oldest row that is
    queued (→ slot A) or claimed_one by a DIFFERENT device (→ slot B). Guarded UPDATE
    (status + device_a guard) so two concurrent claimers cannot both win the same slot."""
    if not device_id:
        raise ValueError("device_id required")
    if not workspace_id:
        raise ValueError("workspace_id required (tenant boundary)")
    ensure_tables(conn)
    rows = conn.execute(
        "SELECT * FROM embed_jobs WHERE workspace_id = ? AND is_golden = 0 "
        "AND status IN ('queued', 'claimed_one') ORDER BY created_at LIMIT 20",
        (workspace_id,),
    ).fetchall()
    for r in rows:
        if r["status"] == "queued":
            updated = conn.execute(
                "UPDATE embed_jobs SET status='claimed_one', device_a=? "
                "WHERE embed_id=? AND status='queued' RETURNING *",
                (device_id, r["embed_id"]),
            ).fetchone()
        else:  # claimed_one
            if r["device_a"] == device_id:
                continue  # same device can't take both slots
            updated = conn.execute(
                "UPDATE embed_jobs SET status='claimed_two', device_b=? "
                "WHERE embed_id=? AND status='claimed_one' AND device_a != ? RETURNING *",
                (device_id, r["embed_id"], device_id),
            ).fetchone()
        if updated is None:
            continue  # lost the race
        conn.commit()
        return _row_to_dict(updated)
    return None


__all__ = ("VALID_STATUS", "ensure_tables", "enqueue", "get", "claim_replica")
