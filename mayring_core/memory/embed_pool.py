"""Distributed embedding pool (#365 Schicht 3) — one row per embed task with two
claim slots (A/B). Two DISTINCT devices each fill a slot; on the second result the
vectors are compared (see submit_result, Task 4). Mirrors pi_jobs' atomic-claim
pattern (SELECT candidates → guarded UPDATE) but with dual slots so 'two distinct
devices' is a single UPDATE guard (device_a != claimer)."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any

from mayring_core.embed_verify import cosine, verify

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
            is_golden    INTEGER NOT NULL DEFAULT 0,  -- is_golden/golden_ref: golden test-jobs, populated in Task 4 (collusion-breaker)
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
    """All columns as a plain dict — drift-free as the DDL grows (sqlite3.Row → dict)."""
    return dict(row)


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


def submit_result(conn: Any, *, embed_id: str, device_id: str,
                  vector: list[float], threshold: float) -> dict:
    """Store this device's vector into its slot. When both slots are filled, compare
    by cosine: agreement → status 'verified' + agreed_vector (slot A as reference);
    divergence → status 'diverged'. Returns a verdict dict the caller acts on
    (canonical write on agreement; quarantine on divergence)."""
    ensure_tables(conn)
    row = conn.execute("SELECT * FROM embed_jobs WHERE embed_id = ?", (embed_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown embed_id {embed_id!r}")
    payload = json.dumps(vector)
    if row["device_a"] == device_id:
        conn.execute("UPDATE embed_jobs SET result_a=? WHERE embed_id=?", (payload, embed_id))
    elif row["device_b"] == device_id:
        conn.execute("UPDATE embed_jobs SET result_b=? WHERE embed_id=?", (payload, embed_id))
    else:
        raise ValueError(f"device {device_id!r} holds no slot on {embed_id!r}")
    conn.commit()
    # WHY(#365): re-read both slots after the slot write — the slot UPDATE has no RETURNING (A/B branch), so re-SELECT is the clean way to read both results atomically (sqlite serialises writes).
    row = conn.execute("SELECT * FROM embed_jobs WHERE embed_id = ?", (embed_id,)).fetchone()
    if not (row["result_a"] and row["result_b"]):
        return {"status": row["status"], "verdict": ""}
    va, vb = json.loads(row["result_a"]), json.loads(row["result_b"])
    sim = cosine(va, vb)
    now = _now_iso()
    devices = [row["device_a"], row["device_b"]]
    if verify(va, vb, threshold=threshold):
        conn.execute(
            "UPDATE embed_jobs SET status='verified', verdict='agreement', "
            "cosine=?, verified_at=? WHERE embed_id=?", (sim, now, embed_id))
        conn.commit()
        return {"status": "verified", "verdict": "agreement", "agreed_vector": va,
                "cosine": sim, "devices": devices, "chunk_ref": row["chunk_ref"],
                "projekt_id": row["projekt_id"]}
    conn.execute(
        "UPDATE embed_jobs SET status='diverged', verdict='divergence', "
        "cosine=?, verified_at=? WHERE embed_id=?", (sim, now, embed_id))
    conn.commit()
    return {"status": "diverged", "verdict": "divergence", "cosine": sim, "devices": devices}


def enqueue_golden(conn: Any, *, workspace_id: str, text: str,
                   reference: list[float], model: str = "bge-m3") -> str:
    """Enqueue a golden test-job: a known sample whose embedding is compared against a stored reference vector — the collusion-breaker for quarantined devices."""
    ensure_tables(conn)
    eid = _new_id()
    conn.execute(
        "INSERT INTO embed_jobs (embed_id, workspace_id, text, model, is_golden, "
        "golden_ref, status, created_at) VALUES (?, ?, ?, ?, 1, ?, 'queued', ?)",
        (eid, workspace_id, text, model, json.dumps(reference), _now_iso()),
    )
    conn.commit()
    return eid


def claim_golden(conn: Any, *, device_id: str, workspace_id: str) -> dict | None:
    """Assign the oldest queued golden job to this (typically quarantined) device."""
    ensure_tables(conn)
    row = conn.execute(
        "UPDATE embed_jobs SET status='claimed_one', device_a=? "
        "WHERE embed_id=(SELECT embed_id FROM embed_jobs WHERE workspace_id=? "
        "AND is_golden=1 AND status='queued' ORDER BY created_at LIMIT 1) "
        "AND status='queued' RETURNING *",
        (device_id, workspace_id),
    ).fetchone()
    conn.commit()
    return _row_to_dict(row) if row else None


def submit_golden(conn: Any, *, embed_id: str, device_id: str,
                  vector: list[float], threshold: float) -> dict:
    """Compare a golden result to its stored reference. passed=True rehabilitates."""
    ensure_tables(conn)
    row = conn.execute("SELECT * FROM embed_jobs WHERE embed_id = ?", (embed_id,)).fetchone()
    if row is None or not row["is_golden"]:
        raise ValueError(f"{embed_id!r} is not a golden job")
    ref = json.loads(row["golden_ref"])
    sim = cosine(vector, ref)
    passed = verify(vector, ref, threshold=threshold)
    conn.execute(
        "UPDATE embed_jobs SET result_a=?, status=?, verdict=?, cosine=?, verified_at=? "
        "WHERE embed_id=?",
        (json.dumps(vector), "verified" if passed else "failed",
         "golden_pass" if passed else "golden_fail", sim, _now_iso(), embed_id),
    )
    conn.commit()
    return {"passed": passed, "cosine": sim, "device_id": device_id}


__all__ = (
    "VALID_STATUS", "ensure_tables", "enqueue", "get", "claim_replica",
    "submit_result", "enqueue_golden", "claim_golden", "submit_golden",
)
