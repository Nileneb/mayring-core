"""Device-Registry + Hook-Event store (#274).

Cloud-side persistence for the device↔cloud channel (companion to
mayring-claude-plugin#5). A device (plugin install or Pi-worker node) registers
a stable ``device_id`` + its ``capabilities``; hook firings are recorded per
device for the app.linn.games Observability dashboard.

This module is the SINGLE SOURCE OF TRUTH for write-job routing:
``effective_capabilities`` resolves a claiming worker's capabilities from the
registry, NOT from the worker's self-report — so a ``capability_required='write'``
job can only ever reach a device that is registered with ``'write'``
(join ``pi_jobs.claimed_by`` ↔ ``devices.device_id``).

Lazy ``ensure_tables`` (CREATE IF NOT EXISTS) mirrors
``src/api/integrations/notifications_store.py``. The canonical schema also lives
in ``mayring_core.memory.store._init_schema`` (schema v5); the lazy path keeps
this module usable against a raw sqlite connection in tests and robust if the
``user_version`` gate skipped the DDL on an already-migrated prod DB.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# WHY(#274): 'write' is the one capability that must be registry-confirmed —
# a write-job mutates real files, so a worker cannot grant it to itself by
# self-reporting. See effective_capabilities().
WRITE_CAPABILITY = "write"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_caps(raw: str | None) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def _join_caps(caps: Any) -> str:
    """Normalise capabilities to a deduped, comma-separated string."""
    items = _split_caps(caps) if isinstance(caps, str) else list(caps or [])
    seen: set[str] = set()
    out: list[str] = []
    for c in items:
        c = str(c).strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return ",".join(out)


def ensure_tables(conn: Any) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS devices (
            device_id    TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            name         TEXT NOT NULL DEFAULT '',
            os           TEXT NOT NULL DEFAULT '',
            capabilities TEXT NOT NULL DEFAULT '',
            last_seen    TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'active',
            created_at   TEXT NOT NULL DEFAULT '',
            trust_score       REAL NOT NULL DEFAULT 0,
            embed_verified    INTEGER NOT NULL DEFAULT 0,
            embed_divergences INTEGER NOT NULL DEFAULT 0,
            quarantined_until TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (device_id, workspace_id)
        );
        CREATE INDEX IF NOT EXISTS idx_devices_workspace
            ON devices(workspace_id);

        CREATE TABLE IF NOT EXISTS hook_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            device_id    TEXT NOT NULL DEFAULT '',
            hook_type    TEXT NOT NULL DEFAULT '',
            fired_at     TEXT NOT NULL DEFAULT '',
            payload      TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_hook_events_ws_fired
            ON hook_events(workspace_id, fired_at);
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

def upsert_device(
    conn: Any,
    *,
    device_id: str,
    workspace_id: str = "default",
    name: str = "",
    os: str = "",
    capabilities: Any = "",
) -> None:
    """Register/refresh a device. The caller declares its full state, so name,
    os and capabilities are overwritten; created_at is preserved across
    re-registers; last_seen + status are bumped to now/active."""
    ensure_tables(conn)
    now = _now_iso()
    conn.execute(
        """INSERT INTO devices (device_id, workspace_id, name, os, capabilities,
                                last_seen, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
           ON CONFLICT(device_id, workspace_id) DO UPDATE SET
               name         = excluded.name,
               os           = excluded.os,
               capabilities = excluded.capabilities,
               last_seen    = excluded.last_seen,
               status       = 'active'""",
        (device_id, workspace_id, name, os, _join_caps(capabilities), now, now),
    )
    conn.commit()


def touch_last_seen(conn: Any, device_id: str, workspace_id: str = "default") -> None:
    """Heartbeat: bump last_seen + mark active. If the device hasn't registered
    yet, create a minimal row so a heartbeat/hook from it still surfaces in
    /devices — capabilities stay empty (it can claim no capability-gated job
    until it registers properly)."""
    ensure_tables(conn)
    now = _now_iso()
    conn.execute(
        """INSERT INTO devices (device_id, workspace_id, name, os, capabilities,
                                last_seen, status, created_at)
           VALUES (?, ?, '', '', '', ?, 'active', ?)
           ON CONFLICT(device_id, workspace_id) DO UPDATE SET
               last_seen = excluded.last_seen,
               status    = 'active'""",
        (device_id, workspace_id, now, now),
    )
    conn.commit()


def list_devices(conn: Any, workspace_id: str) -> list[dict]:
    ensure_tables(conn)
    # WHY(#274, observability): workspace_id == "system" (Service-Token) liefert
    # cross-workspace — konsistent mit dashboard.py ("system sieht alles"). Die
    # strikte per-workspace-Prüfung fürs Write-Routing bleibt in
    # device_capabilities/effective_capabilities, NICHT hier (read-only Admin-View).
    _cols = (
        "SELECT device_id, workspace_id, name, os, capabilities, "
        "last_seen, status, created_at, trust_score, embed_verified, "
        "embed_divergences, quarantined_until FROM devices"
    )
    if workspace_id == "system":
        rows = conn.execute(_cols + " ORDER BY last_seen DESC").fetchall()
    else:
        rows = conn.execute(
            _cols + " WHERE workspace_id = ? ORDER BY last_seen DESC",
            (workspace_id,),
        ).fetchall()
    return [
        {
            "device_id": r[0],
            "workspace_id": r[1],
            "name": r[2],
            "os": r[3],
            "capabilities": _split_caps(r[4]),
            "last_seen": r[5],
            "status": r[6],
            "created_at": r[7],
            "trust_score": r[8],
            "embed_verified": r[9],
            "embed_divergences": r[10],
            "quarantined_until": r[11],
        }
        for r in rows
    ]


def device_capabilities(
    conn: Any, device_id: str, workspace_id: str | None = None
) -> list[str] | None:
    """Return a device's REGISTERED capabilities (authoritative), or None if the
    device isn't registered (in this workspace when given). None ≠ [] on purpose:
    None means "unknown device", [] means "registered with no capabilities"."""
    ensure_tables(conn)
    if workspace_id:
        row = conn.execute(
            "SELECT capabilities FROM devices WHERE device_id = ? AND workspace_id = ?",
            (device_id, workspace_id),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT capabilities FROM devices WHERE device_id = ?",
            (device_id,),
        ).fetchone()
    if row is None:
        return None
    return _split_caps(row[0])


def effective_capabilities(
    conn: Any,
    device_id: str,
    workspace_id: str | None,
    self_reported: list[str] | None,
) -> list[str]:
    """Capabilities to feed claim-matching. The registry wins: a registered
    device's stored caps are authoritative (a worker cannot self-grant 'write').
    An UNregistered device's self-reported caps are honoured EXCEPT 'write' —
    write requires an explicit register, so the distributor only routes
    write-jobs to a registry-confirmed write device (#274 acceptance)."""
    registered = device_capabilities(conn, device_id, workspace_id)
    if registered is not None:
        return registered
    return [c for c in (self_reported or []) if c != WRITE_CAPABILITY]


# ---------------------------------------------------------------------------
# Embed pool: trust / quarantine / eligibility (#365 Schicht 3)
# ---------------------------------------------------------------------------

def record_embed_verified(conn: Any, device_id: str, workspace_id: str = "default") -> None:
    """One agreement: +1 verified, +1.0 trust."""
    ensure_tables(conn)
    conn.execute(
        "UPDATE devices SET embed_verified = embed_verified + 1, "
        "trust_score = trust_score + 1.0 "
        "WHERE device_id = ? AND workspace_id = ?",
        (device_id, workspace_id),
    )
    conn.commit()


def record_embed_divergence(conn: Any, device_id: str, workspace_id: str = "default") -> None:
    """One divergence: +1 divergence, -2.0 trust (divergence costs more than agreement earns)."""
    ensure_tables(conn)
    # WHY(#365): 2:1 asymmetry — one bad embed needs two agreements to recover, so a noisy device drops out of eligibility instead of lingering.
    conn.execute(
        "UPDATE devices SET embed_divergences = embed_divergences + 1, "
        "trust_score = trust_score - 2.0 "
        "WHERE device_id = ? AND workspace_id = ?",
        (device_id, workspace_id),
    )
    conn.commit()


def set_quarantine(conn: Any, device_id: str, workspace_id: str, *, until: str) -> None:
    """Set (or clear, until='') the quarantine deadline. WHY(#365): normalise to the
    Z-suffix _now_iso() format on write so the lexicographic ISO compare in
    eligible_embed_devices is sound regardless of the caller's input format."""
    ensure_tables(conn)
    if until:
        until = datetime.fromisoformat(until.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "UPDATE devices SET quarantined_until = ? WHERE device_id = ? AND workspace_id = ?",
        (until, device_id, workspace_id),
    )
    conn.commit()


def eligible_embed_devices(conn: Any, workspace_id: str, *, now: str, fresh_seconds: int) -> list[str]:
    """device_ids eligible to claim embed jobs: registered with 'embed' capability,
    heartbeat fresher than fresh_seconds, and not currently quarantined (now >= until,
    or until empty). Quarantine gate uses lexicographic ISO compare; freshness parses
    last_seen vs now."""
    def _parse(ts: str):
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    ensure_tables(conn)
    now_dt = _parse(now)
    rows = conn.execute(
        "SELECT device_id, capabilities, last_seen, quarantined_until "
        "FROM devices WHERE workspace_id = ?",
        (workspace_id,),
    ).fetchall()
    out: list[str] = []
    for did, caps, last_seen, quar in rows:
        if "embed" not in _split_caps(caps):
            continue
        if quar and now < quar:
            continue
        ls = _parse(last_seen)
        if ls is None:  # WHY(#365): unknown/empty heartbeat = exclude (fail-closed, not fail-open)
            continue
        if now_dt is not None and (now_dt - ls).total_seconds() > fresh_seconds:
            continue
        out.append(did)
    return out


def is_eligible_embed(conn: Any, device_id: str, workspace_id: str, *, now: str, fresh_seconds: int) -> bool:
    return device_id in eligible_embed_devices(conn, workspace_id, now=now, fresh_seconds=fresh_seconds)


# ---------------------------------------------------------------------------
# Hook events
# ---------------------------------------------------------------------------

def record_hook_event(
    conn: Any,
    *,
    workspace_id: str,
    device_id: str = "",
    hook_type: str = "",
    fired_at: str = "",
    summary: str = "",
) -> int:
    """Best-effort insert of one hook firing; returns the row id. A firing also
    derives a heartbeat for the device (it's alive)."""
    ensure_tables(conn)
    payload = json.dumps({"summary": summary}) if summary else "{}"
    cur = conn.execute(
        "INSERT INTO hook_events (workspace_id, device_id, hook_type, fired_at, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        (workspace_id, device_id, hook_type, fired_at or _now_iso(), payload),
    )
    conn.commit()
    if device_id:
        touch_last_seen(conn, device_id, workspace_id)
    return cur.lastrowid


def list_hook_events(
    conn: Any, workspace_id: str, since: str | None = None, limit: int = 200
) -> list[dict]:
    ensure_tables(conn)
    # workspace_id == "system" (Service-Token) → cross-workspace (s. list_devices).
    clauses: list[str] = []
    params: list[Any] = []
    if workspace_id != "system":
        clauses.append("workspace_id = ?")
        params.append(workspace_id)
    if since:
        clauses.append("fired_at > ?")
        params.append(since)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    rows = conn.execute(
        "SELECT id, workspace_id, device_id, hook_type, fired_at, payload "
        "FROM hook_events" + where + " ORDER BY fired_at DESC LIMIT ?",
        params,
    ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r[5]) if r[5] else {}
        except (TypeError, ValueError):
            payload = {}
        out.append({
            "id": r[0],
            "workspace_id": r[1],
            "device_id": r[2],
            "hook_type": r[3],
            "fired_at": r[4],
            "payload": payload,
        })
    return out


__all__ = (
    "WRITE_CAPABILITY",
    "ensure_tables",
    "upsert_device",
    "touch_last_seen",
    "list_devices",
    "device_capabilities",
    "effective_capabilities",
    "record_embed_verified",
    "record_embed_divergence",
    "set_quarantine",
    "eligible_embed_devices",
    "is_eligible_embed",
    "record_hook_event",
    "list_hook_events",
)
