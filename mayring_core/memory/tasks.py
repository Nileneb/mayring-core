"""Workspace-isolated task tracker — pure storage layer.

No SQL leaks to callers: all queries live here.
Columns are accessed by name via sqlite3.Row.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.schema import is_valid_scope_key

_STATUS = ("open", "in_progress", "done")
_PRIORITY = ("low", "medium", "high")
_COLS = (
    "task_id",
    "workspace_id",
    "title",
    "description",
    "status",
    "priority",
    "due_date",
    "tags",
    "created_by",
    "linked_chunk_id",
    "scope_key",
    "external_id",
    "created_at",
    "updated_at",
    "completed_at",
)
_UPDATABLE = (
    "title",
    "description",
    "status",
    "priority",
    "due_date",
    "tags",
    "linked_chunk_id",
    "scope_key",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: Any) -> dict:
    return {k: row[k] for k in _COLS}


def create_task(
    conn: DBAdapter,
    *,
    workspace_id: str,
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: str | None = None,
    tags: str = "",
    created_by: str | None = None,
    linked_chunk_id: str | None = None,
    scope_key: str | None = None,
    external_id: str | None = None,
) -> dict:
    if not title or not title.strip():
        raise ValueError("title must not be empty")
    if priority not in _PRIORITY:
        raise ValueError(f"priority must be one of {_PRIORITY}, got {priority!r}")
    if not is_valid_scope_key(scope_key):
        raise ValueError(f"invalid scope_key: {scope_key!r}")

    # WHY(task-capture): idempotent capture — a re-fired PostToolUse hook for the
    # same harness task must update the existing row, not create a duplicate.
    if external_id:
        existing = conn.execute(
            "SELECT task_id FROM tasks WHERE workspace_id=? AND external_id=?",
            (workspace_id, external_id),
        ).fetchone()
        if existing is not None:
            existing_id = existing[0]
            conn.execute(
                "UPDATE tasks SET title=?, description=?, priority=?, tags=?, updated_at=? "
                "WHERE task_id=? AND workspace_id=?",
                (title, description, priority, tags, _now(), existing_id, workspace_id),
            )
            conn.commit()
            return get_task(conn, workspace_id, existing_id)  # type: ignore[return-value]

    task_id = "tsk_" + uuid.uuid4().hex[:16]
    now = _now()
    conn.execute(
        """
        INSERT INTO tasks
            (task_id, workspace_id, title, description, status, priority,
             due_date, tags, created_by, linked_chunk_id, scope_key,
             external_id, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            task_id, workspace_id, title, description, priority,
            due_date, tags, created_by, linked_chunk_id, scope_key,
            external_id, now, now,
        ),
    )
    conn.commit()
    return get_task(conn, workspace_id, task_id)  # type: ignore[return-value]


def get_task(conn: DBAdapter, workspace_id: str, task_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM tasks WHERE task_id=? AND workspace_id=?",
        (task_id, workspace_id),
    ).fetchone()
    return _row_to_dict(row) if row is not None else None


def list_tasks(
    conn: DBAdapter,
    workspace_id: str,
    *,
    status: str | None = None,
    tag: str | None = None,
    priority: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM tasks WHERE workspace_id=?"
    params: list[Any] = [workspace_id]

    if status is not None:
        sql += " AND status=?"
        params.append(status)
    if priority is not None:
        sql += " AND priority=?"
        params.append(priority)
    if tag is not None:
        sql += " AND (',' || tags || ',') LIKE ?"
        params.append(f"%,{tag},%")

    sql += """
        ORDER BY
            CASE status WHEN 'open' THEN 0 WHEN 'in_progress' THEN 1 ELSE 2 END,
            CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
            created_at DESC
    """
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_task(
    conn: DBAdapter,
    workspace_id: str,
    task_id: str,
    **fields,
) -> dict | None:
    if get_task(conn, workspace_id, task_id) is None:
        return None

    unknown = set(fields) - set(_UPDATABLE)
    if unknown:
        raise ValueError(f"unknown updatable fields: {unknown}")

    if "status" in fields and fields["status"] not in _STATUS:
        raise ValueError(f"status must be one of {_STATUS}, got {fields['status']!r}")
    if "priority" in fields and fields["priority"] not in _PRIORITY:
        raise ValueError(f"priority must be one of {_PRIORITY}, got {fields['priority']!r}")

    now = _now()
    set_parts = [f"{col}=?" for col in fields]
    set_values = list(fields.values())

    if "status" in fields:
        set_parts.append("completed_at=?")
        set_values.append(now if fields["status"] == "done" else None)

    set_parts.append("updated_at=?")
    set_values.append(now)

    conn.execute(
        f"UPDATE tasks SET {', '.join(set_parts)} WHERE task_id=? AND workspace_id=?",
        (*set_values, task_id, workspace_id),
    )
    conn.commit()
    return get_task(conn, workspace_id, task_id)


def complete_task(conn: DBAdapter, workspace_id: str, task_id: str) -> dict | None:
    return update_task(conn, workspace_id, task_id, status="done")


def delete_task(conn: DBAdapter, workspace_id: str, task_id: str) -> bool:
    conn.execute(
        "DELETE FROM tasks WHERE task_id=? AND workspace_id=?",
        (task_id, workspace_id),
    )
    conn.commit()
    return conn.changes() > 0


