"""CLI-Helper für die kanonische Workspace-Auflösung.

Verwendet vom CLI-Pfad (src/cli.py, src/workflows/*) anstelle des alten
Patterns `getattr(args, "workspace_id", "default") or "default"`.

Auflösungs-Reihenfolge:
    1. args.workspace_id (explicit user choice)
    2. ENV MAYRING_USER_ID
    3. ~/.config/mayring/identity.json (von `mayring login`)
    4. raise IdentityRequiredError — kein silent 'default'-Fallback

Wenn der Caller einen Read-only-Pfad hat (Listing, etc.) und keinen
auto-create wünscht: explizit `auto_create=False` setzen.
"""
from __future__ import annotations

from mayring_core.identity.local_identity import get_local_user_id, load_identity
from mayring_core.identity.workspace_resolver import resolve_workspace
from mayring_core.memory.db_adapter import DBAdapter


def resolve_cli_workspace(
    args, *, conn: DBAdapter | None = None, auto_create: bool = True
) -> str:
    """Resolve the CLI's workspace argument to a canonical workspace_id.

    Lazy-imports init_memory_db so test fixtures with monkeypatched
    CACHE_DIR aren't broken by an early-binding singleton.
    """
    candidate = getattr(args, "workspace_id", None)
    if conn is None:
        from mayring_core.config import CACHE_DIR
        from mayring_core.memory.store import init_memory_db
        conn = init_memory_db(CACHE_DIR / "memory.db")

    ident = load_identity()
    return resolve_workspace(
        conn,
        candidate,
        default_user_id=get_local_user_id(),
        default_email=ident.email if ident else None,
        auto_create_user_workspace=auto_create,
    )
