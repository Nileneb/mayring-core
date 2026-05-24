"""Kanonische Workspace-Auflösung.

Diese Schicht ist die EINE Quelle der Wahrheit für workspace_id.

Workspace-IDs sind seit 2026-05-09 KEINE 'user-N'-Krücken mehr,
sondern human-readable Slugs aus dem Email-Localpart (z.B. 'bene'
für bene@linn.games). Vorteile:
  - Logs lesbar: 'workspace=bene' statt 'workspace=user-2'.
  - Multi-Tenant-vorbereitet: 'team-acme', 'bene', 'system' im selben
    Bucket-Namespace ohne Prefix-Konflikte.
  - Deterministisch: gleiche Email → gleicher Slug, auch wenn 3 Apps
    parallel JWT-Tokens minten.

Auflösungs-Reihenfolge:
  1. input ist None + default_user_id+default_email → ensure_user_workspace
  2. 'system' → kanonischer system-bucket
  3. input existiert in workspaces → return as-is
  4. input ist alias → return aliased canonical
  5. legacy 'user-N' Pattern (für alte JWT-Tokens vor dem Refactor)
     → ensure(slug=user-N, owner=N) als Fallback bis alle Clients
     emails senden
  6. else: raise UnknownWorkspaceError
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mayring_core.memory.db_adapter import DBAdapter

if TYPE_CHECKING:
    from mayring_core.identity.types import TokenInfoLike

USER_WORKSPACE_RE = re.compile(r"^user-(\d+)(?::([\w\-]+))?$")


class UnknownWorkspaceError(ValueError):
    """Raised when an alias/workspace cannot be resolved against the DB."""


class IdentityRequiredError(ValueError):
    """Raised when no workspace_id and no default_user_id was provided."""


# WHY(v2-stufe1-SoT, identity): Diese Funktion ist die EINZIGE quelle
# der wahrheit für workspace_id pro request. Vor dem refactor gab es
# 4+ resolution-pfade (api.auth.get_workspace, mcp_auth._effective_*,
# ad-hoc f"user-{sub}" string-builder, jwt-claim-derivation in
# validate_jwt_token). Mehrere unfixed callsites = workspace=system-leaks.
# Audit: docs/v2-master-audit.md W1.
def _canonicalize_alias(conn, workspace_id: str) -> str:
    """Map a workspace_id through workspace_aliases to its canonical target.

    WHY(workspace-repoint 2026-05-24): nach dem Re-Point 019d6933→019e14d6 tragen
    zirkulierende Tokens (hook.jwt, watcher) noch die alte UUID. Ohne diesen Lookup
    läsen/schrieben sie auf die verwaiste 019d6933 (leer → Hooks finden nichts,
    Straggler-Writes). Der Alias macht alte Tokens transparent gültig — genau
    wofür workspace_aliases da ist. Fail-soft: kein conn / kein Treffer → unverändert.
    """
    if conn is None:
        return workspace_id
    try:
        row = conn.execute(
            "SELECT workspace_id FROM workspace_aliases WHERE alias = ?",
            (workspace_id,)).fetchone()
    except Exception:  # noqa: BLE001 — alias-Tabelle fehlt (alte DB) → kein Alias
        return workspace_id
    return row[0] if row else workspace_id


def resolve_workspace_from_token(
    info: "TokenInfoLike",
    override_header: str | None = None,
    conn=None,
) -> str:
    """ONLY function used to determine workspace_id for any request.

    Args:
        info: Validated TokenInfo (from auth dependency or middleware).
        override_header: Value of X-Workspace-Id header, if any.

    Resolution rules:
        - Service-Token (scope='*') + override-Header → return override
          (admin-tool case: post-deploy-ingest, smoke, cron writes for
          a specific tenant).
        - Service-Token w/o override → return token.workspace_id (typically
          'system').
        - User-JWT (scope='mcp:memory'[+'admin']) → return token.workspace_id.
          Override-Header IGNORIERT (User darf nicht in fremde Buckets).

    Raises:
        ValueError: info is None, or token has empty workspace_id.
    """
    if info is None:
        raise ValueError("resolve_workspace_from_token: info is None")
    workspace_id = info.workspace_id
    if not workspace_id:
        raise ValueError(
            "resolve_workspace_from_token: TokenInfo.workspace_id is empty — "
            "JWT must carry email-claim (-> slug), service-token must default "
            "to 'system'."
        )
    is_service_token = "*" in info.scopes
    if is_service_token and override_header:
        candidate = override_header.strip()
        if candidate:
            return _canonicalize_alias(conn, candidate)
    return _canonicalize_alias(conn, workspace_id)


def email_to_slug(email: str) -> str:
    """Email → human-readable slug für Workspace-IDs.

    'bene@linn.games' → 'bene'
    'foo.bar@example.com' → 'foo-bar'
    'admin+tag@firma.de' → 'admin-tag'

    Slugs sind ASCII-only, lowercase, ohne special chars außer '-'.
    Idempotent: slug(slug(x)) == slug(x).
    """
    local = (email or "").split("@", 1)[0].strip().lower()
    if not local:
        return ""
    out = []
    prev_dash = False
    for ch in local:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def resolve_workspace(
    conn: DBAdapter,
    workspace_input: str | None,
    *,
    default_user_id: int | None = None,
    default_email: str | None = None,
    default_display_name: str | None = None,
    auto_create_user_workspace: bool = True,
) -> str:
    """Resolve a free-form workspace identifier to its canonical form.

    Args:
        conn: open memory.db adapter
        workspace_input: what the caller passed (CLI arg, JWT claim, etc.)
        default_user_id: app.linn.games User.id used if input is None
        auto_create_user_workspace: if True (default), 'user-N' inputs
            that don't yet exist in the workspaces table are upserted as
            kind=user. This keeps JWT-flows zero-friction: a fresh user's
            first request mints their workspace on the fly.

    Returns:
        canonical workspace_id string

    Raises:
        IdentityRequiredError: input is None and no default_user_id
        UnknownWorkspaceError: input is a string we can't map
    """
    # Tolerant gegen non-string Inputs (z.B. MagicMock in Tests). Production-
    # CLI/JWT-Pfade liefern immer str | None. Alles andere wird wie missing
    # behandelt — der hard-error kommt dann sauber über default_user_id.
    if not isinstance(workspace_input, str):
        workspace_input = None
    if workspace_input is None or workspace_input.strip() == "":
        if default_user_id is None:
            raise IdentityRequiredError(
                "workspace_id is None and no default_user_id provided. "
                "Set MAYRING_USER_ID env or run `mayring login`, or pass "
                "--workspace explicitly."
            )
        # Email → Slug. KEIN user-N-Fallback: pre-launch heißt
        # alle echten User haben Email. Fehlt sie → echter Bug, hart
        # failen statt Datenmüll-Bucket schreiben.
        canonical = email_to_slug(default_email or "")
        if not canonical:
            raise IdentityRequiredError(
                f"Workspace-Auflösung für user_id={default_user_id} braucht "
                f"eine Email — sub-only-Tokens werden nicht mehr akzeptiert. "
                f"JwtIssuer (app.linn.games) schickt 'email'-Claim seit 2026-04-18, "
                f"CLI: mayring login schreibt sie in identity.json."
            )
        if auto_create_user_workspace:
            ensure_user_workspace(
                conn, default_user_id,
                slug=canonical, email=default_email,
                display_name=default_display_name,
            )
        return canonical

    candidate = workspace_input.strip()

    # 'system' ist ein reservierter Server-Side-Bucket: post-deploy-ingest,
    # Service-Token-Aufrufe (api/auth.py:37), conversation_watcher,
    # ambient-Snapshots. Wird beim DB-init via ensure_system_workspace
    # angelegt; Resolver akzeptiert ihn auch wenn die Row noch fehlt.
    if candidate == "system":
        if auto_create_user_workspace:
            ensure_system_workspace(conn)
        return candidate

    # USER_WORKSPACE_RE-Shortcut wurde 2026-05-09 entfernt: pre-launch,
    # daher kein impliziter user-N-Bucket mehr. Workspaces müssen
    # explizit via ensure_user_workspace(slug, email, ...) angelegt
    # werden, NICHT durch resolve('user-2').

    # Lookup direct workspace
    row = conn.execute(
        "SELECT id FROM workspaces WHERE id = ?", (candidate,)
    ).fetchone()
    if row:
        return row[0]

    # Lookup alias
    row = conn.execute(
        "SELECT workspace_id FROM workspace_aliases WHERE alias = ?",
        (candidate,),
    ).fetchone()
    if row:
        return row[0]

    raise UnknownWorkspaceError(
        f"workspace_id={candidate!r} is neither a known workspace nor a "
        f"registered alias. Use `mayring workspace add` or pass user-N."
    )


def ensure_system_workspace(conn: DBAdapter) -> str:
    """Upsert kind=system workspace (Service-Token, Cron-Jobs, ambient)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, display_name,
                                   created_at, updated_at)
           VALUES ('system', 'system', NULL, 'System (Service-Token / Cron)',
                   ?, ?)
           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
        (now, now),
    )
    conn.commit()
    return "system"


def ensure_team_workspace(
    conn: DBAdapter,
    org_id: str,
    *,
    display_name: str | None = None,
) -> str:
    """Upsert kind=team workspace for an app.linn.games organization.

    org_id is the org-workspace UUID from the JWT memberships[]. Owned and
    managed by app.linn.games (owner_user_id stays NULL here) — this row only
    makes the org a first-class local FK target + dashboard label. Membership/
    roles are NOT mirrored; they live in the JWT.
    """
    if not org_id:
        raise IdentityRequiredError("ensure_team_workspace braucht eine org_id.")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, display_name,
                                   created_at, updated_at)
           VALUES (?, 'team', NULL, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               display_name = COALESCE(excluded.display_name, workspaces.display_name),
               updated_at = excluded.updated_at""",
        (org_id, display_name or org_id, now, now),
    )
    conn.commit()
    return org_id


def ensure_user_workspace(
    conn: DBAdapter,
    user_id: int,
    *,
    slug: str | None = None,
    email: str | None = None,
    display_name: str | None = None,
) -> str:
    """Upsert kind=user workspace. Email PFLICHT — kein user-N-Fallback."""
    workspace_id = slug or email_to_slug(email or "")
    if not workspace_id:
        raise IdentityRequiredError(
            f"ensure_user_workspace(user_id={user_id}) braucht email oder slug. "
            f"User-N-Buckets sind seit 2026-05-09 verboten."
        )
    now = datetime.now(timezone.utc).isoformat()
    pretty_name = display_name or email or workspace_id
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, email,
                                   display_name, created_at, updated_at)
           VALUES (?, 'user', ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               email = COALESCE(excluded.email, workspaces.email),
               display_name = COALESCE(excluded.display_name, workspaces.display_name),
               updated_at = excluded.updated_at""",
        (workspace_id, user_id, email, pretty_name, now, now),
    )
    conn.commit()
    return workspace_id


def ensure_project_workspace(
    conn: DBAdapter,
    user_id: int,
    slug: str,
    *,
    email: str,
    parent_slug: str | None = None,
    display_name: str | None = None,
) -> str:
    """Upsert kind=project sub-workspace unter dem User-Workspace.

    parent_slug optional — wenn nicht gegeben, wird er aus email
    abgeleitet. Email ist PFLICHT (kein user-N-Fallback).
    """
    parent = parent_slug or email_to_slug(email)
    if not parent:
        raise IdentityRequiredError(
            "ensure_project_workspace braucht email oder parent_slug."
        )
    workspace_id = f"{parent}:{slug}"
    now = datetime.now(timezone.utc).isoformat()
    ensure_user_workspace(
        conn, user_id, slug=parent, email=email, display_name=display_name,
    )
    conn.execute(
        """INSERT INTO workspaces (id, kind, parent_id, owner_user_id,
                                   display_name, created_at, updated_at)
           VALUES (?, 'project', ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
        (workspace_id, parent, user_id, slug, now, now),
    )
    conn.commit()
    return workspace_id


def add_alias(conn: DBAdapter, alias: str, workspace_id: str) -> None:
    """Register a non-canonical name as alias for a canonical workspace.

    Useful for legacy-import: `add_alias(conn, "default", "user-2")`.
    """
    row = conn.execute(
        "SELECT id FROM workspaces WHERE id = ?", (workspace_id,)
    ).fetchone()
    if row is None:
        raise UnknownWorkspaceError(
            f"cannot alias unknown workspace {workspace_id!r}"
        )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspace_aliases (alias, workspace_id, created_at)
           VALUES (?, ?, ?)
           ON CONFLICT(alias) DO UPDATE SET workspace_id = excluded.workspace_id""",
        (alias, workspace_id, now),
    )
    conn.commit()


def list_workspaces_for_user(conn: DBAdapter, user_id: int) -> list[dict]:
    """Return canonical + project workspaces owned by a user."""
    rows = conn.execute(
        """SELECT id, kind, parent_id, display_name FROM workspaces
           WHERE owner_user_id = ? ORDER BY kind, id""",
        (user_id,),
    ).fetchall()
    return [
        {"id": r[0], "kind": r[1], "parent_id": r[2], "display_name": r[3]}
        for r in rows
    ]
