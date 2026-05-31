"""Role-based permission model (tenancy Phase B).

Two axes, decided 2026-05-31: visibility {private/org/public} (Phase A) is
ORTHOGONAL to role {user/editor/admin} (this module). Roles arrive in the JWT
(Membership.role, issued by app.linn.games, legacy vocab owner/editor/viewer)
and are normalised here. The per-workspace policy lives in
`workspace_role_permissions`; a missing row falls back to DEFAULT_MATRIX.
"""
from __future__ import annotations

ROLES = ("user", "editor", "admin")
PERMISSIONS = (
    "search", "write", "share_org", "share_public",
    "manage_members", "manage_foreign",
)

DEFAULT_MATRIX: dict[tuple[str, str], bool] = {
    **{("user", p): p in ("search", "write") for p in PERMISSIONS},
    **{("editor", p): p in ("search", "write", "share_org", "share_public") for p in PERMISSIONS},
    **{("admin", p): True for p in PERMISSIONS},
}

ADMIN_LOCKED_PERMISSIONS = ("manage_members",)

_LEGACY_ROLE_MAP = {"owner": "admin", "viewer": "user"}


def normalize_role(role: str | None) -> str:
    """Map any incoming role string to the canonical {user,editor,admin}.
    Unknown / missing → 'user' (least privilege)."""
    if not role:
        return "user"
    r = _LEGACY_ROLE_MAP.get(role, role)
    return r if r in ROLES else "user"
