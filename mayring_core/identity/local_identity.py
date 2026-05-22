"""Lokal gecachte User-Identity für CLI-Pfade.

`mayring login` schreibt hier User.id + JWT-Token; resolve_workspace()
liest die User.id beim CLI-Aufruf, damit Aufrufe nicht jedes Mal
explizit --workspace user-N brauchen.

Datei: ~/.config/mayring/identity.json
Permissions: 0600 (Token darin gespeichert).

Format:
    {
        "user_id": 2,
        "email": "user@example.com",
        "token": "<JWT>",          // optional, kann fehlen
        "issued_at": "2026-05-09T12:00:00+00:00",
        "expires_at": "2026-08-07T12:00:00+00:00"  // optional
    }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _identity_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mayring" / "identity.json"


@dataclass
class LocalIdentity:
    user_id: int
    email: str | None = None
    token: str | None = None
    issued_at: str | None = None
    expires_at: str | None = None

    def is_token_valid(self) -> bool:
        if not self.expires_at:
            return self.token is not None
        try:
            exp = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < exp


def get_local_user_id() -> int | None:
    """Resolve User.id with the documented fallback chain.

    Order:
      1. ENV MAYRING_USER_ID (numeric)
      2. ~/.config/mayring/identity.json
      3. None (caller must error out — no silent 'default')
    """
    env_uid = os.environ.get("MAYRING_USER_ID", "").strip()
    if env_uid:
        try:
            return int(env_uid)
        except ValueError:
            pass  # malformed env → fall through to file
    ident = load_identity()
    if ident is not None:
        return ident.user_id
    return None


def load_identity() -> LocalIdentity | None:
    path = _identity_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return LocalIdentity(
            user_id=int(data["user_id"]),
            email=data.get("email"),
            token=data.get("token"),
            issued_at=data.get("issued_at"),
            expires_at=data.get("expires_at"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_identity(identity: LocalIdentity) -> Path:
    """Persist identity to ~/.config/mayring/identity.json with 0600.

    Atomic via tempfile + rename so partial writes don't corrupt the file
    if the user ctrl-c's during login.
    """
    path = _identity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_id": identity.user_id,
        "email": identity.email,
        "token": identity.token,
        "issued_at": identity.issued_at or datetime.now(timezone.utc).isoformat(),
        "expires_at": identity.expires_at,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return path


def clear_identity() -> bool:
    """Remove the cached identity. Returns True if a file was deleted."""
    path = _identity_path()
    if path.exists():
        path.unlink()
        return True
    return False
