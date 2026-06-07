"""Deterministic Ampel classification for GitHub-notification hook_events.

The dashboard 'Triggers' panel showed the dead wiki `trigger_stats` table; the
real GitHub notifications (repo_ci / repo_security) already land in `hook_events`
(2974 repo_ci + repo_security in prod). This module classifies them by urgency
(red/yellow/green/grey) straight from the structured payload (conclusion /
severity) — NO LLM: the payload already carries the signal, so deterministic
rules are both cheaper and more reliable than a semantic guess.
"""
from __future__ import annotations

# Which hook_types are surfaced as user-facing notifications (vs session hooks).
NOTIFICATION_HOOK_TYPES: tuple[str, ...] = ("repo_ci", "repo_security")

# Sort key: red first (most urgent), grey last.
URGENCY_ORDER: dict[str, int] = {"red": 0, "yellow": 1, "green": 2, "grey": 3}


def classify_notification(hook_type: str, payload: dict | None) -> str:
    """Return the Ampel urgency ('red'|'yellow'|'green'|'grey') for a hook_event.

    Deterministic, payload-driven. Security events are urgent by default; CI events
    map by their `conclusion`. Unknown hook_types fall through to 'grey' (info)."""
    ht = (hook_type or "").lower()
    data = payload or {}
    conclusion = str(data.get("conclusion", "")).lower()
    severity = str(data.get("severity", "")).lower()

    if ht == "repo_security":
        if severity in ("moderate", "medium", "low"):
            return "yellow"
        return "red"  # critical/high/unknown → urgent (security defaults to red)

    if ht == "repo_ci":
        if conclusion in ("failure", "timed_out", "startup_failure"):
            return "red"
        if conclusion == "success":
            return "green"
        if conclusion == "skipped":
            return "grey"
        # cancelled / action_required / stale / pending / empty → needs attention
        return "yellow"

    return "grey"
