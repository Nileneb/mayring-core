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
# repo_ci / repo_security come from the GitHub-Action → /repo-events pipeline.
# repo_dependabot / repo_pull / repo_issue are the plugin watch-hook's net-new
# types (Hook-A): the Action pipeline does not produce them, so the local gh-poll
# in ci_security_warner.py POSTs them to /stats/notifications/ingest.
NOTIFICATION_HOOK_TYPES: tuple[str, ...] = (
    "repo_ci", "repo_security", "repo_dependabot", "repo_pull", "repo_issue",
)

# Sort key: red first (most urgent), grey last.
URGENCY_ORDER: dict[str, int] = {"red": 0, "yellow": 1, "green": 2, "grey": 3}


def classify_notification(hook_type: str, payload: dict | None) -> str:
    """Return the Ampel urgency ('red'|'yellow'|'green'|'grey') for a hook_event.

    Deterministic, payload-driven. Security/Dependabot events are urgent by default;
    CI events map by their `conclusion`; an assigned issue needs attention (yellow);
    a new PR is informational (grey). Unknown hook_types fall through to 'grey'."""
    ht = (hook_type or "").lower()
    data = payload or {}
    conclusion = str(data.get("conclusion", "")).lower()
    severity = str(data.get("severity", "")).lower()

    if ht in ("repo_security", "repo_dependabot"):
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

    if ht == "repo_issue":
        return "yellow"  # an issue assigned to you needs attention

    if ht == "repo_pull":
        return "grey"  # a newly-opened PR is informational

    return "grey"
