"""v23 Ampel-Notification-Center: deterministic classification + state table."""
from mayring_core.notifications import (
    NOTIFICATION_HOOK_TYPES,
    URGENCY_ORDER,
    classify_notification,
)
from mayring_core.memory import store


def test_ci_conclusion_to_ampel():
    assert classify_notification("repo_ci", {"conclusion": "failure"}) == "red"
    assert classify_notification("repo_ci", {"conclusion": "timed_out"}) == "red"
    assert classify_notification("repo_ci", {"conclusion": "success"}) == "green"
    assert classify_notification("repo_ci", {"conclusion": "skipped"}) == "grey"
    assert classify_notification("repo_ci", {"conclusion": "cancelled"}) == "yellow"
    assert classify_notification("repo_ci", {"conclusion": ""}) == "yellow"


def test_security_defaults_urgent():
    assert classify_notification("repo_security", {"severity": "critical"}) == "red"
    assert classify_notification("repo_security", {"severity": "high"}) == "red"
    assert classify_notification("repo_security", {"severity": "moderate"}) == "yellow"
    assert classify_notification("repo_security", {}) == "red"  # unknown severity → urgent


def test_unknown_and_session_hooks_are_grey():
    assert classify_notification("Stop", {}) == "grey"
    assert classify_notification("UserPromptSubmit", {"conclusion": "failure"}) == "grey"
    assert classify_notification("", None) == "grey"


def test_notification_hook_types_and_order():
    assert "repo_ci" in NOTIFICATION_HOOK_TYPES
    assert "repo_security" in NOTIFICATION_HOOK_TYPES
    assert "Stop" not in NOTIFICATION_HOOK_TYPES
    assert URGENCY_ORDER["red"] < URGENCY_ORDER["yellow"] < URGENCY_ORDER["green"] < URGENCY_ORDER["grey"]


def test_notification_state_table_roundtrip(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "notification_state" in tables
    # a hook_event must exist for the FK target
    conn.execute(
        "INSERT INTO hook_events(id, workspace_id, hook_type, fired_at, payload) "
        "VALUES (1,'ws-1','repo_ci','2026-06-07','{}')")
    conn.execute(
        "INSERT INTO notification_state(hook_event_id, workspace_id, acked, updated_at) "
        "VALUES (1,'ws-1',1,'2026-06-07')")
    row = conn.execute(
        "SELECT acked FROM notification_state WHERE hook_event_id=1").fetchone()
    assert row[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 23
