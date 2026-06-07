"""log_llm_call_async: fire-and-forget telemetry write must land without blocking
the caller, and must use its own connection (not the caller's request conn)."""
import time

from mayring_core.memory import store


def _wait_for_row(conn, call_type: str, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        n = conn.execute(
            "SELECT count(*) FROM llm_calls_log WHERE call_type=?", (call_type,)
        ).fetchone()[0]
        if n:
            return n
        time.sleep(0.05)
    return 0


def test_async_log_lands_in_background(tmp_path):
    db = tmp_path / "m.db"
    # init once so the schema exists; the background thread opens its OWN conn to db.
    conn = store.init_memory_db(db)
    store.log_llm_call_async(
        call_type="vector_search", model="bge-m3", prompt="probe",
        response='{"vector_stage":"ok"}', workspace_id="ws-1", db_path=db,
    )
    assert _wait_for_row(conn, "vector_search") == 1
    row = conn.execute(
        "SELECT model, workspace_id FROM llm_calls_log WHERE call_type='vector_search'"
    ).fetchone()
    assert row[0] == "bge-m3" and row[1] == "ws-1"


def test_async_log_never_raises_on_bad_path(tmp_path):
    # A broken db_path must not surface from the daemon thread (best-effort telemetry).
    store.log_llm_call_async(
        call_type="vector_search", model="bge-m3", prompt="x", response="{}",
        db_path=tmp_path / "nonexistent_dir" / "nope.db",
    )
    time.sleep(0.3)  # give the thread time to fail silently
    # No assertion needed beyond "did not raise" — reaching here is the pass.
