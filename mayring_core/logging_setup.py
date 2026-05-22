"""Strukturiertes JSON-Logging + Live-Memory-Ingest für MayringCoder.

Wird beim FastAPI-/Pi-Server-Start aufgerufen. Konfiguriert:

1. **JsonFormatter** + RotatingFileHandler nach
   ``$MAYRING_LOG_FILE`` (default ``cache/logs/mayring.json``).

2. **MemoryIngestHandler** (optional, opt-in via env):
   batched async POST an ``$MAYRING_API_URL/memory/log-event``
   mit X-Workspace-Id für ``bene:logs``-Sub-Workspace.
   Trigger:
     - level ≥ ERROR/CRITICAL → SOFORT flush (count irrelevant)
     - sonst: nach N lines (default 100) → batch-flush
   Failure-tolerance: bei Endpoint-fail → retry-queue auf Disk
   (``cache/logs/_pending.jsonl``), nächster successful flush
   drained mit. App-Pfad blockt nie.

Damit fällt ein separater 5-min-Cron (alter Polling-Approach) weg.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


SEVERE_LEVELS = {"ERROR", "CRITICAL", "EXCEPTION", "FATAL"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        for key, val in record.__dict__.items():
            if key in ("args", "asctime", "created", "exc_info", "exc_text",
                      "filename", "funcName", "levelname", "levelno",
                      "lineno", "module", "msecs", "message", "msg", "name",
                      "pathname", "process", "processName", "relativeCreated",
                      "stack_info", "thread", "threadName", "taskName"):
                continue
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = str(val)
        return json.dumps(payload, ensure_ascii=False)


class MemoryIngestHandler(logging.Handler):
    """Live-Trigger-Handler.

    Statt eines 5-min-Cron-Pollings sendet jeder Logger Records direkt
    asynchron an /memory/log-event sobald entweder ein severity-Trigger
    feuert ODER der Buffer voll ist.
    """

    def __init__(
        self,
        api_url: str,
        token: str,
        service: str,
        workspace_slug: str = "bene:logs",
        batch_size: int = 100,
        max_queue: int = 5000,
    ) -> None:
        super().__init__(level=logging.WARNING)
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.service = service
        self.workspace_slug = workspace_slug
        self.batch_size = batch_size
        self._queue: queue.Queue[dict] = queue.Queue(max_queue)
        self._stop = threading.Event()
        self._wake = threading.Event()  # instance-level (NICHT class)
        self._thread = threading.Thread(target=self._run_worker, daemon=True)
        self._thread.start()
        atexit.register(self._final_flush)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = {
                "ts": logging.Formatter().formatTime(
                    record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "exc": self.format_exception(record),
            }
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                # Queue voll → drop oldest, behalte neueste (severe lines
                # sind die wichtigsten).
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(event)
            # Severity-Trigger: signal worker, batch sofort flushen.
            if record.levelname.upper() in SEVERE_LEVELS:
                self._wake.set()
        except Exception:
            self.handleError(record)

    @staticmethod
    def format_exception(record: logging.LogRecord) -> str | None:
        if record.exc_info:
            return logging.Formatter().formatException(record.exc_info)
        return None

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            self._drain_until_threshold()

    def _drain_until_threshold(self) -> None:
        batch: list[dict] = []
        while True:
            try:
                ev = self._queue.get_nowait()
            except queue.Empty:
                break
            batch.append(ev)
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)

    def _flush(self, events: list[dict]) -> None:
        if not events:
            return
        body = {"service": self.service, "events": events}
        req = urllib.request.Request(
            self.api_url + "/memory/log-event",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "X-Workspace-Id": self.workspace_slug,
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except (urllib.error.URLError, OSError) as exc:
            # Persist auf Disk für nächsten Drain.
            self._persist_pending(events)

    def _persist_pending(self, events: list[dict]) -> None:
        path = Path(os.environ.get("MAYRING_LOG_FILE",
                                   "cache/logs/mayring.json")).parent / "_pending.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _final_flush(self) -> None:
        self._stop.set()
        self._wake.set()
        time.sleep(0.5)
        self._drain_until_threshold()


def configure_json_logging() -> None:
    """Idempotent — wird beim Server-Start aufgerufen."""
    if os.environ.get("MAYRING_LOG_FILE_CONFIGURED") == "1":
        return
    log_file = os.environ.get("MAYRING_LOG_FILE", "cache/logs/mayring.json")
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(path),
        maxBytes=int(os.environ.get("MAYRING_LOG_MAX_BYTES",
                                    str(50 * 1024 * 1024))),
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Live-Memory-Ingest opt-in. In Container: MCP_SERVICE_TOKEN ist da
    # → handler ist aktiv. Lokal/CI: token leer → handler skipped.
    token = os.environ.get("MCP_SERVICE_TOKEN", "").strip()
    if token:
        api_url = os.environ.get("MAYRING_INTERNAL_API_URL",
                                 "http://localhost:8090")
        service = os.environ.get("MAYRING_SERVICE_NAME", "mayring-api")
        workspace = os.environ.get("LOG_INGEST_TARGET_SLUG", "bene:logs")
        root.addHandler(MemoryIngestHandler(
            api_url=api_url, token=token, service=service,
            workspace_slug=workspace,
            batch_size=int(os.environ.get("LOG_INGEST_BATCH_SIZE", "100")),
        ))

    os.environ["MAYRING_LOG_FILE_CONFIGURED"] = "1"
