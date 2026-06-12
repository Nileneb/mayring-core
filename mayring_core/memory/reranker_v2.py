"""Memory-Injection v2.0 reranker — runtime model loader + scorer.

Loads ``cache/rerank_v2.json`` (produced by ``tools/train_reranker.py``)
and exposes:

* ``get_active_reranker(query_hint)`` — decides which version to use
  for a given request.
* ``score_v2(stage_scores, model)`` — applies the learned weights to
  the 5 stage-score features (v, s, r, a, f) and returns a sigmoid
  probability in [0, 1] for ranking.

Selection precedence:

    RERANKER_VERSION=v1  →   always v1
    RERANKER_VERSION=v2  →   v2 if model file present, else v1 (silent fallback)
    RERANKER_VERSION=auto →  hash(query_hint) % 2 split (50/50 A/B)
    unset                 →  v1 (default — no surprise behaviour change)

The model JSON is loaded once and cached in process; ``invalidate_v2_cache()``
forces a reload after a fresh training run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Memory-Injection v3 features (Issue #180):
#   * `v, s, r, a`: retrieval stages (vector, symbolic, recency, source-aff)
#
# Removed vs v2 setup:
#   * `f` (linear-combo of v,s,r,a — multikollin)
#   * `sf` (target leakage: derived from same chunk_feedback table as label)
#   * `sl` (~70% rows have sl=0.5 default — almost constant)
#
# score_v2() reads these from stage_scores at runtime. Backward compatible
# with old model files (weights for missing features are treated as 0, so a
# model trained with extra/legacy features still scores).
# cat_match (#270 reranker-v3): strukturierter category_id-Abgleich Query↔Kandidat.
# Geloggt + vom täglichen Trainer gewichtet; backward-compat (fehlt im Modell → w=0).
_FEATURES = ("v", "s", "r", "a", "cat_match")
_LOCK = threading.Lock()
# Per-version cache: version → (model_dict_or_None, mtime). Keyed by version so
# v2/v3/v4 can be loaded independently (reranker-version-table 2026-05-30).
_MODEL_CACHE: dict[str, tuple[dict[str, Any] | None, float]] = {}

_VERSION_RE = re.compile(r"^v\d+$")


def _cache_dir() -> Path:
    # WHY: frisch aufgelöst (nicht Modul-Level-Konstante) damit MAYRING_CACHE_DIR-env
    # in Tests und Prod gleich wirkt — identisch zu model_router._text_override_model().
    from mayring_core.config import _resolve_cache_dir, BASE_DIR
    return _resolve_cache_dir(BASE_DIR)


def _model_path(version: str = "v2") -> Path:
    return _cache_dir() / f"rerank_{version}.json"


# WHY(#180): degenerate-model sanity-gate. Production-incident 2026-05-09:
# das täglich trainierte v2-Modell hatte negative Weights auf vector + recency
# + llm_advisor (sf=8.77 dominant durch target leakage von chunk_feedback).
# Loader lehnt jetzt v_w<0 ODER s_w<0 ab (retrieval-positive Features dürfen
# nicht negativ rankt werden). Alle Calls fallen lautlos auf v1-Weights
# zurück. CHANGE WITH CARE — ohne diesen Gate war 5 Tage v2 produktiv,
# hat aber Vector-Treffer aktiv runter gerankt.
def _load_model(version: str = "v2") -> dict[str, Any] | None:
    """Return the model dict for `version` if its file exists and parses, else None.
    Result is cached per-version based on file mtime so retraining/version-switches
    pick up cleanly on the next call without restart."""
    path = _model_path(version)
    try:
        st = path.stat()
    except (FileNotFoundError, OSError):
        with _LOCK:
            _MODEL_CACHE.pop(version, None)
        return None
    cached = _MODEL_CACHE.get(version)
    if cached is not None and cached[0] is not None and st.st_mtime == cached[1]:
        return cached[0]
    with _LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _log.warning("rerank_%s.json unreadable: %s — falling back to v1", version, e)
            _MODEL_CACHE[version] = (None, st.st_mtime)
            return None
        if not isinstance(data, dict) or "weights" not in data:
            _log.warning("rerank_%s.json malformed — falling back to v1", version)
            _MODEL_CACHE[version] = (None, st.st_mtime)
            return None
        # Sanity-check: ein gelerntes Modell mit NEGATIVEM Vector- ODER
        # Symbolic-Gewicht ist degeneriert. Vector-Similarity und
        # Symbolic-Token-Overlap sind beides retrieval-positive Signale —
        # negative Gewichte heißen das Modell hat etwas Unsinniges aus
        # collinearen Features oder feedback-Lecks gelernt (gesehen bei
        # n_train=1732, sf_weight=8.77, v_weight=-0.51 → 'symbolic-only'-
        # ranking, vector-Treffer wurden RUNTER sortiert). Lieber sauber
        # auf v1 fallen als degenerierte rankings ausliefern. Trainings-
        # bug muss am train_reranker.py-loop behoben werden, nicht hier.
        weights = data.get("weights") or {}
        v_w = float(weights.get("v") or 0.0)
        pt_w = float(weights.get("pt") or 0.0)
        # 2026-06-05: s (symbolic) + re (rationale_edge) sind KEINE v2-Features mehr.
        # s war kollinear zu v (corr +0.51) → Ridge schob das geteilte Query-Match-
        # Signal in +v und −s → der strikte s<0-Gate rejectete JEDES Modell, obwohl
        # v stark positiv war (kein echter Leak). s gedroppt (v trägt den Query-Match).
        # re war am Inferenz-Pfad nicht lieferbar. Hard-Gate jetzt nur noch auf v
        # (echter #180-Leak-Schutz: Vektor-Sim darf nie anti-korrelieren) + pt>=0.
        if v_w < 0 or pt_w < 0:
            _log.error(
                "rerank_%s.json degenerate (v=%.3f pt=%.3f); "
                "refusing to load. Retrieval-positive Features dürfen "
                "nicht negativ rankt werden.", version, v_w, pt_w,
            )
            _MODEL_CACHE[version] = (None, st.st_mtime)
            return None
        _MODEL_CACHE[version] = (data, st.st_mtime)
        return data


def _model_healthy(version: str) -> bool:
    """True iff the version's model file loads + passes the degenerate gate."""
    return _load_model(version) is not None


def invalidate_v2_cache() -> None:
    """Force a reload on the next call. Call after a training run / version switch."""
    with _LOCK:
        _MODEL_CACHE.clear()


def _ab_pick(query_hint: str | None) -> str:
    """50/50 deterministic split. Same query routes to the same version
    so A/B comparisons stay stable across paginated calls."""
    h = hashlib.sha256((query_hint or "").encode("utf-8")).hexdigest()
    return "v2" if int(h[:8], 16) % 2 == 0 else "v1"


def _default_state_path() -> Path:
    return _cache_dir() / "rerank_default.txt"


def _read_runtime_default() -> str:
    """Persisted default decided by the auto-rollout cron.

    Lives in cache/rerank_default.txt because env vars need a restart
    to take effect; a tiny file lets the cron flip the default
    runtime without touching docker-compose. Values: 'auto' / 'v1' /
    'v2'. Default 'auto' so the system runs a 50/50 A/B from day one
    rather than silently sitting on v1 forever.
    """
    p = _default_state_path()
    try:
        v = p.read_text(encoding="utf-8").strip().lower()
        if v in ("v1", "auto") or _VERSION_RE.match(v):
            return v
    except (OSError, FileNotFoundError):
        pass
    return "auto"


def write_runtime_default(version: str) -> str:
    """Set the active reranker version (manual selection from the dashboard table).

    Accepts 'v1' (baseline), 'auto' (A/B), or any 'v<N>' whose model file exists.
    Returns the value actually written.
    """
    version = (version or "").strip().lower()
    if version in ("v1", "auto"):
        pass
    elif _VERSION_RE.match(version):
        if not _model_path(version).exists():
            raise ValueError(f"no model file for {version!r} (rerank_{version}.json missing)")
    else:
        raise ValueError(f"invalid reranker version: {version!r}")
    p = _default_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(version, encoding="utf-8")
    invalidate_v2_cache()
    return version


def _autorollout_path() -> Path:
    return _cache_dir() / "rerank_autorollout.txt"


def read_autorollout_enabled() -> bool:
    """Darf der Auto-Rollout-Cron den Default automatisch flippen? Default True
    (Status quo). 'off' = das A/B-Gate ist abgestellt → die manuelle Wahl
    (write_runtime_default / activate) bleibt stehen, der Cron-Flip ist deaktiviert."""
    try:
        return _autorollout_path().read_text(encoding="utf-8").strip().lower() != "off"
    except (OSError, FileNotFoundError):
        return True


def write_autorollout_enabled(enabled: bool) -> bool:
    p = _autorollout_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("on" if enabled else "off", encoding="utf-8")
    return enabled


def delete_reranker_version(version: str) -> bool:
    """Lösche ein trainiertes Modell-File (rerank_v<N>.json). Guard: nie v1/auto
    (Baseline ohne File) und nie das AKTIVE Modell (sonst zeigt der Runtime-Default
    ins Leere). Returns True wenn gelöscht, False wenn File nicht existierte."""
    version = (version or "").strip().lower()
    if version in ("v1", "auto") or not _VERSION_RE.match(version):
        raise ValueError(f"nicht löschbar: {version!r} (nur trainierte v<N>-Modelle)")
    if _read_runtime_default() == version:
        raise ValueError(f"{version!r} ist aktiv — erst eine andere Version aktivieren")
    p = _model_path(version)
    if not p.exists():
        return False
    p.unlink()
    invalidate_v2_cache()
    return True


def _active_path() -> Path:
    return _cache_dir() / "rerank_active.json"


def read_active_versions() -> list[str]:
    """Return the list of 1–2 active reranker versions.

    Falls back to migrating from ``rerank_default.txt`` when the new file
    is absent or unreadable: a concrete value other than 'auto' becomes a
    single-element list; 'auto' / missing → ['v1'] (safe baseline).
    """
    p = _active_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, list) and 1 <= len(raw) <= 2:
            versions = [str(v).strip().lower() for v in raw]
            if all(_VERSION_RE.match(v) or v == "v1" for v in versions):
                return versions
    except (OSError, FileNotFoundError, json.JSONDecodeError):
        pass
    # WHY: migrate from the old single-default file so existing installs don't
    # fall back to a hardcoded constant when the new file is simply not yet written.
    legacy = _read_runtime_default()
    if legacy and legacy != "auto" and (_VERSION_RE.match(legacy) or legacy == "v1"):
        return [legacy]
    return ["v1"]


def write_active_versions(versions: list[str]) -> list[str]:
    """Persist 1–2 active reranker versions.

    Each version must be 'v1' (baseline, no file required) or a 'v<N>'
    whose model file exists in the cache directory.  Raises ValueError on
    invalid input so callers get an explicit error instead of silent misrouting.
    """
    versions = [str(v).strip().lower() for v in versions]
    if not 1 <= len(versions) <= 2:
        raise ValueError(
            f"active_versions must have 1 or 2 entries, got {len(versions)}"
        )
    for v in versions:
        if not (_VERSION_RE.match(v) or v == "v1"):
            raise ValueError(f"invalid version {v!r} — expected 'v1' or 'v<N>'")
        if v != "v1" and not _model_path(v).exists():
            raise ValueError(
                f"no model file for {v!r} (rerank_{v}.json missing in cache)"
            )
    p = _active_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(versions), encoding="utf-8")
    invalidate_v2_cache()
    return versions


def _ab_pick_pair(query_hint: str | None, versions: list[str]) -> str:
    """Deterministic 50/50 split across exactly two versions using SHA-1."""
    h = hashlib.sha1((query_hint or "").encode("utf-8")).hexdigest()
    return versions[int(h, 16) % 2]


def list_reranker_versions() -> list[dict[str, Any]]:
    """All selectable reranker versions for the dashboard table.

    Always includes the v1 baseline (hardcoded weights, no file) and every
    cache/rerank_v<N>.json. Each entry carries metadata + an `active` flag.
    """
    active = _read_runtime_default()
    out: list[dict[str, Any]] = [{
        "version": "v1", "baseline": True, "active": active == "v1",
        "healthy": True, "trained_at": None, "n_train": None, "metrics": None,
    }]
    try:
        files = sorted(_cache_dir().glob("rerank_v*.json"),
                       key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    except OSError:
        files = []
    for f in files:
        version = f.stem.replace("rerank_", "")  # rerank_v3 → v3
        if not _VERSION_RE.match(version):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        out.append({
            "version": version,
            "baseline": False,
            "active": active == version,
            "healthy": _model_healthy(version),
            "trained_at": data.get("trained_at"),
            "n_train": data.get("n_train"),
            "n_test": data.get("n_test"),
            "metrics": data.get("metrics"),
            "weights": data.get("weights"),
        })
    return out


def get_active_reranker(
    query_hint: str | None = None,
    explicit_override: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve which reranker version to use for THIS call.

    Precedence (highest first):
      1. ``explicit_override`` from request body (?reranker=v2)
      2. ``RERANKER_VERSION`` env var
      3. ``cache/rerank_default.txt`` runtime state file
         (the auto-rollout cron flips this on 25%+ uplift)
      4. fallback: ``auto`` (50/50 A/B)

    Args:
        query_hint: Stable string (query text or session id) that A/B
            mode hashes for the deterministic split.
        explicit_override: Per-request override ('v1' / 'v2' / 'auto').

    Returns:
        (version, model_dict | None)
    """
    # Explicit override and env var take precedence (highest priority).
    pin = explicit_override or os.getenv("RERANKER_VERSION")
    if pin:
        raw = pin.strip().lower()
        if raw not in ("auto", "v1") and not _VERSION_RE.match(raw):
            raw = "auto"
        if raw == "auto":
            raw = _ab_pick(query_hint)
        if raw != "v1":
            model = _load_model(raw)
            if model is None:
                return "v1", None
            return raw, model
        return "v1", None

    # New active-versions table: 1 entry → always that version;
    # 2 entries → deterministic 50/50 hash split.
    active = read_active_versions()
    raw = active[0] if len(active) == 1 else _ab_pick_pair(query_hint, active)
    if raw != "v1":
        model = _load_model(raw)
        if model is None:
            return "v1", None  # silent fallback — never crash on missing/degenerate model
        return raw, model
    return "v1", None


def score_v2(stage_scores: dict[str, float], model: dict[str, Any]) -> float:
    """Apply learned weights to {v,s,r,a,f}. Returns probability in [0,1]
    via sigmoid so the score is comparable to v1's [0,1]-ish range."""
    weights = model.get("weights", {})
    intercept = float(model.get("intercept", 0.0))
    z = intercept
    for f in _FEATURES:
        z += float(weights.get(f, 0.0)) * float(stage_scores.get(f, 0.0))
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def model_summary() -> dict[str, Any]:
    """For /stats/admin endpoints — quick view of the active model."""
    model = _load_model()
    if model is None:
        return {"active": False, "reason": "no rerank_v2.json"}
    return {
        "active": True,
        "version": model.get("version"),
        "trained_at": model.get("trained_at"),
        "n_train": model.get("n_train"),
        "n_test": model.get("n_test"),
        "metrics": model.get("metrics"),
        "weights": model.get("weights"),
        "intercept": model.get("intercept"),
    }
