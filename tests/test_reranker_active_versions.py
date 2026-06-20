import json
from mayring_core.memory import reranker_v2 as rr


def test_write_then_read_two_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v3.json").write_text("{}")
    (tmp_path / "rerank_v4.json").write_text("{}")
    rr.write_active_versions(["v3", "v4"])
    assert rr.read_active_versions() == ["v3", "v4"]


def test_max_two(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    for v in ("v2", "v3", "v4"):
        (tmp_path / f"rerank_{v}.json").write_text("{}")
    import pytest
    with pytest.raises(ValueError):
        rr.write_active_versions(["v2", "v3", "v4"])


def test_write_active_syncs_legacy_default(tmp_path, monkeypatch):
    """rerank_default.txt (legacy SoT for the delete-guard + migration fallback)
    must track the primary active version, so the two never diverge (the bug that
    showed default=v4 while serving=v3)."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v3.json").write_text("{}")
    (tmp_path / "rerank_v4.json").write_text("{}")
    rr.write_active_versions(["v3", "v4"])
    assert rr._read_runtime_default() == "v3"


def test_delete_guard_protects_every_active_version(tmp_path, monkeypatch):
    """The delete-guard must protect EVERY serving-active version (both A/B sides),
    not just the legacy single default — else the second A/B model is deletable and
    rerank_active.json points into the void."""
    import pytest
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    for v in ("v3", "v4", "v5"):
        (tmp_path / f"rerank_{v}.json").write_text("{}")
    rr.write_active_versions(["v3", "v4"])
    with pytest.raises(ValueError):
        rr.delete_reranker_version("v4")  # second A/B side — still active
    assert rr.delete_reranker_version("v5") is True  # not active → deletable


def test_migration_from_default_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v4.json").write_text("{}")
    (tmp_path / "rerank_default.txt").write_text("v4")
    assert rr.read_active_versions() == ["v4"]


def test_migration_auto_becomes_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_default.txt").write_text("auto")
    assert rr.read_active_versions() == ["v1"]
