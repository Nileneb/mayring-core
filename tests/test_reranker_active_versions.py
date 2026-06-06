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


def test_migration_from_default_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v4.json").write_text("{}")
    (tmp_path / "rerank_default.txt").write_text("v4")
    assert rr.read_active_versions() == ["v4"]


def test_migration_auto_becomes_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_default.txt").write_text("auto")
    assert rr.read_active_versions() == ["v1"]
