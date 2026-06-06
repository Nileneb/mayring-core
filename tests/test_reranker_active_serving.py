from mayring_core.memory import reranker_v2 as rr


def test_single_active_always_that_version(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("RERANKER_VERSION", raising=False)
    (tmp_path / "rerank_v4.json").write_text('{"weights":{}}')
    rr.write_active_versions(["v4"])
    versions = {rr.get_active_reranker(query_hint=f"q{i}")[0] for i in range(20)}
    assert versions == {"v4"}


def test_two_active_splits_between_both(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("RERANKER_VERSION", raising=False)
    (tmp_path / "rerank_v3.json").write_text('{"weights":{}}')
    (tmp_path / "rerank_v4.json").write_text('{"weights":{}}')
    rr.write_active_versions(["v3", "v4"])
    versions = {rr.get_active_reranker(query_hint=f"q{i}")[0] for i in range(50)}
    assert versions == {"v3", "v4"}
