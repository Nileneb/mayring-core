import os
from pathlib import Path
from mayring_core.model_router import ModelRouter


def test_text_override_file_wins_over_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "text_model.txt").write_text("qwen3.5-mayring:2b", encoding="utf-8")
    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") == "qwen3.5-mayring:2b"


def test_no_override_falls_back_to_yaml_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") != ""


def test_empty_override_file_falls_back_to_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "text_model.txt").write_text("   \n", encoding="utf-8")
    r = ModelRouter(ollama_url="http://x")
    yaml_default = "mistral:7b-instruct"
    assert r.resolve("text") == yaml_default


def test_override_does_not_affect_other_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "text_model.txt").write_text("qwen3.5-mayring:2b", encoding="utf-8")
    r = ModelRouter(ollama_url="http://x")
    assert r.resolve("text") == "qwen3.5-mayring:2b"
    assert r.resolve("embedding") != "qwen3.5-mayring:2b"
