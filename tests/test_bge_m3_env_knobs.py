"""bge-m3 migration: env-driven embedding model, blue-green collection name,
and the model-specific task-dedup threshold. The embedding model resolves from
the central model_routes.yaml (last-resort bge-m3, never nomic); the collection
name + dedup threshold keep their prior defaults unless the env var is set."""
import importlib

from mayring_core.memory import store


def test_memory_chunks_collection_remapped_via_env(tmp_path, monkeypatch):
    store._chroma_clients.clear()
    store._chroma_collections.clear()
    monkeypatch.setenv("MEMORY_CHUNKS_COLLECTION", "memory_chunks_bge")
    path = tmp_path / "chroma"

    remapped = store.get_chroma_collection("memory_chunks", path=path)
    direct = store.get_chroma_collection("memory_chunks_bge", path=path)
    other = store.get_chroma_collection("codebook_categories", path=path)

    # "memory_chunks" resolves to the bge collection (blue-green flip)...
    assert remapped is direct
    assert remapped.name == "memory_chunks_bge"
    # ...while other collection names pass through untouched.
    assert remapped is not other
    assert other.name == "codebook_categories"


def test_memory_chunks_collection_default_unchanged(tmp_path, monkeypatch):
    store._chroma_clients.clear()
    store._chroma_collections.clear()
    monkeypatch.delenv("MEMORY_CHUNKS_COLLECTION", raising=False)

    col = store.get_chroma_collection("memory_chunks", path=tmp_path / "chroma")
    assert col.name == "memory_chunks"


def test_embedding_model_env_driven(monkeypatch):
    import mayring_core.config as config

    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    importlib.reload(config)
    assert config.EMBEDDING_MODEL == "bge-m3"

    # No env → resolve from the central model_routes.yaml ('embedding' task),
    # last-resort bge-m3. NEVER nomic — the store moved off nomic(768d) long ago
    # and a stray nomic default loaded a 2nd embedder into VRAM (week-old bug).
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    importlib.reload(config)
    assert config.EMBEDDING_MODEL == "bge-m3"


def test_task_sim_threshold_env_driven(monkeypatch):
    import mayring_core.memory.task_derivation as td

    monkeypatch.setenv("MAYRING_TASK_SIM_THRESHOLD", "0.78")
    importlib.reload(td)
    assert td._TASK_SIM_THRESHOLD == 0.78

    monkeypatch.delenv("MAYRING_TASK_SIM_THRESHOLD", raising=False)
    importlib.reload(td)
    assert td._TASK_SIM_THRESHOLD == 0.85
