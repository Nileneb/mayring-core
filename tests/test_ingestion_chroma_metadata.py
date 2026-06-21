"""Test that ingest() writes visibility/org_id/user_id into Chroma metadata.

Tenancy Phase A: these three fields must land in every upserted Chroma document
so that Chroma $where filters can scope retrieval across org/public chunks from
other tenants without a SQLite round-trip.
"""
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from mayring_core.memory.ingestion.core import ingest
from mayring_core.memory.schema import Source
from mayring_core.memory.store import init_memory_db


# ---------------------------------------------------------------------------
# Fake Chroma collection that captures the last upsert call
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self):
        self._upsert_calls: list[dict] = []

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[Any],
        metadatas: list[dict],
    ) -> None:
        self._upsert_calls.append({
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        })

    def count(self) -> int:
        return len(self._upsert_calls)

    @property
    def last_metadatas(self) -> list[dict]:
        if not self._upsert_calls:
            return []
        return self._upsert_calls[-1]["metadatas"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(visibility: str = "org", org_id: str = "org-x", user_id: str = "u-7") -> Source:
    return Source(
        source_id="test:owner/repo:path/to/file.py",
        source_type="repo_file",
        repo="owner/repo",
        path="path/to/file.py",
        visibility=visibility,
        org_id=org_id,
        user_id=user_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChromaMetadataVisibility:

    def test_visibility_org_id_user_id_in_chroma_metadata(self, monkeypatch, tmp_path):
        """Chroma upsert must include visibility, org_id, user_id."""
        fake_emb = [0.1] * 384

        # Patch embed_texts in mayring_core.providers so that the lazy local
        # import inside ingest() (`from mayring_core.providers import embed_texts
        # as _embed_texts`) picks up the stub.
        import mayring_core.providers as _providers
        monkeypatch.setattr(_providers, "embed_texts", lambda texts, url: [fake_emb] * len(texts))

        # Patch conversation_filter (loaded lazily inside the chunk loop)
        import mayring_core.memory.ingestion.conversation_filter as _cf
        monkeypatch.setattr(_cf, "should_skip_chunk", lambda text, stype: (False, ""))

        # SQLite DB in temp dir — avoids touching real ~/.cache/mayringcoder
        db_conn = init_memory_db(tmp_path / "memory.db")

        collection = _FakeCollection()
        source = _make_source(visibility="org", org_id="org-x", user_id="u-7")

        ingest(
            source=source,
            content="Hello tenancy. This chunk carries org visibility.",
            conn=db_conn,
            chroma_collection=collection,
            ollama_url="http://localhost:11434",
            model="",   # no LLM categorize
            opts={"categorize": False},
        )

        assert collection.count() > 0, "Expected at least one Chroma upsert"

        for meta in collection.last_metadatas:
            assert "visibility" in meta, "metadata must contain 'visibility'"
            assert "org_id" in meta, "metadata must contain 'org_id'"
            assert "user_id" in meta, "metadata must contain 'user_id'"
            assert meta["visibility"] == "org"
            assert meta["org_id"] == "org-x"
            assert meta["user_id"] == "u-7"
            # repo-scoping-hardfilter + reference-doc-layer: repo + source_class
            # must land in the vector index so build_chroma_where can hard-filter.
            assert meta["repo"] == "owner/repo"
            assert meta["source_class"] == "code"

    def test_reference_source_class_in_chroma_metadata(self, monkeypatch, tmp_path):
        """A reference source must carry source_class='reference' into Chroma."""
        fake_emb = [0.1] * 384
        import mayring_core.providers as _providers
        monkeypatch.setattr(_providers, "embed_texts", lambda texts, url: [fake_emb] * len(texts))
        import mayring_core.memory.ingestion.conversation_filter as _cf
        monkeypatch.setattr(_cf, "should_skip_chunk", lambda text, stype: (False, ""))

        db_conn = init_memory_db(tmp_path / "memory.db")
        collection = _FakeCollection()
        source = Source(
            source_id="unity-docs:webgl/tier",
            source_type="note",
            repo="",
            path="webgl/tier",
            visibility="private",
            user_id="u-7",
            source_class="reference",
        )
        ingest(
            source=source,
            content="Unity WebGL tier reference documentation chunk.",
            conn=db_conn,
            chroma_collection=collection,
            ollama_url="http://localhost:11434",
            model="",
            opts={"categorize": False},
        )
        assert collection.count() > 0
        for meta in collection.last_metadatas:
            assert meta["source_class"] == "reference"

    def test_none_org_id_becomes_empty_string(self, monkeypatch, tmp_path):
        """Chroma rejects None values — org_id/user_id=None must land as ''."""
        fake_emb = [0.1] * 384

        import mayring_core.providers as _providers
        monkeypatch.setattr(_providers, "embed_texts", lambda texts, url: [fake_emb] * len(texts))

        import mayring_core.memory.ingestion.conversation_filter as _cf
        monkeypatch.setattr(_cf, "should_skip_chunk", lambda text, stype: (False, ""))

        db_conn = init_memory_db(tmp_path / "memory.db")

        collection = _FakeCollection()
        source = Source(
            source_id="test:owner/repo:path/private.py",
            source_type="repo_file",
            repo="owner/repo",
            path="path/private.py",
            visibility="private",
            org_id=None,
            user_id=None,
        )

        ingest(
            source=source,
            content="Private chunk, no org, no user.",
            conn=db_conn,
            chroma_collection=collection,
            ollama_url="http://localhost:11434",
            model="",
            opts={"categorize": False},
        )

        assert collection.count() > 0, "Expected at least one Chroma upsert"

        for meta in collection.last_metadatas:
            assert meta["visibility"] == "private"
            assert meta["org_id"] == "", "None org_id must become empty string"
            assert meta["user_id"] == "", "None user_id must become empty string"
