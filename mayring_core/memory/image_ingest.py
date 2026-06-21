"""Batch image discovery and ingestion pipeline."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from mayring_core.memory.schema import Source
from mayring_core.memory.store import init_memory_db


def discover_images(root: Path, max_file_bytes: int = 5 * 1024 * 1024, max_images: int = 50) -> list[Path]:
    """Walk root recursively, return image files within size and count limits."""
    from mayring_core.memory.ingest import _IMAGE_EXTENSIONS  # local import avoids circular dep

    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(found) >= max_images:
            break
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                print(f"[ingest-images] Skip (too large): {path.relative_to(root)}")
                continue
        except OSError:
            continue
        found.append(path)
    return found


def run_image_ingest(
    repo_url: str,
    ollama_url: str,
    vision_model: str = "qwen2.5vl:3b",
    embed_model: str | None = None,
    max_images: int = 50,
    max_file_bytes: int = 5 * 1024 * 1024,
    force_reingest: bool = False,
    workspace_id: str = "default",
) -> dict:
    """Shallow-clone repo, caption all images, ingest into memory."""
    # Resolve the embedder from the single source of truth, never a nomic literal.
    if embed_model is None:
        from mayring_core.config import EMBEDDING_MODEL
        embed_model = EMBEDDING_MODEL
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("github.com/", 1)
    repo_owner_name = parts[1] if len(parts) == 2 else url
    repo_slug = repo_owner_name.replace("/", "-").lower()
    clone_dir = tempfile.mkdtemp(prefix=f"mayring-{repo_slug}-images-")
    try:
        return _run_image_ingest_from_clone(
            repo_url=url, clone_dir=Path(clone_dir),
            repo_owner_name=repo_owner_name, ollama_url=ollama_url,
            vision_model=vision_model, embed_model=embed_model,
            max_images=max_images, max_file_bytes=max_file_bytes,
            force_reingest=force_reingest, workspace_id=workspace_id,
        )
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)


def _run_image_ingest_from_clone(
    repo_url: str, clone_dir: Path, repo_owner_name: str, ollama_url: str,
    vision_model: str, embed_model: str, max_images: int, max_file_bytes: int,
    force_reingest: bool, workspace_id: str = "default",
) -> dict:
    print(f"[ingest-images] git clone --depth=1 {repo_url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

    images = discover_images(clone_dir, max_file_bytes=max_file_bytes, max_images=max_images)
    print(f"[ingest-images] {len(images)} Bilder gefunden")
    if not images:
        return {"images_found": 0, "images_captioned": 0, "images_skipped": 0, "images_failed": 0, "repo": repo_owner_name}

    from mayring_core.memory.store import deactivate_chunks_by_source
    from mayring_core.memory.retrieval import invalidate_query_cache
    from mayring_core.memory.ingest import get_or_create_chroma_collection, ingest_image  # local import avoids circular dep

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()
    captioned = skipped = failed = 0

    if force_reingest:
        for img_path in images:
            rel = str(img_path.relative_to(clone_dir))
            deactivate_chunks_by_source(conn, Source.make_id(repo_owner_name, rel))
        invalidate_query_cache()

    try:
        for img_path in images:
            rel = str(img_path.relative_to(clone_dir))
            source = Source(
                source_id=Source.make_id(repo_owner_name, rel),
                source_type="repo_file", repo=repo_owner_name, path=rel,
            )
            try:
                r = ingest_image(
                    source=source, image_path=img_path, conn=conn,
                    chroma_collection=chroma, ollama_url=ollama_url,
                    model=embed_model, vision_model=vision_model, workspace_id=workspace_id,
                )
                if r.get("deduped", 0) > 0:
                    skipped += 1
                    print(f"[ingest-images] Dedup: {rel}")
                else:
                    captioned += 1
                    print(f"[ingest-images] OK: {rel}")
            except Exception as exc:
                failed += 1
                print(f"[ingest-images] FEHLER {rel}: {exc}")
    finally:
        conn.close()

    return {"images_found": len(images), "images_captioned": captioned,
            "images_skipped": skipped, "images_failed": failed, "repo": repo_owner_name}
