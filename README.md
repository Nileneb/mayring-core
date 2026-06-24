# mayring-core

> **Ökosystem:** Teil des 4-Repo-MayringCoder-Systems (Layer 3 · Core/Ports).
> Gesamtkarte: [`MayringCoder/ARCHITECTURE.md`](https://github.com/Nileneb/MayringCoder/blob/master/ARCHITECTURE.md).
> Eingebunden als Git-Submodule `vendor/mayring-core`. Importiert **nie** `src.*` —
> Host-Implementierungen kommen über `mayring_core.providers` (DI).

Shared core library for the MayringCoder ecosystem: memory retrieval/ingestion,
LLM routing, identity/workspace resolution, and the Ollama client.

Extracted from [`MayringCoder`](https://github.com/Nileneb/MayringCoder) (`core/`)
in #267 and split into its own repo to break the `MayringCoder ↔ mayring-pi-agent`
dependency cycle: [`mayring-pi-agent`](https://github.com/Nileneb/mayring-pi-agent)
now depends on this package directly instead of pulling the entire MayringCoder
repo via a git-subdirectory dependency.

## Install

```bash
pip install -e .                                              # local dev
pip install "mayring-core @ git+https://github.com/Nileneb/mayring-core.git@v0.1.0"
```

Runtime deps are deliberately core-only (`httpx`, `pyyaml`, `chromadb`) — no
API/agent stack (fastapi, uvicorn, mcp, anthropic, …).

## Consumers

- **MayringCoder** — vendored as the `vendor/mayring-core` git-submodule (editable).
- **mayring-pi-agent** — git-dependency pinned to a tag.
