"""Ollama model discovery and interactive model selection."""

from __future__ import annotations

import json
import sys

import httpx


_FALLBACK_MODEL = "llama3.1:8b"


def fetch_ollama_models(ollama_url: str, timeout: int = 2) -> list[str] | None:
    """Query /api/tags and return a sorted list of model names.

    Returns None if Ollama is unreachable or the response is malformed.
    """
    url = ollama_url.rstrip("/") + "/api/tags"
    try:
        resp = httpx.get(url, timeout=float(timeout))
        resp.raise_for_status()
        data = resp.json()
        # Exclude embedding-only models — they can't be used for text generation.
        # Sentence-Transformers wie all-MiniLM, BGE, nomic-embed-text werden auf
        # Ollama OHNE "embed" im Namen veröffentlicht ("all-minilm:l6-v2"),
        # /api/generate quittiert sie mit 400 Bad Request. Production-incident
        # 2026-05-09: Ambient-snapshot-Generator wählte all-minilm:l6-v2 →
        # 0 Snapshots in 5 Tagen, niemand hat es gemerkt weil der Job-status
        # 'done' meldet (CLI fängt den 400 silent).
        _EMBED_KEYWORDS = ("embed", "embedding", "minilm", "bge", "gte-", "nomic-")
        models = [
            m["name"] for m in data.get("models", [])
            if "name" in m and not any(kw in m["name"].lower() for kw in _EMBED_KEYWORDS)
        ]
        return sorted(models) if models else None
    except Exception:
        return None


def prompt_user_for_model(available_models: list[str]) -> str:
    """Display a numbered menu of models and return the user's selection.

    All display output goes to stderr so that stdout stays clean for shell
    command substitution.
    Falls back to the first model in the list on invalid input after 3 attempts.
    """
    print("\nKein Modell konfiguriert. Verfügbare Ollama-Modelle:", file=sys.stderr)
    for i, name in enumerate(available_models, 1):
        print(f"  {i}. {name}", file=sys.stderr)

    names_lower = {m.lower(): m for m in available_models}

    attempts = 0
    while attempts < 3:
        try:
            print(f"\nModell auswählen [1–{len(available_models)} oder Name]: ", end="", flush=True, file=sys.stderr)
            raw = input("").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return available_models[0]

        # Accept numeric index.
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(available_models):
                return available_models[idx - 1]

        # Accept exact or case-insensitive name match.
        if raw.lower() in names_lower:
            return names_lower[raw.lower()]

        attempts += 1
        remaining = 3 - attempts
        if remaining > 0:
            print(f"Ungültige Eingabe (Nummer oder Name). Noch {remaining} Versuch(e).", file=sys.stderr)

    print(f"Zu viele ungültige Eingaben — verwende '{available_models[0]}'.", file=sys.stderr)
    return available_models[0]


def resolve_model(ollama_url: str, cli_model: str | None, env_model: str | None) -> str:
    """Resolve the model to use.

    Priority: CLI flag → env var → MAYRING_DEFAULT_MODEL env →
    erstes verfügbares Ollama-Modell → hardcoded fallback.

    KEIN interactive prompt mehr — Backend-Pipelines (queue-worker,
    cron-jobs, smoke) hatten sonst hängende stdin-reads bzw. fielen
    durch EOFError auf das erste Modell der Liste, das oft ein
    Embedding-only-Modell war. Plus: das Output 'Kein Modell
    konfiguriert. Modell auswählen [1–15]…' verwirrte User in der
    Job-History.
    """
    if cli_model:
        return cli_model
    if env_model:
        return env_model

    import os
    default_env = os.environ.get("MAYRING_DEFAULT_MODEL", "").strip()
    if default_env:
        return default_env

    available = fetch_ollama_models(ollama_url)
    if available:
        # Erstes nicht-embedding-Modell aus der Ollama-Liste.
        # fetch_ollama_models filtert embeddings schon raus.
        print(f"# resolve_model: kein --model/MAYRING_DEFAULT_MODEL gesetzt, "
              f"nehme erstes Ollama-Modell: {available[0]}", file=sys.stderr)
        return available[0]

    print(f"# resolve_model: Ollama nicht erreichbar — fallback "
          f"'{_FALLBACK_MODEL}'", file=sys.stderr)
    return _FALLBACK_MODEL
