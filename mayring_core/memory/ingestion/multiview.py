"""Multi-view chunking for GitHub issues: fact/impl/decision/entities + raw."""
from __future__ import annotations

from mayring_core.memory.schema import Chunk
from mayring_core.memory.ingestion.utils import coerce_str


_MULTIVIEW_SYSTEM_PROMPT = """Du bist ein präziser Informationsextrahierer.
Antworte NUR mit dem angeforderten JSON-Objekt, ohne Erklärungen oder Markdown-Blöcke."""

_MULTIVIEW_PROMPT_TEMPLATE = """Analysiere dieses GitHub Issue und extrahiere vier strukturierte Sichten.

ISSUE-TEXT:
{content}

Antworte mit genau diesem JSON:
{{
  "fact_summary": "<2-4 Sätze: Wer meldet was, welcher konkrete Fehler/Feature-Request, betroffene Komponente>",
  "impl_summary": "<2-4 Sätze: Betroffene Module/Dateien/Services, vermutete Ursache, mögliche Fix-Strategien. Leer lassen wenn unklar>",
  "decision_summary": "<2-4 Sätze: Getroffene Entscheidungen, gewählte Ansätze, offene Fragen. Leer lassen wenn keine>",
  "entities_keywords": "<kommagetrennte Liste: Technologien, Fehlercodes, Dateinamen, Komponenten, Schlüsselbegriffe>"
}}"""


def generate_multiview_chunks(
    source_id: str,
    content: str,
    ollama_url: str,
    model: str,
) -> list[Chunk]:
    """Generiert 5 semantische View-Chunks für ein GitHub Issue via LLM.

    Views:
      - view_fact: Fakten-Zusammenfassung (Wer, Was, Welcher Fehler)
      - view_impl: Betroffene Module, Ursache, Fix-Strategien
      - view_decision: Entscheidungen und offene Fragen
      - view_entities: Keywords und Entitäten
      - view_full: Originaltext als Fallback

    Falls der LLM-Call fehlschlägt, wird nur view_full zurückgegeben.
    """
    import json as _json_mod
    from mayring_core.providers import generate_text as _ollama_generate

    view_full = Chunk(
        chunk_id=Chunk.make_id(source_id, 0, "view_full"),
        source_id=source_id,
        chunk_level="view_full",
        ordinal=0,
        text=content,
        text_hash=Chunk.compute_text_hash(content),
    )

    if not model or not ollama_url:
        return [view_full]

    try:
        prompt = _MULTIVIEW_PROMPT_TEMPLATE.format(content=content[:3000])
        raw = _ollama_generate(
            prompt, ollama_url, model,
            label="multiview",
            system_prompt=_MULTIVIEW_SYSTEM_PROMPT,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json_mod.loads(raw)
        if not isinstance(parsed, dict):
            return [view_full]
    except Exception:
        return [view_full]

    chunks: list[Chunk] = []
    for ordinal, (level, key) in enumerate([
        ("view_fact", "fact_summary"),
        ("view_impl", "impl_summary"),
        ("view_decision", "decision_summary"),
        ("view_entities", "entities_keywords"),
    ]):
        text = coerce_str(parsed.get(key)).strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=Chunk.make_id(source_id, ordinal, level),
            source_id=source_id,
            chunk_level=level,
            ordinal=ordinal,
            text=text,
            text_hash=Chunk.compute_text_hash(text),
        ))

    chunks.append(view_full)
    return chunks
