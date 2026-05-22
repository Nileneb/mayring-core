"""Language-aware structural chunking for memory ingestion."""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from mayring_core.memory.schema import Chunk


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_file_chunk(text: str, source_id: str, ordinal: int = 0) -> Chunk:
    """Single file-level fallback chunk."""
    text_hash = Chunk.compute_text_hash(text)
    return Chunk(
        chunk_id=Chunk.make_id(source_id, ordinal, "file"),
        source_id=source_id,
        chunk_level="file",
        ordinal=ordinal,
        start_offset=0,
        end_offset=len(text),
        text=text,
        text_hash=text_hash,
        dedup_key=text_hash,
        created_at=_now_iso(),
    )


def _chunk_python(text: str, source_id: str) -> list[Chunk]:
    """AST-based chunking for Python: top-level functions and classes."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    # Build cumulative char offsets per line (0-indexed)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    chunks: list[Chunk] = []
    ordinal = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Only top-level nodes (direct children of module)
        if node not in tree.body:  # type: ignore[attr-defined]
            continue

        level = "class" if isinstance(node, ast.ClassDef) else "function"
        start_line = node.lineno - 1  # 0-indexed
        end_line = getattr(node, "end_lineno", start_line) - 1  # 0-indexed
        start_off = line_offsets[start_line] if start_line < len(line_offsets) else 0
        end_off = (
            line_offsets[end_line] + len(lines[end_line])
            if end_line < len(lines)
            else len(text)
        )
        chunk_text = text[start_off:end_off]
        if not chunk_text.strip():
            continue

        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, level),
                source_id=source_id,
                chunk_level=level,
                ordinal=ordinal,
                start_offset=start_off,
                end_offset=end_off,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def _chunk_js(text: str, source_id: str) -> list[Chunk]:
    """Regex + brace-depth chunking for JS/TS: functions and classes."""
    # Match: function X(, async function X(, class X {, const X = (async)? (, export variants
    pattern = re.compile(
        r"(?:^|\n)((?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+\w+"
        r"|(?:export\s+)?class\s+\w+"
        r"|(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\()",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for i, m in enumerate(matches):
        start = m.start() if text[m.start()] != "\n" else m.start() + 1
        # Find end by counting braces from the match start
        brace_depth = 0
        end = len(text)
        found_open = False
        for j in range(start, len(text)):
            if text[j] == "{":
                brace_depth += 1
                found_open = True
            elif text[j] == "}":
                brace_depth -= 1
                if found_open and brace_depth == 0:
                    end = j + 1
                    break

        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue

        header = m.group(1)
        level = "class" if "class" in header else "function"
        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, level),
                source_id=source_id,
                chunk_level=level,
                ordinal=ordinal,
                start_offset=start,
                end_offset=end,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def _chunk_markdown(text: str, source_id: str) -> list[Chunk]:
    """Split Markdown on headings (# / ## / ###)."""
    heading_re = re.compile(r"^#{1,3}\s+[^\n]+", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, "section"),
                source_id=source_id,
                chunk_level="section",
                ordinal=ordinal,
                start_offset=start,
                end_offset=end,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def _chunk_yaml_json(text: str, source_id: str, filename: str) -> list[Chunk]:
    """Chunk YAML/JSON by top-level keys."""
    _MAX_CHUNK_CHARS = 2000
    try:
        if filename.endswith(".json"):
            data = json.loads(text)
        elif _HAS_YAML:
            data = _yaml.safe_load(text)
        else:
            return []
    except Exception:
        return []

    if not isinstance(data, dict) or not data:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for key, value in data.items():
        chunk_text = json.dumps({key: value}, ensure_ascii=False)[:_MAX_CHUNK_CHARS]
        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, "block"),
                source_id=source_id,
                chunk_level="block",
                ordinal=ordinal,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def structural_chunk(text: str, source_id: str, filename: str) -> list[Chunk]:
    """Dispatch to language-specific chunker based on file extension.

    Falls back to a single file-level chunk on parse failure or unknown extension.
    """
    if not text.strip():
        return [_make_file_chunk(text, source_id)]

    ext = Path(filename).suffix.lower()

    if ext == ".py":
        chunks = _chunk_python(text, source_id)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        chunks = _chunk_js(text, source_id)
    elif ext in (".md", ".markdown"):
        chunks = _chunk_markdown(text, source_id)
    elif ext in (".yaml", ".yml", ".json"):
        chunks = _chunk_yaml_json(text, source_id, filename)
    else:
        chunks = []

    return chunks if chunks else [_make_file_chunk(text, source_id)]


_SECTION_RE = re.compile(
    r'^(?:\d+\.?\s+)?(?:Abstract|Introduction|Related Work|Background|'
    r'Methodology|Methods?|Experiments?|Results?|Discussion|'
    r'Conclusion|References?|Appendix)\b',
    re.IGNORECASE | re.MULTILINE,
)


def chunk_paper(text: str, source_id: str) -> list[Chunk]:
    """Chunk paper text into section-level chunks.

    Splits on standard academic section headers. Falls back to a single
    file-level chunk if no sections are detected.
    """
    if not text.strip():
        return [_make_file_chunk(text, source_id)]

    splits = list(_SECTION_RE.finditer(text))
    if not splits:
        return [_make_file_chunk(text, source_id)]

    chunks: list[Chunk] = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        section_name = match.group(0).strip().lower().split()[0]
        text_hash = Chunk.compute_text_hash(section_text)
        chunks.append(Chunk(
            chunk_id=Chunk.make_id(source_id, i, section_name),
            source_id=source_id,
            chunk_level=section_name,
            ordinal=i,
            start_offset=start,
            end_offset=end,
            text=section_text,
            text_hash=text_hash,
            dedup_key=text_hash,
            created_at=_now_iso(),
        ))

    return chunks if chunks else [_make_file_chunk(text, source_id)]


def extract_pdf_text(pdf_path: str) -> str | None:
    """Extract text from a PDF file via pypdf. Returns None if pypdf not installed."""
    try:
        import pypdf  # type: ignore
        import io as _io
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(_io.BytesIO(f.read()))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip()) or None
    except ImportError:
        return None
    except Exception:
        return None
