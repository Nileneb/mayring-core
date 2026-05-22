"""Pre-ingest heuristic filter for conversation_summary chunks.

Conversation JSONL contains raw assistant turns including pasted code, diffs,
and tool outputs. Storing these as "knowledge" pollutes the memory store and
yields noisy LLM categorization (e.g. PHP code labeled as "ui, data_access").

This module identifies chunks that are predominantly code/tool-output rather
than reasoning, so they can be skipped during ingestion.
"""
from __future__ import annotations

import re

# Lines that strongly indicate raw code or tool output, not reasoning.
_CODE_LINE_PATTERNS = [
    re.compile(r"^\s*<\?php\b"),
    re.compile(r"^\s*(import|from)\s+\w"),
    re.compile(r"^\s*(class|def|function|public|private|protected|interface|trait)\s+\w"),
    re.compile(r"^\s*(use|namespace)\s+[\w\\]+"),
    re.compile(r"^\s*[+\-]{3}\s+[ab]/"),       # diff +++/--- a/b/
    re.compile(r"^\s*@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@"),  # diff hunk
    re.compile(r"^\s*[\d]+→"),                  # cat -n style line numbers
    re.compile(r"^\s*[\}\]\)]\s*[,;]?\s*$"),   # closing braces only
    re.compile(r"^\s*//"),
    re.compile(r"^\s*#\s*include\b"),
]

# Fenced code blocks: ```lang ... ```
_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)


def code_density(text: str) -> float:
    """Fraction of characters inside fenced code blocks or matching code-line patterns.

    Returns a value in [0.0, 1.0]. Higher = more code-like.
    """
    if not text:
        return 0.0
    total = len(text)

    fenced_chars = sum(len(m.group(0)) for m in _FENCE_RE.finditer(text))
    remainder = _FENCE_RE.sub("", text)

    code_line_chars = 0
    for line in remainder.splitlines(keepends=True):
        if any(p.search(line) for p in _CODE_LINE_PATTERNS):
            code_line_chars += len(line)

    return min(1.0, (fenced_chars + code_line_chars) / total)


def natural_language_words(text: str) -> int:
    """Count words in natural-language regions (outside code fences)."""
    if not text:
        return 0
    prose = _FENCE_RE.sub("", text)
    # Strip lines that match code patterns
    cleaned_lines = [
        ln for ln in prose.splitlines()
        if not any(p.search(ln) for p in _CODE_LINE_PATTERNS)
    ]
    cleaned = "\n".join(cleaned_lines)
    return len(re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", cleaned))


_DIFF_HUNK_RE = re.compile(r"^@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@", re.MULTILINE)
_DIFF_HEADER_RE = re.compile(r"^---\s+[ab]/.*\n\+\+\+\s+[ab]/", re.MULTILINE)


def should_skip_chunk(text: str, source_type: str) -> tuple[bool, str]:
    """Decide if a chunk should be skipped during ingestion.

    Only filters source_type=conversation_summary; other sources (repo files,
    explicit notes, papers) pass through unchanged.

    Returns (skip, reason).
    """
    if source_type != "conversation_summary":
        return False, ""

    if len(text) < 80:
        return True, "too_short"

    # Diff/patch output — categorical skip regardless of density
    if _DIFF_HUNK_RE.search(text) or _DIFF_HEADER_RE.search(text):
        return True, "diff_output"

    density = code_density(text)
    words = natural_language_words(text)

    # Mostly code AND not enough surrounding prose to be useful commentary
    if density > 0.6 and words < 40:
        return True, f"code_density={density:.2f} words={words}"

    # Almost entirely code regardless of length
    if density > 0.85:
        return True, f"code_density={density:.2f}"

    return False, ""
