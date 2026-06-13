"""Embedding agreement check for the distributed embedding pool (#365 Schicht 3).

Two devices embed the same text; bge-m3 is NOT bit-identical across GPUs/drivers
(FP non-determinism), so a hash compare would quarantine honest pairs. We compare
by cosine similarity against a high threshold instead — FP noise stays above it,
real divergence (wrong model/text/tampering) falls below.
"""
from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 on length mismatch or a zero vector
    (treated as divergence by verify(), never a crash)."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def verify(a: list[float], b: list[float], *, threshold: float) -> bool:
    """True iff the two vectors agree at or above `threshold`."""
    return cosine(a, b) >= threshold


__all__ = ("cosine", "verify")
