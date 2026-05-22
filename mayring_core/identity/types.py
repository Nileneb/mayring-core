"""Structural types shared by mayring_core without coupling to the api package.

``llm.endpoint`` and ``identity.workspace_resolver`` only read a handful of
attributes off the auth token. Depending on the concrete
``src.api.jwt_auth.TokenInfo`` dataclass would re-introduce a core→api import,
so they type against this Protocol instead. The real dataclass satisfies it
structurally — no inheritance required.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenInfoLike(Protocol):
    workspace_id: str
    scopes: tuple[str, ...]
    llm_provider: str
    llm_model: str | None
    llm_endpoint: str | None
    llm_requires_key: bool
    sub: str | None
    iat: int | None

    @property
    def uses_custom_provider(self) -> bool: ...
