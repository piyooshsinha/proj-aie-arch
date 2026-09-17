"""Provider contract.

Kept deliberately narrow and wire-shaped: ``generate`` takes a plain
GenerationRequest and returns a plain GenerationResult, with no provider types
leaking across the boundary. That is what makes it possible to move the whole
gateway behind an HTTP hop (or into another language) later without touching a
single caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aie.types import GenerationRequest, GenerationResult


class ProviderError(RuntimeError):
    """Base for provider failures.

    ``retryable`` drives the gateway's backoff loop; ``fatal`` means do not even
    try the fallback model (bad request, refusal) because the next model will
    fail identically.
    """

    def __init__(self, message: str, *, retryable: bool = False, fatal: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.fatal = fatal


class ProviderRefusal(ProviderError):
    """The model declined the request. Not retryable, not a fallback candidate."""

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message, retryable=False, fatal=True)
        self.category = category


@runtime_checkable
class Provider(Protocol):
    name: str

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    async def aclose(self) -> None: ...
