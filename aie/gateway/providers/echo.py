"""Deterministic in-process provider for tests and offline development.

Not part of the production path -- it exists so the pipeline, guardrails and
cache can be tested without a network or a GPU.
"""

from __future__ import annotations

import asyncio

from aie.gateway.providers.base import ProviderError
from aie.types import GenerationRequest, GenerationResult


class EchoProvider:
    name = "echo"

    def __init__(
        self,
        reply: str | None = None,
        *,
        latency_ms: float = 0.0,
        fail_times: int = 0,
        retryable: bool = True,
    ) -> None:
        self._reply = reply
        self._latency_ms = latency_ms
        self._fail_times = fail_times
        self._retryable = retryable
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ProviderError("echo: induced failure", retryable=self._retryable)
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000.0)

        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        text = self._reply if self._reply is not None else f"echo: {last_user}"
        return GenerationResult(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=sum(len(m.content) // 4 for m in request.messages),
            output_tokens=len(text) // 4,
            latency_ms=self._latency_ms,
        )

    async def aclose(self) -> None:
        return None
