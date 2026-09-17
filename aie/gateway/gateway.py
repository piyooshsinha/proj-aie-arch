"""The model gateway.

Every model call in the platform goes through here. That single choke point is
what makes the rest tractable: credentials, routing, retries, fallback, cost
accounting and per-call tracing all have exactly one implementation.

Retry policy lives here rather than in the provider SDKs (which are configured
with max_retries=0) so that one trace shows every attempt, and so a retry
budget can be enforced across providers instead of per-provider.
"""

from __future__ import annotations

import asyncio
import logging
import random

from aie.gateway.catalog import ModelCatalog, ModelSpec
from aie.gateway.providers.base import Provider, ProviderError, ProviderRefusal
from aie.observe.metrics import COST_BUCKETS_USD, TOKEN_BUCKETS
from aie.observe.trace import METRICS, Trace
from aie.types import GenerationRequest, GenerationResult, Message

logger = logging.getLogger("aie.gateway")


class GatewayError(RuntimeError):
    pass


class ModelGateway:
    def __init__(
        self,
        catalog: ModelCatalog,
        providers: dict[str, Provider],
        *,
        max_attempts_per_model: int = 2,
        base_backoff_s: float = 0.25,
        max_backoff_s: float = 8.0,
    ) -> None:
        self._catalog = catalog
        self._providers = providers
        self._max_attempts = max_attempts_per_model
        self._base_backoff_s = base_backoff_s
        self._max_backoff_s = max_backoff_s

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    async def generate(
        self,
        messages: list[Message],
        *,
        route: str = "default",
        trace: Trace | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        response_format: dict | None = None,
        timeout_s: float = 60.0,
    ) -> GenerationResult:
        resolved = self._catalog.resolve(route)
        candidates = resolved.candidates()
        errors: list[str] = []

        for position, spec in enumerate(candidates):
            provider = self._providers.get(spec.provider)
            if provider is None:
                errors.append(f"{spec.provider}: no provider registered")
                continue

            request = GenerationRequest(
                messages=messages,
                model=spec.model_id,
                max_tokens=max_tokens or spec.max_output_tokens,
                temperature=temperature,
                response_format=response_format,
                timeout_s=timeout_s,
            )

            try:
                result = await self._call_with_retries(provider, spec, request, trace, position)
            except ProviderRefusal as exc:
                # A refusal is a decision, not an outage. Trying a different
                # model to get a different answer is exactly the wrong move.
                METRICS.incr("gateway.refusal", provider=spec.provider)
                raise GatewayError(str(exc)) from exc
            except ProviderError as exc:
                errors.append(f"{spec.provider}:{spec.model_id}: {exc}")
                if exc.fatal:
                    raise GatewayError(
                        f"route {route!r} failed fatally on {spec.model_id}: {exc}"
                    ) from exc
                METRICS.incr("gateway.failover", **{"from": spec.model_id})
                logger.warning("gateway: %s exhausted, falling back: %s", spec.model_id, exc)
                continue

            result.cost_usd = spec.cost(result.input_tokens, result.output_tokens)
            if trace is not None:
                trace.add_cost(result.cost_usd)

            labels = {"model": spec.model_id, "provider": spec.provider, "route": route}
            METRICS.incr("gateway.tokens.input", result.input_tokens, model=spec.model_id)
            METRICS.incr("gateway.tokens.output", result.output_tokens, model=spec.model_id)
            METRICS.incr("gateway.cost_usd", result.cost_usd, model=spec.model_id)
            # Histograms, because "what is p95 generation latency" is not a
            # question a counter can answer.
            METRICS.observe("gateway.latency.seconds", result.latency_ms / 1000.0, **labels)
            METRICS.observe(
                "gateway.cost.usd", result.cost_usd, buckets=COST_BUCKETS_USD, **labels
            )
            METRICS.observe(
                "gateway.tokens.output.count",
                result.output_tokens,
                buckets=TOKEN_BUCKETS,
                **labels,
            )
            if position > 0:
                METRICS.incr("gateway.served_by_fallback", model=spec.model_id)
            return result

        raise GatewayError(
            f"route {route!r}: every candidate failed -> " + "; ".join(errors)
        )

    async def _call_with_retries(
        self,
        provider: Provider,
        spec: ModelSpec,
        request: GenerationRequest,
        trace: Trace | None,
        position: int,
    ) -> GenerationResult:
        last: ProviderError | None = None

        for attempt in range(1, self._max_attempts + 1):
            span_ctx = (
                trace.span(
                    "gateway.generate",
                    provider=spec.provider,
                    model=spec.model_id,
                    attempt=attempt,
                    fallback_position=position,
                )
                if trace is not None
                else _null_span()
            )
            with span_ctx as span:
                try:
                    result = await provider.generate(request)
                except ProviderError as exc:
                    last = exc
                    if span is not None:
                        span.set(outcome="error", retryable=exc.retryable)
                    if not exc.retryable or attempt == self._max_attempts:
                        raise
                else:
                    if span is not None:
                        span.set(
                            outcome="ok",
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            finish_reason=result.finish_reason,
                        )
                    return result

            # Exponential backoff with full jitter, outside the span so the
            # wait is not attributed to the provider's latency.
            delay = min(self._base_backoff_s * 2 ** (attempt - 1), self._max_backoff_s)
            await asyncio.sleep(random.uniform(0, delay))

        assert last is not None
        raise last

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


class _null_span:
    """Context manager used when no trace is supplied."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False
