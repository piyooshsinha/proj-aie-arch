"""The request path, in the order the architecture draws it.

    query -> response cache -> context construction -> input guardrails
          -> model gateway -> output guardrails -> response
                                    |
                                    +-- RETRY --> back to context construction

Two things worth reading closely:

*Cache first.* The lookup happens before context construction so a hit costs
one hash, not a retrieval round trip. The key is user-scoped (see
``aie.cache.response``) so a hit cannot cross a permission boundary.

*The loop is bounded and it feeds back real information.* An output guardrail
returning RETRY appends its message as a repair note and re-enters context
construction. ``max_iterations`` caps it, because an unbounded repair loop is
how a 2-second request becomes a 40-second one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aie.cache.response import ResponseCache
from aie.context.construction import ContextConstructor
from aie.gateway.gateway import GatewayError, ModelGateway
from aie.guardrails.base import GuardrailChain, first_stop
from aie.observe.trace import METRICS, Trace
from aie.store.memory import ChatHistoryStore
from aie.types import (
    GuardrailAction,
    GuardrailResult,
    Message,
    PipelineResponse,
    Query,
)

logger = logging.getLogger("aie.pipeline")


@dataclass
class PipelineConfig:
    max_iterations: int = 2
    cache_responses: bool = True
    persist_history: bool = True
    max_tokens: int | None = None
    timeout_s: float = 60.0
    blocked_message: str = (
        "I can't help with that request."
    )


class Pipeline:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        context: ContextConstructor,
        input_guardrails: GuardrailChain,
        output_guardrails_for,
        cache: ResponseCache,
        history: ChatHistoryStore,
        config: PipelineConfig | None = None,
    ) -> None:
        self._gateway = gateway
        self._context = context
        self._input_guardrails = input_guardrails
        # A callable, not a chain: the output chain depends on the request
        # (a query asking for JSON needs the schema validator bound to it).
        self._output_guardrails_for = output_guardrails_for
        self._cache = cache
        self._history = history
        self.config = config or PipelineConfig()

    async def run(self, query: Query, *, trace: Trace | None = None) -> PipelineResponse:
        trace = trace or Trace(
            tenant_id=query.tenant_id, user_id=query.user_id, route=query.route
        )
        guardrail_results: list[GuardrailResult] = []

        try:
            # 1. Response cache, before anything expensive.
            with trace.span("cache.lookup", scope=self._cache.scope) as span:
                entry = self._cache.get(query) if self.config.cache_responses else None
                span.set(hit=entry is not None)

            if entry is not None:
                METRICS.incr("pipeline.cache_hit", route=query.route)
                trace.finish()
                return PipelineResponse(
                    text=entry.value,
                    trace_id=trace.trace_id,
                    cached=True,
                    model=entry.model,
                    latency_ms=trace.latency_ms,
                )

            repair_notes: list[str] = []
            last_text = ""
            last_model: str | None = None
            last_provider: str | None = None

            for iteration in range(1, self.config.max_iterations + 1):
                # 2. Context construction (retrieval, rewriting, assembly).
                context = self._context.build(query, trace=trace, repair_notes=repair_notes)
                messages = context.to_messages()

                # 3. Input guardrails, on the assembled user turn.
                *prefix, user_turn = messages
                guarded_text, input_results = self._input_guardrails.run(
                    user_turn.content,
                    trace=trace,
                    context={"query": query, "iteration": iteration},
                )
                guardrail_results.extend(input_results)

                stop = first_stop(input_results)
                if stop is not None:
                    METRICS.incr("pipeline.blocked", stage="input", name=stop.name)
                    return self._blocked(trace, guardrail_results, iteration, stop)

                messages = [*prefix, Message("user", guarded_text)]

                # 4. Model gateway.
                result = await self._gateway.generate(
                    messages,
                    route=query.route,
                    trace=trace,
                    max_tokens=self.config.max_tokens,
                    response_format=query.response_schema,
                    timeout_s=self.config.timeout_s,
                )
                last_text, last_model, last_provider = (
                    result.text,
                    result.model,
                    result.provider,
                )

                # 5. Output guardrails.
                output_chain = self._output_guardrails_for(query)
                final_text, output_results = output_chain.run(
                    result.text,
                    trace=trace,
                    context={
                        "query": query,
                        "iteration": iteration,
                        "input_text": query.text,
                    },
                )
                guardrail_results.extend(output_results)

                stop = first_stop(output_results)
                if stop is None:
                    return self._succeed(
                        query, final_text, result, trace, guardrail_results, iteration
                    )

                if stop.action is GuardrailAction.RETRY and iteration < self.config.max_iterations:
                    # 6. The feedback arrow: hand the failure back to context
                    #    construction as a correction note and go around again.
                    METRICS.incr("pipeline.repair", name=stop.name)
                    repair_notes.append(stop.message or "The previous response was rejected.")
                    continue

                METRICS.incr("pipeline.blocked", stage="output", name=stop.name)
                return self._blocked(trace, guardrail_results, iteration, stop)

            # Loop exhausted without a clean pass.
            METRICS.incr("pipeline.repair_exhausted", route=query.route)
            return PipelineResponse(
                text=last_text or self.config.blocked_message,
                trace_id=trace.trace_id,
                model=last_model,
                provider=last_provider,
                blocked=not last_text,
                guardrails=guardrail_results,
                iterations=self.config.max_iterations,
                cost_usd=trace.cost_usd,
                latency_ms=trace.latency_ms,
            )

        except GatewayError as exc:
            logger.error("pipeline: gateway failed: %s", exc)
            METRICS.incr("pipeline.gateway_error", route=query.route)
            raise
        finally:
            trace.emit()

    def _succeed(
        self, query, text, result, trace, guardrail_results, iteration
    ) -> PipelineResponse:
        if self.config.cache_responses:
            self._cache.set(query, text, model=result.model)
        if self.config.persist_history:
            self._history.append(
                query.tenant_id,
                query.session_id,
                Message("user", query.text),
                Message("assistant", text),
            )
        trace.finish()
        METRICS.incr("pipeline.ok", route=query.route)
        return PipelineResponse(
            text=text,
            trace_id=trace.trace_id,
            model=result.model,
            provider=result.provider,
            guardrails=guardrail_results,
            iterations=iteration,
            cost_usd=trace.cost_usd,
            latency_ms=trace.latency_ms,
        )

    def _blocked(self, trace, guardrail_results, iteration, stop) -> PipelineResponse:
        trace.finish()
        # A blocked response is never cached and never written to history:
        # caching a refusal makes it permanent for that user.
        return PipelineResponse(
            text=stop.content if stop.name == "pii_leak" else self.config.blocked_message,
            trace_id=trace.trace_id,
            blocked=True,
            guardrails=guardrail_results,
            iterations=iteration,
            cost_usd=trace.cost_usd,
            latency_ms=trace.latency_ms,
        )
