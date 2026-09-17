"""Guardrail contract and the chain that runs them.

Two decisions the diagram leaves implicit, made explicit here:

* **Sync or async.** A guardrail that calls a model doubles your p50. Each one
  declares ``blocking``; non-blocking guardrails are still evaluated but can
  never hold the response (use them to gather data on a new check before you
  let it gate traffic).
* **Fail-open or fail-closed.** A guardrail that throws must not take the
  request with it, but "let it through" is not always right either. Each one
  declares ``fail_closed``; a safety check should, a PII redactor probably
  should, a telemetry-ish check should not.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from aie.observe.trace import METRICS, Trace
from aie.types import GuardrailAction, GuardrailResult

logger = logging.getLogger("aie.guardrails")


@runtime_checkable
class Guardrail(Protocol):
    name: str
    blocking: bool
    fail_closed: bool

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult: ...


class GuardrailChain:
    """Runs guardrails in order, threading modified content through the chain."""

    def __init__(self, guardrails: list[Guardrail], *, stage: str) -> None:
        self._guardrails = guardrails
        self._stage = stage

    def run(
        self, content: str, *, trace: Trace | None = None, context: dict | None = None
    ) -> tuple[str, list[GuardrailResult]]:
        results: list[GuardrailResult] = []
        current = content

        for guardrail in self._guardrails:
            try:
                result = guardrail.check(current, context=context)
            except Exception as exc:  # a broken check must not 500 the request
                logger.exception("guardrail %s raised", guardrail.name)
                METRICS.incr("guardrail.error", stage=self._stage, name=guardrail.name)
                result = GuardrailResult(
                    name=guardrail.name,
                    action=(
                        GuardrailAction.BLOCK if guardrail.fail_closed else GuardrailAction.ALLOW
                    ),
                    content=current,
                    message=f"guardrail failed: {type(exc).__name__}",
                )

            if not guardrail.blocking and result.action in (
                GuardrailAction.BLOCK,
                GuardrailAction.RETRY,
            ):
                # Shadow mode: record what it would have done, let traffic pass.
                METRICS.incr("guardrail.shadow_trip", stage=self._stage, name=guardrail.name)
                result = GuardrailResult(
                    name=result.name,
                    action=GuardrailAction.ALLOW,
                    content=current,
                    findings=result.findings,
                    message=f"[shadow] would have {result.action.value}: {result.message}",
                )

            results.append(result)
            METRICS.incr(
                "guardrail.action", stage=self._stage, name=guardrail.name, action=result.action.value
            )
            if trace is not None:
                with trace.span(
                    f"guardrail.{self._stage}.{guardrail.name}",
                    action=result.action.value,
                    findings=len(result.findings),
                ):
                    pass

            current = result.content
            if result.action in (GuardrailAction.BLOCK, GuardrailAction.RETRY):
                break

        return current, results


def first_stop(results: list[GuardrailResult]) -> GuardrailResult | None:
    """The result that halted the chain, if any."""
    for result in results:
        if result.action in (GuardrailAction.BLOCK, GuardrailAction.RETRY):
            return result
    return None
