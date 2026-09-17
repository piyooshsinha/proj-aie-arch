"""Online scoring: measure live traffic without becoming part of it.

Three rules this module exists to enforce:

1. **A scorer can never change a response.** Everything runs after the answer
   is final, and every scorer call is wrapped -- a scorer that throws produces
   a missing measurement, not a failed request.
2. **A scorer can never cost the user latency.** Only cheap synchronous
   scorers run inline (microseconds of string work). Model-based judges are
   dispatched as background tasks and their results land after the response
   has already gone out.
3. **Sampling is explicit.** Scoring 100% of traffic with heuristics is free
   and the default; scoring 100% with a judge is a second API bill, so the
   judge gets its own, much lower, rate.
"""

from __future__ import annotations

import asyncio
import logging
import random

from aie.eval.scorers import HEURISTIC_SCORERS, Scorer
from aie.eval.types import EvalSample, Score
from aie.observe.trace import METRICS, Trace

logger = logging.getLogger("aie.eval.online")


class OnlineScorer:
    def __init__(
        self,
        scorers: list[Scorer] | None = None,
        *,
        sample_rate: float = 1.0,
        judge: object | None = None,
        judge_sample_rate: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self._scorers = list(scorers if scorers is not None else HEURISTIC_SCORERS)
        self.sample_rate = sample_rate
        self._judge = judge
        self.judge_sample_rate = judge_sample_rate
        self._rng = rng or random.Random()
        self._background: set[asyncio.Task] = set()

    @property
    def scorer_names(self) -> list[str]:
        return [s.name for s in self._scorers]

    def observe(self, sample: EvalSample, *, trace: Trace | None = None) -> list[Score]:
        """Score a finished response. Returns [] when this request isn't sampled."""
        if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
            return []

        scores: list[Score] = []
        for scorer in self._scorers:
            try:
                score = scorer.score(sample)
            except Exception:
                logger.exception("scorer %s raised", getattr(scorer, "name", scorer))
                METRICS.incr("score.error", scorer=getattr(scorer, "name", "unknown"))
                continue

            scores.append(score)
            METRICS.incr(
                "score.result",
                scorer=score.name,
                outcome="pass" if score.passed else "fail",
                route=sample.route,
            )
            METRICS.observe(
                "score.value", score.value, buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
                scorer=score.name, route=sample.route,
            )

        if trace is not None and scores:
            with trace.span("score.online") as span:
                span.set(**{s.name: round(s.value, 3) for s in scores})
                failures = [s.name for s in scores if not s.passed]
                if failures:
                    span.set(failed=",".join(failures))

        failed = [s for s in scores if not s.passed]
        if failed:
            # One structured line per scored failure: this is the signal you
            # grep for when someone says "it got worse this week".
            logger.warning(
                "online scoring flagged a response",
                extra={
                    "failed_scorers": [s.name for s in failed],
                    "detail": {s.name: s.detail for s in failed},
                    "route": sample.route,
                },
            )

        self._maybe_dispatch_judge(sample, trace)
        return scores

    def _maybe_dispatch_judge(self, sample: EvalSample, trace: Trace | None) -> None:
        if self._judge is None or self.judge_sample_rate <= 0.0:
            return
        if self._rng.random() >= self.judge_sample_rate:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # scored outside an event loop; nothing to schedule on

        task = loop.create_task(self._run_judge(sample))
        # Hold a reference: a task with no strong reference can be GC'd mid-flight.
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _run_judge(self, sample: EvalSample) -> None:
        try:
            score = await self._judge.score_async(sample)  # type: ignore[union-attr]
        except Exception:
            logger.exception("llm judge failed")
            METRICS.incr("score.error", scorer="llm_judge")
            return
        METRICS.incr(
            "score.result",
            scorer=score.name,
            outcome="pass" if score.passed else "fail",
            route=sample.route,
        )
        METRICS.observe(
            "score.value", score.value, buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            scorer=score.name, route=sample.route,
        )
        if not score.passed:
            logger.warning(
                "judge failed a response", extra={"reason": score.detail, "route": sample.route}
            )

    async def drain(self) -> None:
        """Await in-flight judge tasks. For tests and clean shutdown."""
        if self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)
