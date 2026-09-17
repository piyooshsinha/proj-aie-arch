"""Scorers.

Heuristic scorers first, and they are not a placeholder for "real" model-based
scoring. They are free, deterministic, and run on every request, which means
they catch a regression the same afternoon it ships. A model judge costs money
per call and disagrees with itself between runs, so it earns its place on a
sample of traffic and in offline evals -- not on the hot path.

Every scorer returns a normalized 0..1 value plus a pass/fail against its own
threshold, so a mixed set of scorers aggregates into one report.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from aie.context.retrieval import tokenize
from aie.eval.types import EvalSample, Score


NON_ANSWER_MARKERS = (
    "i don't know", "i do not know", "i cannot", "i can't help", "i can't",
    "no information", "not contain the answer", "unable to", "don't have",
    "do not have", "not covered", "no relevant",
)


def is_non_answer(text: str) -> bool:
    """Does the response decline rather than assert?

    Load-bearing: with no retrieved context, declining is the *correct*
    behaviour and asserting is a hallucination. Several scorers below invert
    their verdict on this.
    """
    lowered = text.strip().lower()
    return not lowered or any(marker in lowered for marker in NON_ANSWER_MARKERS)


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, sample: EvalSample) -> Score: ...


class Groundedness:
    """Fraction of the answer's content words that appear in retrieved context.

    A lexical proxy for "did the model make this up". It is not a hallucination
    detector -- a fluent paraphrase scores lower than a copy-paste, and a
    confident fabrication that reuses context vocabulary scores high. What it
    reliably catches is the answer that shares almost nothing with what was
    retrieved, which is the failure worth paging on.
    """

    name = "groundedness"

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold

    def score(self, sample: EvalSample) -> Score:
        if not sample.chunks:
            # The interesting case, and the one an earlier version of this
            # scorer waved through: nothing was retrieved, so an answer that
            # asserts anything is asserting it from thin air.
            if is_non_answer(sample.response):
                return Score(self.name, 1.0, True, "no context, and the answer declines")
            return Score(
                self.name, 0.0, False,
                "answered substantively with no retrieved context to support it",
            )

        answer_terms = set(tokenize(sample.response))
        if not answer_terms:
            return Score(self.name, 0.0, False, "empty answer")

        context_terms = set()
        for chunk in sample.chunks:
            context_terms |= set(tokenize(chunk.text))
        context_terms |= set(tokenize(sample.query))

        overlap = len(answer_terms & context_terms) / len(answer_terms)
        return Score(
            self.name,
            overlap,
            overlap >= self.threshold,
            f"{len(answer_terms & context_terms)}/{len(answer_terms)} answer terms in context",
        )


class CitationPresence:
    """Did the answer cite a passage it was actually given?

    Catches two distinct failures: no citation at all, and a citation to a
    passage id that was never retrieved (a fabricated source, which is worse
    than none).
    """

    name = "citation"

    _CITATION = re.compile(r"[\[\(]([a-zA-Z0-9_.#/-]+)[\]\)]")

    def score(self, sample: EvalSample) -> Score:
        cited = {m.group(1) for m in self._CITATION.finditer(sample.response)}

        if not sample.chunks:
            if cited:
                return Score(
                    self.name, 0.0, False,
                    f"cites {sorted(cited)} but nothing was retrieved",
                )
            return Score(self.name, 1.0, True, "no context, no citation expected")

        if not cited:
            return Score(self.name, 0.0, False, "answer cites no passage")

        ids = {c.id for c in sample.chunks}
        known = ids | {f"{c.source}#{c.id}" for c in sample.chunks}
        # Exact match, or an exact match on the id after a `source#` prefix.
        # Substring matching would accept "[d]" as a citation of passage "d1".
        valid = {c for c in cited if c in known or c.rsplit("#", 1)[-1] in ids}
        if not valid:
            return Score(
                self.name, 0.0, False, f"cites unknown passage(s): {sorted(cited)}"
            )
        return Score(self.name, 1.0, True, f"cites {sorted(valid)}")


class LatencySLO:
    name = "latency_slo"

    def __init__(self, max_ms: float = 5000.0) -> None:
        self.max_ms = max_ms

    def score(self, sample: EvalSample) -> Score:
        # Normalized so 0 latency is 1.0 and the SLO boundary is 0.0.
        value = max(0.0, 1.0 - sample.latency_ms / self.max_ms)
        return Score(
            self.name,
            value,
            sample.latency_ms <= self.max_ms,
            f"{sample.latency_ms:.0f}ms against a {self.max_ms:.0f}ms budget",
        )


class CostBudget:
    name = "cost_budget"

    def __init__(self, max_usd: float = 0.05) -> None:
        self.max_usd = max_usd

    def score(self, sample: EvalSample) -> Score:
        value = max(0.0, 1.0 - sample.cost_usd / self.max_usd) if self.max_usd else 1.0
        return Score(
            self.name,
            value,
            sample.cost_usd <= self.max_usd,
            f"${sample.cost_usd:.4f} against a ${self.max_usd:.4f} budget",
        )


class AnswerPresent:
    """Catches refusals and non-answers that still returned 200."""

    name = "answer_present"

    def score(self, sample: EvalSample) -> Score:
        if not sample.response.strip():
            return Score(self.name, 0.0, False, "empty")
        if sample.blocked:
            return Score(self.name, 0.0, False, "blocked by a guardrail")
        if is_non_answer(sample.response):
            # Declining is correct when nothing was retrieved and a miss when
            # the answer was sitting in the context all along.
            if sample.chunks:
                return Score(
                    self.name, 0.0, False,
                    f"declined despite {len(sample.chunks)} retrieved passage(s)",
                )
            return Score(self.name, 1.0, True, "correctly declined, nothing retrieved")
        return Score(self.name, 1.0, True, "")


class KeywordRecall:
    """Offline: fraction of expected keywords present in the answer."""

    name = "keyword_recall"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def score(self, sample: EvalSample) -> Score:
        if not sample.expected_keywords:
            return Score(self.name, 1.0, True, "no expected keywords")
        text = sample.response.lower()
        hits = [k for k in sample.expected_keywords if k.lower() in text]
        value = len(hits) / len(sample.expected_keywords)
        missing = sorted(set(sample.expected_keywords) - set(hits))
        return Score(
            self.name,
            value,
            value >= self.threshold,
            f"missing {missing}" if missing else "all present",
        )


HEURISTIC_SCORERS: list[Scorer] = [
    Groundedness(),
    CitationPresence(),
    AnswerPresent(),
    LatencySLO(),
    CostBudget(),
]


# --- model-based judge -----------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "score", "reason"],
    "properties": {
        "verdict": {"type": "string"},
        "score": {"type": "integer"},
        "reason": {"type": "string"},
    },
}

JUDGE_PROMPT = """You are grading one answer from a retrieval system.

<question>{query}</question>
<retrieved_context>{context}</retrieved_context>
<answer>{answer}</answer>

Grade the answer 1-5 on whether it is supported by the retrieved context and
actually answers the question. An answer that correctly says the context does
not cover the question scores 5. An answer containing claims absent from the
context scores 1.

Respond with JSON only: {{"verdict": "pass"|"fail", "score": 1-5, "reason": "one sentence"}}"""


class LLMJudge:
    """Model-based scoring -- the diagram's "Scoring", made explicit.

    Async and off the hot path by construction: it costs a generation per call
    and is non-deterministic between runs, so it belongs on a sample of traffic
    and in offline evals. Defaults to the ``cheap`` route rather than whatever
    served the request, because a judge does not need the strongest model and
    grading every request with one doubles your bill.
    """

    name = "llm_judge"

    def __init__(self, gateway, *, route: str = "cheap", pass_at: int = 4) -> None:
        self._gateway = gateway
        self.route = route
        self.pass_at = pass_at

    async def score_async(self, sample: EvalSample) -> Score:
        from aie.types import Message

        context = "\n".join(f"[{c.source}#{c.id}] {c.text}" for c in sample.chunks)
        prompt = JUDGE_PROMPT.format(
            query=sample.query, context=context or "(none)", answer=sample.response
        )
        result = await self._gateway.generate(
            [Message("user", prompt)],
            route=self.route,
            response_format=JUDGE_SCHEMA,
            max_tokens=512,
        )
        try:
            parsed = json.loads(result.text.strip().removeprefix("```json").removesuffix("```").strip())
            rating = int(parsed["score"])
        except (ValueError, KeyError, TypeError) as exc:
            # A judge that cannot be parsed is a missing measurement, never a
            # failing one -- do not let it manufacture regressions.
            return Score(self.name, 0.0, True, f"unparseable judge output: {exc}")

        return Score(
            self.name,
            max(0.0, min(1.0, (rating - 1) / 4)),
            rating >= self.pass_at,
            str(parsed.get("reason", ""))[:200],
        )
