"""Scoring: the scorers themselves, online sampling, and the offline harness."""

from __future__ import annotations

import asyncio
import random

import pytest

from aie.eval.online import OnlineScorer
from aie.eval.runner import EvalCase, run_eval
from aie.eval.scorers import (
    HEURISTIC_SCORERS,
    AnswerPresent,
    CitationPresence,
    CostBudget,
    Groundedness,
    KeywordRecall,
    LatencySLO,
    is_non_answer,
)
from aie.eval.types import EvalSample
from aie.observe.trace import METRICS, Trace
from aie.types import Query, RetrievedChunk
from tests.conftest import ScriptedProvider, make_platform

CHUNKS = [
    RetrievedChunk("d1", "Refunds are issued within 14 days for EU orders.", 1.0, "policy")
]


def sample(response: str, **kwargs) -> EvalSample:
    kwargs.setdefault("chunks", CHUNKS)
    return EvalSample(query="refund policy for EU orders", response=response, **kwargs)


# --- scorers ---------------------------------------------------------------


def test_groundedness_separates_a_grounded_answer_from_a_fabricated_one():
    grounded = Groundedness().score(
        sample("Refunds for EU orders are issued within 14 days.")
    )
    fabricated = Groundedness().score(sample("Our llamas dance on Thursdays."))

    assert grounded.passed and grounded.value > 0.8
    assert not fabricated.passed and fabricated.value < 0.2


def test_answering_with_no_retrieved_context_is_a_failure():
    """The regression this suite exists for.

    An earlier version passed any answer when retrieval returned nothing, which
    waves through exactly the confident-fabrication case.
    """
    asserted = Groundedness().score(
        EvalSample(query="airspeed of a swallow", response="About 24 mph.", chunks=[])
    )
    declined = Groundedness().score(
        EvalSample(
            query="airspeed of a swallow",
            response="I don't have information about that.",
            chunks=[],
        )
    )

    assert not asserted.passed, "asserting with zero context must not pass"
    assert declined.passed, "declining with zero context is correct behaviour"


def test_a_citation_to_a_passage_that_was_never_retrieved_fails():
    assert CitationPresence().score(sample("See [policy#d1].")).passed
    assert not CitationPresence().score(sample("See [policy#d99].")).passed
    assert not CitationPresence().score(sample("No citation here.")).passed
    # A citation with nothing retrieved is necessarily invented.
    assert not CitationPresence().score(
        EvalSample(query="q", response="See [ornithology#d99].", chunks=[])
    ).passed


def test_citation_matching_is_exact_not_substring():
    # "[d]" is not a citation of passage "d1".
    assert not CitationPresence().score(sample("See [d].")).passed
    assert CitationPresence().score(sample("See [d1].")).passed


def test_declining_is_scored_against_whether_context_was_available():
    had_context = AnswerPresent().score(sample("I don't know."))
    no_context = AnswerPresent().score(
        EvalSample(query="q", response="I don't know.", chunks=[])
    )

    assert not had_context.passed, "declining with the answer in context is a miss"
    assert no_context.passed


def test_blocked_and_empty_responses_fail():
    assert not AnswerPresent().score(sample("", blocked=False)).passed
    assert not AnswerPresent().score(sample("an answer", blocked=True)).passed


def test_budget_scorers_normalize_against_their_threshold():
    assert LatencySLO(max_ms=1000).score(sample("x", latency_ms=500)).value == 0.5
    assert not LatencySLO(max_ms=1000).score(sample("x", latency_ms=1500)).passed
    assert not CostBudget(max_usd=0.01).score(sample("x", cost_usd=0.5)).passed


def test_keyword_recall_reports_what_is_missing():
    score = KeywordRecall().score(
        EvalSample(query="q", response="14 days", expected_keywords=["14 days", "EU"])
    )
    assert score.value == 0.5
    assert "EU" in score.detail


def test_non_answer_detection():
    assert is_non_answer("I don't know.")
    assert is_non_answer("   ")
    assert not is_non_answer("Refunds take 14 days.")


# --- online scoring --------------------------------------------------------


def test_online_scoring_records_metrics_and_annotates_the_trace():
    trace = Trace()
    scorer = OnlineScorer(HEURISTIC_SCORERS)
    scores = scorer.observe(sample("Refunds are issued within 14 days [policy#d1]."), trace=trace)

    assert {s.name for s in scores} == {s.name for s in HEURISTIC_SCORERS}
    counters = METRICS.snapshot()
    assert any(k.startswith("score.result{") and "groundedness" in k for k in counters)
    span = next(s for s in trace.root.walk() if s.name == "score.online")
    assert "groundedness" in span.attributes


def test_sampling_skips_unsampled_requests():
    never = OnlineScorer(HEURISTIC_SCORERS, sample_rate=0.0, rng=random.Random(0))
    always = OnlineScorer(HEURISTIC_SCORERS, sample_rate=1.0, rng=random.Random(0))

    assert never.observe(sample("anything")) == []
    assert always.observe(sample("anything")) != []


def test_a_broken_scorer_cannot_break_scoring():
    class Exploding:
        name = "exploding"

        def score(self, sample):
            raise RuntimeError("scorer bug")

    scorer = OnlineScorer([Exploding(), Groundedness()])
    scores = scorer.observe(sample("Refunds take 14 days."))

    assert [s.name for s in scores] == ["groundedness"]
    assert any("score.error" in k for k in METRICS.snapshot())


async def test_scoring_never_changes_the_response(settings):
    class Hostile:
        """A scorer that tries its best to interfere."""

        name = "hostile"

        def score(self, sample):
            sample.response = "TAMPERED"
            raise RuntimeError("and then it raises")

    platform = make_platform(settings, ScriptedProvider(["the real answer"]))
    platform.pipeline._scorer = OnlineScorer([Hostile()])

    result = await platform.pipeline.run(
        Query(text="refund policy", user_id="u", tenant_id="acme")
    )
    assert result.text == "the real answer"


async def test_the_judge_runs_in_the_background_and_off_the_hot_path():
    started = asyncio.Event()

    class SlowJudge:
        name = "llm_judge"
        calls = 0

        async def score_async(self, sample):
            from aie.eval.types import Score

            SlowJudge.calls += 1
            started.set()
            await asyncio.sleep(0)
            return Score("llm_judge", 1.0, True, "fine")

    scorer = OnlineScorer(
        [Groundedness()], judge=SlowJudge(), judge_sample_rate=1.0, rng=random.Random(0)
    )
    scores = scorer.observe(sample("Refunds take 14 days."))

    assert [s.name for s in scores] == ["groundedness"], "the judge is not inline"
    await scorer.drain()
    assert SlowJudge.calls == 1


def test_judge_dispatch_outside_an_event_loop_is_a_no_op():
    class Judge:
        name = "llm_judge"

        async def score_async(self, sample):  # pragma: no cover - never called
            raise AssertionError("should not run")

    scorer = OnlineScorer([Groundedness()], judge=Judge(), judge_sample_rate=1.0)
    assert scorer.observe(sample("Refunds take 14 days.")) != []


# --- offline harness -------------------------------------------------------


async def test_eval_runner_aggregates_pass_rates_and_failures(settings):
    platform = make_platform(
        settings,
        ScriptedProvider(
            [
                "Refunds are issued within 14 days for EU orders [policy#d1].",
                "Our llamas dance on Thursdays.",
            ]
        ),
    )
    cases = [
        EvalCase(query="refund policy for EU orders", id="good"),
        EvalCase(query="refund policy for EU orders", id="bad"),
    ]

    report = await run_eval(platform, cases, concurrency=1)

    assert report.total == 2
    assert report.passed == 1
    assert report.by_scorer()["groundedness"]["pass_rate"] == 0.5
    failures = report.to_dict()["failures"]
    assert [f["id"] for f in failures] == ["bad"]
    assert "groundedness" in {s["name"] for s in failures[0]["failed"]}


async def test_eval_disables_the_cache_so_repeats_are_really_re_run(settings):
    provider = ScriptedProvider(["first", "second", "third"])
    platform = make_platform(settings, provider)
    cases = [EvalCase(query="same question", id=f"c{i}") for i in range(3)]

    await run_eval(platform, cases, concurrency=1)

    assert len(provider.requests) == 3, "a warm cache turned the eval into a no-op"
    # And the setting is restored afterwards.
    assert platform.pipeline.config.cache_responses is True


async def test_eval_records_a_case_error_without_aborting_the_run(settings):
    from aie.gateway.providers.echo import EchoProvider

    platform = make_platform(settings, EchoProvider(fail_times=99))
    report = await run_eval(platform, [EvalCase(query="anything", id="boom")], concurrency=1)

    assert report.errors == 1
    assert report.passed == 0
    assert "every candidate failed" in report.to_dict()["failures"][0]["error"]


async def test_report_renders_without_blowing_up_on_an_empty_run(settings):
    platform = make_platform(settings, ScriptedProvider([]))
    report = await run_eval(platform, [], concurrency=1)
    assert report.total == 0
    assert "0/0 passed" in report.render()
