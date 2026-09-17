"""Offline eval: run a dataset through the real pipeline and score it.

Deliberately runs the *whole* pipeline -- cache, retrieval, guardrails,
gateway -- not just a model call. Most regressions in a system like this come
from retrieval or a prompt change, and a harness that bypasses those layers
cannot see them.

One trap this handles for you: the response cache is disabled for the duration
of a run. Leave it on and the second occurrence of a repeated question returns
a cached answer in 0ms, which quietly turns your eval into a measurement of
your cache.

    python -m aie.eval.runner examples/eval_cases.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from aie.eval.scorers import HEURISTIC_SCORERS, Scorer
from aie.eval.types import EvalSample, Score
from aie.types import Query


@dataclass
class EvalCase:
    query: str
    id: str = ""
    user_id: str = "eval-user"
    tenant_id: str = "eval-tenant"
    route: str = "default"
    expected: str | None = None
    expected_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int) -> "EvalCase":
        known = {f for f in cls.__dataclass_fields__}
        case = cls(**{k: v for k, v in raw.items() if k in known})
        if not case.id:
            case.id = f"case-{index:03d}"
        return case


@dataclass
class CaseResult:
    case: EvalCase
    response: str = ""
    scores: list[Score] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    blocked: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores)


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    def by_scorer(self) -> dict[str, dict[str, float]]:
        names: list[str] = []
        for result in self.results:
            for score in result.scores:
                if score.name not in names:
                    names.append(score.name)

        summary: dict[str, dict[str, float]] = {}
        for name in names:
            scores = [s for r in self.results for s in r.scores if s.name == name]
            if not scores:
                continue
            summary[name] = {
                "pass_rate": round(sum(1 for s in scores if s.passed) / len(scores), 4),
                "mean": round(statistics.fmean(s.value for s in scores), 4),
                "n": len(scores),
            }
        return summary

    def latency(self) -> dict[str, float]:
        values = sorted(r.latency_ms for r in self.results if r.error is None)
        if not values:
            return {}
        return {
            "p50": round(values[len(values) // 2], 2),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 2),
            "max": round(values[-1], 2),
        }

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.results), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.passed / self.total, 4) if self.total else 0.0,
            "errors": self.errors,
            "total_cost_usd": self.total_cost_usd,
            "latency_ms": self.latency(),
            "scorers": self.by_scorer(),
            "failures": [
                {
                    "id": r.case.id,
                    "query": r.case.query,
                    "response": r.response[:200],
                    "error": r.error,
                    "failed": [s.to_dict() for s in r.scores if not s.passed],
                }
                for r in self.results
                if not r.passed
            ],
        }

    def render(self) -> str:
        lines = [
            "",
            f"  cases      {self.passed}/{self.total} passed"
            + (f"  ({self.errors} errored)" if self.errors else ""),
            f"  cost       ${self.total_cost_usd:.4f}",
        ]
        latency = self.latency()
        if latency:
            lines.append(
                f"  latency    p50 {latency['p50']:.0f}ms  p95 {latency['p95']:.0f}ms"
                f"  max {latency['max']:.0f}ms"
            )
        lines.append("")
        width = max((len(n) for n in self.by_scorer()), default=10)
        for name, stats in self.by_scorer().items():
            bar = "#" * round(stats["pass_rate"] * 20)
            lines.append(
                f"  {name:<{width}}  {stats['pass_rate']:>6.1%}  mean {stats['mean']:.2f}  {bar}"
            )

        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.extend(["", f"  {len(failures)} failing case(s):"])
            for result in failures[:10]:
                reason = result.error or ", ".join(
                    f"{s.name}({s.detail})" for s in result.scores if not s.passed
                )
                lines.append(f"    {result.case.id}  {result.case.query[:48]!r}")
                lines.append(f"      -> {reason[:150]}")
        return "\n".join(lines) + "\n"


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read a JSONL dataset (one case per line)."""
    cases: list[EvalCase] = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(EvalCase.from_dict(json.loads(line), index))
    return cases


async def run_eval(
    platform,
    cases: Sequence[EvalCase],
    *,
    scorers: list[Scorer] | None = None,
    judge: Any | None = None,
    concurrency: int = 4,
) -> EvalReport:
    scorers = list(scorers if scorers is not None else HEURISTIC_SCORERS)

    # See the module docstring: a warm cache silently invalidates the run.
    cache_was_enabled = platform.pipeline.config.cache_responses
    platform.pipeline.config.cache_responses = False
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        async with semaphore:
            result = CaseResult(case=case)
            try:
                response = await platform.pipeline.run(
                    Query(
                        text=case.query,
                        user_id=case.user_id,
                        tenant_id=case.tenant_id,
                        route=case.route,
                    )
                )
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                return result

            result.response = response.text
            result.latency_ms = response.latency_ms
            result.cost_usd = response.cost_usd
            result.blocked = response.blocked

            trace = platform.traces.get(response.trace_id) if platform.traces else None
            chunks = _chunks_from(platform, case)
            sample = EvalSample(
                query=case.query,
                response=response.text,
                chunks=chunks,
                expected=case.expected,
                expected_keywords=case.expected_keywords,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
                blocked=response.blocked,
                route=case.route,
                metadata={"trace_id": response.trace_id, "trace": trace is not None},
            )
            for scorer in scorers:
                try:
                    result.scores.append(scorer.score(sample))
                except Exception as exc:
                    result.scores.append(
                        Score(getattr(scorer, "name", "?"), 0.0, True, f"scorer error: {exc}")
                    )
            if judge is not None:
                try:
                    result.scores.append(await judge.score_async(sample))
                except Exception as exc:
                    result.scores.append(Score("llm_judge", 0.0, True, f"judge error: {exc}"))
            return result

    try:
        results = await asyncio.gather(*(run_one(c) for c in cases))
    finally:
        platform.pipeline.config.cache_responses = cache_was_enabled

    return EvalReport(list(results))


def _chunks_from(platform, case: EvalCase):
    """Re-run retrieval to score groundedness against what the model was given."""
    try:
        return platform.read_actions.call(
            "search_documents",
            tenant_id=case.tenant_id,
            query=case.query,
            k=platform.pipeline.context.config.top_k,
        )
    except Exception:
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline eval.")
    parser.add_argument("dataset", help="JSONL file, one case per line")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--judge", action="store_true", help="also grade with the model judge (costs money)"
    )
    parser.add_argument(
        "--fail-under", type=float, default=None,
        help="exit non-zero if the pass rate falls below this (for CI)",
    )
    args = parser.parse_args(argv)

    from aie.config import build_platform
    from aie.eval.scorers import LLMJudge
    from aie.observe.logging import configure_logging

    configure_logging(json_format=True)
    platform = build_platform()
    try:
        cases = load_cases(args.dataset)
    except FileNotFoundError:
        print(f"no such dataset: {args.dataset}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"{args.dataset} is not valid JSONL: {exc}", file=sys.stderr)
        return 2
    if not cases:
        print(f"no cases in {args.dataset}", file=sys.stderr)
        return 2

    report = asyncio.run(
        run_eval(
            platform,
            cases,
            judge=LLMJudge(platform.gateway) if args.judge else None,
            concurrency=args.concurrency,
        )
    )

    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())

    if args.fail_under is not None:
        rate = report.passed / report.total if report.total else 0.0
        if rate < args.fail_under:
            print(f"pass rate {rate:.1%} is below --fail-under {args.fail_under:.1%}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
