"""Metrics with real types, and a Prometheus exposition.

The first pass at this was a dict of floats, which is fine until someone asks
"what is p95 latency" -- a question a counter cannot answer at any sampling
rate. Histograms are the reason this module exists.

Naming: metrics are written with dots (``gateway.tokens.input``) because that
reads better at the call site, and sanitized to underscores on the way out,
because Prometheus does not accept dots.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Iterable

_INVALID = re.compile(r"[^a-zA-Z0-9_]")

# Seconds. Tuned for LLM work: the interesting range is 100ms to a minute, and
# the default Prometheus buckets top out at 10s, which puts most of our
# generations in +Inf where they are invisible.
LATENCY_BUCKETS_S: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0,
)
# USD per request. A single Opus call on a long context can clear a dollar.
COST_BUCKETS_USD: tuple[float, ...] = (
    0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
)
TOKEN_BUCKETS: tuple[float, ...] = (
    64, 256, 1024, 4096, 16384, 65536, 262144, 1048576,
)

LabelKey = tuple[tuple[str, str], ...]


def _labels(tags: dict[str, object]) -> LabelKey:
    return tuple(sorted((str(k), str(v)) for k, v in tags.items()))


def _render_labels(labels: LabelKey, extra: tuple[str, str] | None = None) -> str:
    pairs = list(labels)
    if extra:
        pairs.append(extra)
    if not pairs:
        return ""
    inner = ",".join(f'{_INVALID.sub("_", k)}="{_escape(v)}"' for k, v in pairs)
    return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _legacy_key(name: str, labels: LabelKey) -> str:
    """The flat ``name{k=v,...}`` form, kept for the JSON snapshot."""
    if not labels:
        return name
    return f"{name}{{{','.join(f'{k}={v}' for k, v in labels)}}}"


@dataclass
class Counter:
    name: str
    help: str = ""
    values: dict[LabelKey, float] = field(default_factory=dict)

    def add(self, value: float, labels: LabelKey) -> None:
        self.values[labels] = self.values.get(labels, 0.0) + value

    def expose(self) -> Iterable[str]:
        metric = _INVALID.sub("_", self.name)
        yield f"# HELP {metric} {self.help or self.name}"
        yield f"# TYPE {metric} counter"
        for labels, value in sorted(self.values.items()):
            yield f"{metric}{_render_labels(labels)} {value!r}"


@dataclass
class _Bucketed:
    counts: list[int]
    total: float = 0.0
    count: int = 0

    def observe(self, value: float, bounds: tuple[float, ...]) -> None:
        self.total += value
        self.count += 1
        for index, bound in enumerate(bounds):
            if value <= bound:
                self.counts[index] += 1
        # The +Inf bucket always gets everything.
        self.counts[-1] += 1


@dataclass
class Histogram:
    name: str
    help: str = ""
    buckets: tuple[float, ...] = LATENCY_BUCKETS_S
    values: dict[LabelKey, _Bucketed] = field(default_factory=dict)

    def observe(self, value: float, labels: LabelKey) -> None:
        series = self.values.get(labels)
        if series is None:
            series = _Bucketed(counts=[0] * (len(self.buckets) + 1))
            self.values[labels] = series
        series.observe(value, self.buckets)

    def quantile(self, q: float, labels: LabelKey = ()) -> float | None:
        """Bucket-interpolated quantile, the same estimate Prometheus makes.

        Returns the upper bound of the bucket the quantile falls into, so it is
        an over-estimate bounded by bucket width -- good enough for an SLO
        check, not a substitute for real percentile data.
        """
        series = self.values.get(labels)
        if series is None or series.count == 0:
            return None
        target = q * series.count
        # counts are already cumulative: observe() increments every bucket
        # whose bound the value satisfies.
        for index, bound in enumerate(self.buckets):
            if series.counts[index] >= target:
                return bound
        return math.inf

    def expose(self) -> Iterable[str]:
        metric = _INVALID.sub("_", self.name)
        yield f"# HELP {metric} {self.help or self.name}"
        yield f"# TYPE {metric} histogram"
        for labels, series in sorted(self.values.items()):
            for index, bound in enumerate(self.buckets):
                yield (
                    f"{metric}_bucket"
                    f"{_render_labels(labels, ('le', repr(bound)))} "
                    f"{series.counts[index]}"
                )
            yield f"{metric}_bucket{_render_labels(labels, ('le', '+Inf'))} {series.counts[-1]}"
            yield f"{metric}_sum{_render_labels(labels)} {series.total!r}"
            yield f"{metric}_count{_render_labels(labels)} {series.count}"


class MetricsRegistry:
    """Thread-safe registry. Swap for a real client by reimplementing this class."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str, help: str = "") -> Counter:
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                counter = self._counters[name] = Counter(name, help)
            return counter

    def histogram(
        self, name: str, help: str = "", buckets: tuple[float, ...] = LATENCY_BUCKETS_S
    ) -> Histogram:
        with self._lock:
            histogram = self._histograms.get(name)
            if histogram is None:
                histogram = self._histograms[name] = Histogram(name, help, buckets)
            return histogram

    # --- call-site API -----------------------------------------------------

    def incr(self, metric: str, value: float = 1.0, **tags: object) -> None:
        # First arg is `metric`, not `name`: callers legitimately tag with
        # name=<component>, which would collide with a positional `name`.
        counter = self.counter(metric)
        with self._lock:
            counter.add(value, _labels(tags))

    def observe(
        self,
        metric: str,
        value: float,
        *,
        buckets: tuple[float, ...] = LATENCY_BUCKETS_S,
        **tags: object,
    ) -> None:
        histogram = self.histogram(metric, buckets=buckets)
        with self._lock:
            histogram.observe(value, _labels(tags))

    # --- readback ----------------------------------------------------------

    @property
    def counters(self) -> dict[str, float]:
        """Flat ``name{k=v}`` -> value, for the JSON metrics endpoint."""
        with self._lock:
            return {
                _legacy_key(counter.name, labels): value
                for counter in self._counters.values()
                for labels, value in counter.values.items()
            }

    def snapshot(self) -> dict[str, float]:
        return self.counters

    def histogram_summary(self) -> dict[str, dict[str, float | None]]:
        with self._lock:
            histograms = list(self._histograms.values())
        summary: dict[str, dict[str, float | None]] = {}
        for histogram in histograms:
            for labels, series in histogram.values.items():
                key = _legacy_key(histogram.name, labels)
                summary[key] = {
                    "count": series.count,
                    "sum": round(series.total, 6),
                    "avg": round(series.total / series.count, 6) if series.count else 0.0,
                    "p50": histogram.quantile(0.5, labels),
                    "p95": histogram.quantile(0.95, labels),
                    "p99": histogram.quantile(0.99, labels),
                }
        return summary

    def prometheus(self) -> str:
        with self._lock:
            families = [*self._counters.values(), *self._histograms.values()]
        lines: list[str] = []
        for family in families:
            lines.extend(family.expose())
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


METRICS = MetricsRegistry()
