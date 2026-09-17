"""Request tracing, cost and latency accounting.

The reference architecture diagram has no observability plane -- it is all
request path. This module is that missing plane. Every layer opens a span, so
one trace answers "where did the 4 seconds and the 3 cents go".

Stdlib only, and intentionally export-shaped: ``Trace.to_dict`` produces a
structure that maps cleanly onto OpenTelemetry spans when we swap the exporter.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from aie.types import new_id, now_ms

logger = logging.getLogger("aie.trace")


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or now_ms()) - self.start_ms

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
        }
        if self.error:
            out["error"] = self.error
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out


class Trace:
    """A single request's span tree. Not thread-shared; one per request."""

    def __init__(self, trace_id: str | None = None, **attributes: Any) -> None:
        self.trace_id = trace_id or new_id("trace")
        self.attributes: dict[str, Any] = dict(attributes)
        self.root = Span(name="request", start_ms=now_ms())
        self._stack: list[Span] = [self.root]
        self.cost_usd: float = 0.0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        span = Span(name=name, start_ms=now_ms(), attributes=dict(attrs))
        self._stack[-1].children.append(span)
        self._stack.append(span)
        try:
            yield span
        except Exception as exc:  # record then re-raise; tracing never swallows
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ms = now_ms()
            self._stack.pop()

    def add_cost(self, usd: float) -> None:
        self.cost_usd += usd

    def finish(self) -> None:
        if self.root.end_ms is None:
            self.root.end_ms = now_ms()

    @property
    def latency_ms(self) -> float:
        return self.root.duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
            **self.attributes,
            **self.root.to_dict(),
        }

    def emit(self) -> None:
        self.finish()
        logger.info("trace %s", json.dumps(self.to_dict(), default=str))


class MetricsSink:
    """Process-local counters. Swap for a StatsD/Prometheus client later."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}

    def incr(self, metric: str, value: float = 1.0, **tags: Any) -> None:
        # First arg is `metric`, not `name`: callers legitimately tag with
        # name=<component>, which would collide with a positional `name`.
        key = metric if not tags else f"{metric}{{{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}}}"
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self.counters)

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()


METRICS = MetricsSink()
