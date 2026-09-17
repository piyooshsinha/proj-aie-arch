"""Request tracing: span tree, cost and latency attribution, export.

The reference architecture diagram has no observability plane -- it is all
request path. This module is that missing plane.

Three things make it more than a logger:

* **Wall-clock anchoring.** Spans are timed with ``perf_counter`` (monotonic,
  immune to clock steps) but exported against a single wall-clock anchor taken
  at trace start, so a span has both an accurate duration and a real timestamp.
* **An exporter seam.** ``Trace.emit`` fans out to registered exporters.
  ``to_otlp`` already produces OpenTelemetry's span shape, so pointing this at a
  collector is a new exporter, not a rewrite.
* **Attribution.** ``summary()`` answers "where did the 4 seconds and the 3
  cents go" by layer and by model, which is the question you actually have at
  3am.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

from aie.observe.metrics import METRICS  # re-exported: callers import it from here
from aie.types import new_id, now_ms

__all__ = ["METRICS", "Span", "Trace", "TraceStore", "TRACES", "register_exporter"]

logger = logging.getLogger("aie.trace")


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list["Span"] = field(default_factory=list)
    span_id: str = field(default_factory=lambda: new_id("span"))

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or now_ms()) - self.start_ms

    @property
    def self_ms(self) -> float:
        """Duration minus time accounted for by children.

        This is what tells you whether a slow ``context.construct`` is slow
        itself or just waiting on the retrieval span underneath it.
        """
        return max(0.0, self.duration_ms - sum(c.duration_ms for c in self.children))

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def walk(self) -> Iterator["Span"]:
        yield self
        for child in self.children:
            yield from child.walk()

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
        self.start_unix_ns = time.time_ns()
        self._start_perf_ms = now_ms()
        self.root = Span(name="request", start_ms=self._start_perf_ms)
        self._stack: list[Span] = [self.root]
        self.cost_usd: float = 0.0
        self._emitted = False

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

    @property
    def has_error(self) -> bool:
        return any(s.error for s in self.root.walk())

    # --- attribution -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Where the time and money went, grouped by layer and model."""
        by_layer: dict[str, float] = {}
        by_model: dict[str, float] = {}
        tokens = {"input": 0, "output": 0}
        attempts = 0

        for span in self.root.walk():
            if span is self.root:
                continue
            layer = span.name.split(".")[0]
            by_layer[layer] = round(by_layer.get(layer, 0.0) + span.self_ms, 2)

            if span.name == "gateway.generate":
                attempts += 1
                model = str(span.attributes.get("model", "unknown"))
                by_model[model] = round(by_model.get(model, 0.0) + span.duration_ms, 2)
                tokens["input"] += int(span.attributes.get("input_tokens", 0) or 0)
                tokens["output"] += int(span.attributes.get("output_tokens", 0) or 0)

        return {
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "ms_by_layer": dict(sorted(by_layer.items(), key=lambda kv: -kv[1])),
            "ms_by_model": by_model,
            "tokens": tokens,
            "generation_attempts": attempts,
            "error": self.has_error,
        }

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.start_unix_ns // 1_000_000,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
            **self.attributes,
            **self.root.to_dict(),
        }

    def _unix_nanos(self, perf_ms: float) -> int:
        return self.start_unix_ns + int((perf_ms - self._start_perf_ms) * 1_000_000)

    def to_otlp(self) -> dict[str, Any]:
        """OpenTelemetry ``resourceSpans`` shape.

        Deliberately built by hand rather than pulling in the OTel SDK: it is
        thirty lines, it keeps the dependency out of every service that imports
        this package, and the moment a collector is actually wanted, a real
        exporter can consume this directly.
        """
        spans: list[dict[str, Any]] = []

        def visit(span: Span, parent: str | None) -> None:
            attributes = [
                {"key": k, "value": _otlp_value(v)} for k, v in span.attributes.items()
            ]
            record: dict[str, Any] = {
                "traceId": self.trace_id,
                "spanId": span.span_id,
                "name": span.name,
                "kind": 1,  # SPAN_KIND_INTERNAL
                "startTimeUnixNano": str(self._unix_nanos(span.start_ms)),
                "endTimeUnixNano": str(
                    self._unix_nanos(span.end_ms if span.end_ms is not None else now_ms())
                ),
                "attributes": attributes,
                "status": {"code": 2, "message": span.error} if span.error else {"code": 1},
            }
            if parent:
                record["parentSpanId"] = parent
            spans.append(record)
            for child in span.children:
                visit(child, span.span_id)

        visit(self.root, None)
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "aie-platform"}},
                            *[
                                {"key": k, "value": _otlp_value(v)}
                                for k, v in self.attributes.items()
                            ],
                        ]
                    },
                    "scopeSpans": [{"scope": {"name": "aie"}, "spans": spans}],
                }
            ]
        }

    def emit(self) -> None:
        """Finish the trace and hand it to every registered exporter.

        Idempotent: the pipeline emits in a ``finally``, and an exception on the
        way out must not produce two copies of the same trace.
        """
        if self._emitted:
            return
        self._emitted = True
        self.finish()

        summary = self.summary()
        METRICS.observe("request.latency.seconds", summary["latency_ms"] / 1000.0)
        if summary["cost_usd"]:
            from aie.observe.metrics import COST_BUCKETS_USD

            METRICS.observe(
                "request.cost.usd", summary["cost_usd"], buckets=COST_BUCKETS_USD
            )

        for exporter in list(_EXPORTERS):
            try:
                exporter.export(self)
            except Exception:  # an exporter must never break a request
                logger.exception("trace exporter %r failed", type(exporter).__name__)


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


# --- exporters -------------------------------------------------------------


@runtime_checkable
class SpanExporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class LoggingExporter:
    """One structured line per trace. The zero-infrastructure option.

    Fields go through ``extra`` rather than into the message string. Under the
    JSON formatter that makes them queryable top-level keys instead of a
    JSON-encoded blob nested inside ``message``; under a plain formatter the
    message alone still reads as a useful summary.
    """

    def __init__(self, level: int = logging.INFO, summary_only: bool = False) -> None:
        self.level = level
        self.summary_only = summary_only

    def export(self, trace: Trace) -> None:
        summary = trace.summary()
        payload = (
            {"trace_id": trace.trace_id, **trace.attributes, **summary}
            if self.summary_only
            else {"trace_id": trace.trace_id, **trace.attributes, "span_tree": trace.to_dict()}
        )
        logger.log(
            self.level,
            "trace %s %.0fms $%.4f%s",
            trace.trace_id,
            summary["latency_ms"],
            summary["cost_usd"],
            " ERROR" if summary["error"] else "",
            extra=_safe_extra(payload),
        )


def _safe_extra(payload: dict[str, Any]) -> dict[str, Any]:
    """Rename keys that would collide with LogRecord's own attributes.

    ``logging`` raises KeyError if ``extra`` shadows a standard field such as
    ``name`` or ``module``, and a trace attribute is arbitrary caller data.
    """
    from aie.observe.logging import RESERVED_LOG_FIELDS

    return {
        (f"trace_{k}" if k in RESERVED_LOG_FIELDS else k): v for k, v in payload.items()
    }


class TraceStore:
    """Bounded in-memory ring of recent traces, so ``/traces`` can serve them.

    A ring, not a list: an unbounded trace buffer is a memory leak with a
    dashboard attached. Production sends these to a collector and keeps this
    for local debugging.
    """

    def __init__(self, max_traces: int = 200) -> None:
        self._traces: deque[Trace] = deque(maxlen=max_traces)
        self._lock = threading.Lock()

    def export(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            return next((t for t in self._traces if t.trace_id == trace_id), None)

    def recent(self, limit: int = 20, *, errors_only: bool = False) -> list[Trace]:
        with self._lock:
            traces = list(self._traces)
        if errors_only:
            traces = [t for t in traces if t.has_error]
        return list(reversed(traces))[:limit]

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    def __len__(self) -> int:
        return len(self._traces)


TRACES = TraceStore()
_EXPORTERS: list[SpanExporter] = [LoggingExporter(summary_only=True), TRACES]


def register_exporter(exporter: SpanExporter) -> None:
    _EXPORTERS.append(exporter)


def reset_exporters(exporters: list[SpanExporter] | None = None) -> None:
    """Replace the exporter list. Used by tests and by the composition root."""
    global _EXPORTERS
    _EXPORTERS = list(exporters) if exporters is not None else [LoggingExporter(), TRACES]
