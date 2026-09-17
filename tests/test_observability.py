"""Tracing, metrics and log correlation."""

from __future__ import annotations

import json
import logging
import time

import pytest

from aie.observe.logging import JSONFormatter, bind_trace, current_trace_id
from aie.observe.metrics import MetricsRegistry
from aie.observe.trace import TRACES, LoggingExporter, Trace, TraceStore
from aie.types import Query
from tests.conftest import ScriptedProvider, make_platform


def build_trace() -> Trace:
    trace = Trace(tenant_id="acme", route="default")
    with trace.span("cache.lookup", hit=False):
        time.sleep(0.002)
    with trace.span("context.construct") as outer:
        with trace.span("action.search_documents", cached=False):
            time.sleep(0.01)
        outer.set(chunks=2)
    with trace.span("gateway.generate", model="claude-opus-5", attempt=1) as span:
        time.sleep(0.005)
        span.set(input_tokens=1200, output_tokens=300)
    trace.add_cost(0.0135)
    trace.finish()
    return trace


def test_self_time_is_not_charged_to_a_parent_span():
    trace = build_trace()
    summary = trace.summary()

    # context.construct wraps the retrieval action; its own time is tiny and
    # the 10ms belongs to `action`, not to `context`.
    assert summary["ms_by_layer"]["action"] > 5
    assert summary["ms_by_layer"]["context"] < 5


def test_summary_attributes_cost_tokens_and_attempts():
    summary = build_trace().summary()

    assert summary["cost_usd"] == 0.0135
    assert summary["tokens"] == {"input": 1200, "output": 300}
    assert summary["generation_attempts"] == 1
    assert summary["ms_by_model"]["claude-opus-5"] > 0
    assert summary["error"] is False


def test_errors_are_recorded_and_reraised():
    trace = Trace()
    with pytest.raises(ValueError):
        with trace.span("gateway.generate"):
            raise ValueError("upstream exploded")

    assert trace.has_error
    assert trace.summary()["error"] is True
    assert "ValueError: upstream exploded" in trace.root.children[0].error


def test_otlp_export_has_valid_parentage_and_timestamps():
    trace = build_trace()
    spans = trace.to_otlp()["resourceSpans"][0]["scopeSpans"][0]["spans"]

    by_id = {s["spanId"]: s for s in spans}
    root = next(s for s in spans if "parentSpanId" not in s)
    assert root["name"] == "request"

    # Every non-root span points at a span that exists in the same payload.
    for span in spans:
        if "parentSpanId" in span:
            assert span["parentSpanId"] in by_id
        assert int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"])
        assert int(span["startTimeUnixNano"]) > 1_700_000_000_000_000_000

    nested = next(s for s in spans if s["name"] == "action.search_documents")
    assert by_id[nested["parentSpanId"]]["name"] == "context.construct"


def test_otlp_attribute_types_are_typed_not_stringified():
    trace = Trace()
    with trace.span("gateway.generate", model="m", attempt=1, cached=False, score=0.5):
        pass
    span = next(
        s
        for s in trace.to_otlp()["resourceSpans"][0]["scopeSpans"][0]["spans"]
        if s["name"] == "gateway.generate"
    )
    values = {a["key"]: a["value"] for a in span["attributes"]}
    assert values["attempt"] == {"intValue": "1"}
    assert values["cached"] == {"boolValue": False}
    assert values["score"] == {"doubleValue": 0.5}
    assert values["model"] == {"stringValue": "m"}


def test_emit_is_idempotent():
    from aie.observe import trace as trace_module

    store = TraceStore()
    trace = Trace()

    original = trace_module._EXPORTERS
    trace_module._EXPORTERS = [store]
    try:
        trace.emit()
        trace.emit()
    finally:
        trace_module._EXPORTERS = original

    assert len(store) == 1


def test_a_broken_exporter_does_not_break_the_request():
    class Broken:
        def export(self, trace):
            raise RuntimeError("collector is down")

    from aie.observe import trace as trace_module

    store = TraceStore()
    original = trace_module._EXPORTERS
    trace_module._EXPORTERS = [Broken(), store]
    try:
        Trace().emit()  # must not raise
    finally:
        trace_module._EXPORTERS = original

    assert len(store) == 1, "a later exporter still ran after an earlier one failed"


def test_trace_store_is_bounded_and_searchable():
    store = TraceStore(max_traces=3)
    traces = [Trace() for _ in range(5)]
    for trace in traces:
        trace.finish()
        store.export(trace)

    assert len(store) == 3
    assert store.get(traces[0].trace_id) is None, "the ring must evict the oldest"
    assert store.get(traces[-1].trace_id) is not None
    assert [t.trace_id for t in store.recent(2)] == [
        traces[4].trace_id,
        traces[3].trace_id,
    ]


def test_trace_store_can_filter_to_errors():
    store = TraceStore()
    clean = Trace()
    clean.finish()
    broken = Trace()
    with pytest.raises(ValueError):
        with broken.span("boom"):
            raise ValueError("x")
    broken.finish()
    store.export(clean)
    store.export(broken)

    assert [t.trace_id for t in store.recent(errors_only=True)] == [broken.trace_id]


# --- metrics ---------------------------------------------------------------


def test_histogram_quantiles_bound_the_true_value():
    metrics = MetricsRegistry()
    for value in (0.2, 0.4, 1.5, 3.0, 12.0):
        metrics.observe("lat", value)
    histogram = metrics.histogram("lat")

    assert histogram.quantile(0.5) == 2.0     # true median 1.5, bucket bound 2.0
    assert histogram.quantile(0.95) == 15.0   # true p95 ~12, bucket bound 15.0
    assert MetricsRegistry().histogram("empty").quantile(0.5) is None


def test_prometheus_exposition_is_well_formed():
    metrics = MetricsRegistry()
    metrics.incr("pipeline.ok", route="default")
    metrics.observe("gateway.latency.seconds", 0.3, model="claude-opus-5")
    text = metrics.prometheus()

    assert "# TYPE pipeline_ok counter" in text
    assert 'pipeline_ok{route="default"} 1.0' in text
    assert "# TYPE gateway_latency_seconds histogram" in text
    assert 'gateway_latency_seconds_bucket{model="claude-opus-5",le="+Inf"} 1' in text
    assert 'gateway_latency_seconds_count{model="claude-opus-5"} 1' in text
    # No dots survive into metric names; Prometheus rejects them.
    for line in text.splitlines():
        if not line.startswith("#"):
            assert "." not in line.split("{")[0].split(" ")[0]


def test_label_values_are_escaped():
    metrics = MetricsRegistry()
    metrics.incr("weird", model='say "hi"\\now')
    assert '\\"hi\\"' in metrics.prometheus()


def test_counter_name_does_not_collide_with_a_name_label():
    metrics = MetricsRegistry()
    metrics.incr("guardrail.action", name="pii_redaction", action="modify")
    assert metrics.counters == {
        "guardrail.action{action=modify,name=pii_redaction}": 1.0
    }


# --- log correlation -------------------------------------------------------


def test_log_records_carry_the_bound_trace_id():
    formatter = JSONFormatter()

    def render(message: str, **extra) -> dict:
        record = logging.LogRecord("aie.test", logging.INFO, "f", 1, message, (), None)
        record.__dict__.update(extra)
        return json.loads(formatter.format(record))

    assert "trace_id" not in render("unbound")

    with bind_trace("trace_abc", tenant_id="acme"):
        payload = render("bound", guardrail="pii_leak")
        assert payload["trace_id"] == "trace_abc"
        assert payload["tenant_id"] == "acme"
        assert payload["guardrail"] == "pii_leak"

    assert current_trace_id() is None, "the binding must not leak past its block"


async def test_pipeline_binds_the_trace_id_for_correlation(settings):
    seen: list[str | None] = []

    class Peeking(ScriptedProvider):
        async def generate(self, request):
            seen.append(current_trace_id())
            return await super().generate(request)

    platform = make_platform(settings, Peeking(["ok"]))
    result = await platform.pipeline.run(
        Query(text="refund policy", user_id="u", tenant_id="acme")
    )

    assert seen == [result.trace_id]


async def test_a_served_request_lands_in_the_trace_store(settings):
    platform = make_platform(settings, ScriptedProvider(["ok"]))
    result = await platform.pipeline.run(
        Query(text="refund policy for EU orders", user_id="u", tenant_id="acme")
    )

    stored = TRACES.get(result.trace_id)
    assert stored is not None
    summary = stored.summary()
    assert summary["generation_attempts"] == 1
    assert "gateway" in summary["ms_by_layer"]


def test_trace_log_fields_are_queryable_not_nested_json(caplog):
    from aie.observe import trace as trace_module
    from aie.observe.trace import LoggingExporter

    original = trace_module._EXPORTERS
    trace_module._EXPORTERS = [LoggingExporter(summary_only=True)]
    try:
        with caplog.at_level(logging.INFO, logger="aie.trace"):
            trace = Trace(tenant_id="acme")
            trace.add_cost(0.02)
            trace.emit()
    finally:
        trace_module._EXPORTERS = original

    record = caplog.records[-1]
    assert record.tenant_id == "acme"
    assert record.cost_usd == 0.02
    assert "{" not in record.getMessage(), "the payload must not be JSON inside the message"


def test_a_trace_attribute_shadowing_a_log_field_does_not_raise():
    from aie.observe import trace as trace_module
    from aie.observe.trace import LoggingExporter

    original = trace_module._EXPORTERS
    trace_module._EXPORTERS = [LoggingExporter(summary_only=True)]
    try:
        # `name`, `module` and `args` are LogRecord attributes; logging raises
        # KeyError if `extra` shadows them.
        Trace(name="collides", module="also collides", args="and this").emit()
    finally:
        trace_module._EXPORTERS = original
