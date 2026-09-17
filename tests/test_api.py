"""HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aie.actions.write import WriteAction
from aie.api.app import create_app
from aie.gateway.providers.echo import EchoProvider
from tests.conftest import make_platform


@pytest.fixture
def client(settings):
    platform = make_platform(settings, EchoProvider("Refunds take 14 days."))
    sent: list[tuple[str, str]] = []
    platform.write_actions.register(
        WriteAction(
            name="send_email",
            description="Send an email",
            handler=lambda *, tenant_id, to, subject: sent.append((to, subject)) or "sent",
            describe=lambda to, subject: f"Send an email to {to} with subject {subject!r}",
        )
    )
    app = create_app(platform)
    with TestClient(app) as client:
        client.sent = sent
        yield client


def test_health_reports_wiring(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "default" in body["routes"]
    assert body["documents"] == 4


def test_query_returns_an_answer_with_its_trace(client):
    response = client.post(
        "/query",
        json={"query": "refund policy for EU orders", "user_id": "alice", "tenant_id": "acme"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Refunds take 14 days."
    assert body["cached"] is False
    assert body["trace_id"].startswith("trace_")
    assert body["latency_ms"] >= 0


def test_repeated_query_reports_a_cache_hit(client):
    payload = {"query": "office hours", "user_id": "alice", "tenant_id": "acme"}
    client.post("/query", json=payload)
    body = client.post("/query", json=payload).json()
    assert body["cached"] is True


def test_empty_query_is_rejected(client):
    response = client.post("/query", json={"query": "", "user_id": "a", "tenant_id": "b"})
    assert response.status_code == 422


def test_unknown_route_is_a_client_error(client):
    response = client.post(
        "/query",
        json={"query": "hello", "user_id": "a", "tenant_id": "b", "route": "nope"},
    )
    assert response.status_code == 400


def test_adding_a_document_invalidates_cached_answers(client):
    payload = {"query": "what is the parking policy", "user_id": "alice", "tenant_id": "acme"}
    client.post("/query", json=payload)
    assert client.post("/query", json=payload).json()["cached"] is True

    client.post("/documents", json={"id": "d9", "text": "Parking is free for staff."})

    assert client.post("/query", json=payload).json()["cached"] is False


def test_metrics_expose_cache_and_guardrail_counters(client):
    client.post(
        "/query",
        json={"query": "call me on (415) 555-0132", "user_id": "a", "tenant_id": "b"},
    )
    body = client.get("/metrics").json()
    assert body["response_cache"]["scope"] == "user"
    assert any("guardrail.action" in key for key in body["counters"])


def test_write_requires_approval_before_it_runs(client):
    created = client.post(
        "/writes",
        json={
            "action": "send_email",
            "tenant_id": "acme",
            "user_id": "alice",
            "arguments": {"to": "bob@x.com", "subject": "Refund"},
        },
    ).json()

    assert created["status"] == "proposed"
    assert "bob@x.com" in created["preview"]
    assert client.sent == [], "the write ran without approval"

    pending = client.get("/writes/pending").json()
    assert len(pending) == 1

    approved = client.post(f"/writes/{created['id']}/approve?approver=carol").json()
    assert approved["status"] == "executed"
    assert client.sent == [("bob@x.com", "Refund")]


# --- observability endpoints ----------------------------------------------


def test_prometheus_endpoint_serves_scrapeable_text(client):
    client.post("/query", json={"query": "refund policy", "user_id": "a", "tenant_id": "b"})
    response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# TYPE pipeline_ok counter" in body
    assert "# TYPE request_latency_seconds histogram" in body


def test_traces_endpoint_lists_recent_requests_with_attribution(client):
    query = client.post(
        "/query", json={"query": "refund policy", "user_id": "alice", "tenant_id": "acme"}
    ).json()

    listed = client.get("/traces").json()
    assert listed[0]["trace_id"] == query["trace_id"]
    assert listed[0]["tenant_id"] == "acme"
    assert "ms_by_layer" in listed[0]


def test_a_single_trace_can_be_fetched_as_a_tree_or_as_otlp(client):
    trace_id = client.post(
        "/query", json={"query": "refund policy", "user_id": "a", "tenant_id": "b"}
    ).json()["trace_id"]

    tree = client.get(f"/traces/{trace_id}").json()
    assert tree["name"] == "request"
    assert any(c["name"] == "gateway.generate" for c in _flatten(tree))

    otlp = client.get(f"/traces/{trace_id}?format=otlp").json()
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert {s["traceId"] for s in spans} == {trace_id}

    summary = client.get(f"/traces/{trace_id}?format=summary").json()
    assert "ms_by_layer" in summary and "children" not in summary


def _flatten(node):
    yield node
    for child in node.get("children", []):
        yield from _flatten(child)


def test_missing_trace_is_a_404(client):
    assert client.get("/traces/trace_doesnotexist").status_code == 404


def test_scores_endpoint_rolls_up_online_scoring(client):
    client.post(
        "/query",
        json={"query": "refund policy for EU orders", "user_id": "a", "tenant_id": "acme"},
    )
    body = client.get("/scores").json()

    assert "groundedness" in body["scorers"]
    assert body["sample_rate"] == 1.0
    assert body["results"]["groundedness"]["pass"] + body["results"]["groundedness"]["fail"] == 1
    assert 0.0 <= body["results"]["groundedness"]["pass_rate"] <= 1.0


def test_metrics_endpoint_includes_histogram_percentiles(client):
    client.post("/query", json={"query": "refund policy", "user_id": "a", "tenant_id": "b"})
    body = client.get("/metrics").json()

    assert body["traces_buffered"] >= 1
    latency = next(k for k in body["histograms"] if k.startswith("request.latency.seconds"))
    assert body["histograms"][latency]["p95"] is not None
