# proj-aie-arch

A working implementation of the AI engineering platform architecture: a query
path that goes cache → context construction → input guardrails → model gateway →
output guardrails → response, with a bounded repair loop back into context
construction.

One process, module boundaries drawn where a service boundary would eventually
go. Runs against a local LLM or the Anthropic API.

```
                    ┌──────────────────────── observability (aie/observe) ───────────────────────┐
                    │                                                                            │
   query ──▶ response cache ──▶ context construction ──▶ input guardrails ──▶ model gateway ──┐  │
             (aie/cache)         (aie/context)            (aie/guardrails)     (aie/gateway)  │  │
                  │ hit               ▲                                                       ▼  │
                  ▼                   │                                            output guardrails
              response  ◀─────────────┴──────────── RETRY (repair note) ───────────── (aie/guardrails)
                                                                                              │
   read-only actions ──▶ action cache ──▶ (context)          write actions ◀── approval ──────┘
   (aie/actions/read_only.py)                                (aie/actions/write.py, off-path)
```

## Quickstart

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env

.venv/bin/python -m pytest -q          # 76 tests, no network needed
.venv/bin/python -m uvicorn examples.demo:app --port 8099   # offline demo
```

```bash
curl -X POST localhost:8099/query -H 'content-type: application/json' \
  -d '{"query":"refund policy for EU orders","user_id":"alice","tenant_id":"acme"}'
```

To serve real traffic, point `AIE_LOCAL_BASE_URL` at a running Ollama/vLLM or
export `ANTHROPIC_API_KEY`, then:

```bash
.venv/bin/python -m aie --port 8000        # JSON logs; --plain-logs for humans
```

## Where each box lives

| Architecture box   | Module                        | State                                                    |
| ------------------ | ----------------------------- | -------------------------------------------------------- |
| Response cache     | `aie/cache/response.py`       | Built. Exact match, TTL, scope-keyed                      |
| Semantic cache     | `aie/cache/semantic.py`       | **Off by default**, refuses to enable without an eval     |
| Context construction| `aie/context/construction.py` | Built. Lexical query rewriting, retrieval, assembly       |
| Retrieval          | `aie/context/retrieval.py`    | Built. BM25 over the document store                       |
| Read-only actions  | `aie/actions/read_only.py`    | Built. Registry + TTL action cache                        |
| Databases          | `aie/store/memory.py`         | In-memory. Interfaces ready for Postgres/pgvector         |
| Input guardrails   | `aie/guardrails/input.py`     | Built. PII redaction; injection heuristic in shadow mode  |
| Model gateway      | `aie/gateway/`                | Built. Catalog, routing, retry, fallback, cost accounting |
| Output guardrails  | `aie/guardrails/output.py`    | Built. Schema validation, PII leak check, empty check     |
| Write actions      | `aie/actions/write.py`        | Built, deliberately **not** wired to the request path     |
| Observability      | `aie/observe/`                | Built. Traces, metrics, structured logs — see below       |
| Scoring            | `aie/eval/`                   | Built. Online scorers + offline harness                   |

## Where this deviates from the diagram, and why

**An observability plane was added.** The diagram is all request path and has no
telemetry. `aie/observe` traces every layer, so one trace answers where the
latency and the cost went. `Scoring` in the diagram sits inside the gateway,
which conflates online scoring with eval; scoring is not implemented yet and
belongs next to the eval harness, not in the request path.

**The three "Cache" boxes are three different systems.** They are drawn
identically but have different keys, TTLs and failure modes. The response cache
(`cache/response.py`) and the action cache (`actions/read_only.py`) are separate
classes. The semantic cache is a third thing again, and it ships disabled —
a threshold slightly too loose returns another question's answer, fluently and
silently. `SemanticCache` raises if you enable it without naming a hit-quality
eval.

**Cache keys carry an isolation scope.** The diagram puts the cache in front of
permission-scoped retrieval, so a key built from query text alone serves user
A's answer to user B. Keys here always include a scope, defaulting to per-user;
widening to tenant or global is a decision you make, not one you fall into.
`tests/test_isolation.py` holds that line.

**Write actions are a subsystem, not a box.** Approval gate, idempotency keys,
dry run, audit log — and nothing in the pipeline calls them. An agent that can
write is built deliberately.

**Guardrails declare their own policy.** Each one says whether it is `blocking`
(can hold a response) and `fail_closed` (what happens when it throws). A new
check goes in non-blocking, collects shadow data, and gets promoted on evidence.
The injection heuristic is in shadow mode for exactly this reason.

**The feedback arrow carries information.** An output guardrail returning RETRY
appends its failure message as a `<correction>` note and re-enters context
construction. Bounded by `max_iterations` (default 2): a model that has failed a
schema twice will not find it on the third pass.

## Observability

The diagram has no telemetry at all, so this is the largest addition. Three
pieces, all in `aie/observe/`:

**Traces** (`trace.py`). Every layer opens a span. `summary()` attributes time
by layer using *self* time, not wall time — so a slow `context.construct` is
distinguishable from one that is merely waiting on retrieval underneath it:

```json
{"latency_ms": 2840.2, "cost_usd": 0.0135,
 "ms_by_layer": {"gateway": 2790.3, "action": 41.2, "guardrail": 5.1, "cache": 0.7},
 "ms_by_model": {"claude-opus-5": 2790.3},
 "tokens": {"input": 1200, "output": 300}, "generation_attempts": 1}
```

Spans are timed with `perf_counter` (monotonic, immune to clock steps) but
exported against a wall-clock anchor taken at trace start, so they carry both an
accurate duration and a real timestamp. `to_otlp()` emits OpenTelemetry's span
shape, so pointing this at a collector is a new exporter, not a rewrite.

**Metrics** (`metrics.py`). Counters *and* histograms — "what is p95 generation
latency" is not a question a counter can answer at any sampling rate. Latency,
cost and token buckets are tuned for LLM ranges; the Prometheus defaults top out
at 10s, which puts most generations in `+Inf` where they are invisible.

**Logs** (`logging.py`). JSON, with `trace_id` bound through a context variable
so every line emitted during a request — including from libraries that know
nothing about any of this — joins back to its trace.

| Endpoint | What it serves |
| --- | --- |
| `GET /traces` | Recent traces, newest first, with attribution |
| `GET /traces/{id}` | One trace as `tree`, `summary` or `otlp` |
| `GET /metrics` | Counters, histogram percentiles, cache stats |
| `GET /metrics/prometheus` | Scrape endpoint |
| `GET /scores` | Online scoring pass rates |

`TraceStore` is a bounded ring (default 200). An unbounded trace buffer is a
memory leak with a dashboard attached; production sends these to a collector and
keeps the ring for local debugging.

## Scoring and evaluation

The diagram buries `Scoring` inside the model gateway, which conflates online
scoring with eval. Here it is its own layer (`aie/eval/`) with one sample type
shared by both paths — a scorer you trust offline is worthless if it cannot also
run against live traffic.

**Online** (`online.py`) runs cheap heuristics on every response, after the
answer is final. Three invariants, each with a test: a scorer can never change a
response, can never add latency to one, and a scorer that throws produces a
missing measurement rather than a failed request. The model judge is dispatched
as a background task at its own (much lower) sample rate.

**Offline** (`runner.py`) runs a JSONL dataset through the *whole* pipeline —
cache, retrieval, guardrails, gateway — because most regressions in a system
like this come from retrieval or a prompt change, and a harness that bypasses
those layers cannot see them.

```bash
.venv/bin/python examples/run_eval.py                      # offline demo, no model server
.venv/bin/python -m aie.eval.runner cases.jsonl --fail-under 0.9   # CI gate
```

```
  cases      3/5 passed
  groundedness     60.0%  mean 0.45  ############
  citation         80.0%  mean 1.00  ################

  2 failing case(s):
    out-of-scope  'What is the airspeed velocity of an unladen swal'
      -> groundedness(answered substantively with no retrieved context),
         citation(cites ['ornithology#d99'] but nothing was retrieved)
```

Two things the scorers get right that are easy to get wrong:

- **Answering with no retrieved context is a failure, not a free pass.** The
  first version of `Groundedness` passed any answer when retrieval returned
  nothing — which waves through exactly the confident-fabrication case. Now an
  answer that asserts with zero context fails, and one that declines passes.
- **Declining is scored against whether the answer was available.** "I don't
  know" is correct when nothing was retrieved and a miss when the answer was
  sitting in the context.

The response cache is disabled for the duration of an eval run. Leave it on and
the second occurrence of a repeated question returns a cached answer in 0ms,
which quietly turns your eval into a measurement of your cache.

## Design notes

**Why Python.** Every box here is I/O-bound — waiting on a model API or a
database. Rust's throughput advantage is spent on a path dominated by a 1–3s
model call, and would cost two build systems and a network hop. The gateway and
guardrails are the parts that would eventually justify it (narrow interfaces,
CPU-bound scanning, standalone infra), so `Provider` and `Guardrail` are
wire-shaped protocols with no framework types crossing them.

**Retry policy lives in the gateway**, not the provider SDKs (constructed with
`max_retries=0`). One trace shows every attempt, and the retry budget is
enforced across providers rather than per-provider.

**A refusal is not an outage.** `ProviderRefusal` is fatal and never routed
around — trying a different model to get a different answer defeats the point.
Transport failures fall back; refusals propagate.

## Not built yet

Streaming, real vector search, an agent loop over the action registry,
authentication, and durable stores. The interfaces for each are in place; see
the table above.

The scorers are lexical. `Groundedness` is a term-overlap proxy, so a fluent
paraphrase scores lower than a copy-paste and a fabrication that reuses context
vocabulary scores high. What it reliably catches is the answer sharing almost
nothing with what was retrieved. `LLMJudge` exists for the rest and is off by
default because it costs a generation per call and disagrees with itself between
runs.
