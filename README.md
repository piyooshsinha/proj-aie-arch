# proj-aie-arch

A working implementation of the AI engineering platform architecture — the
layered design where a query flows through a cache, context construction,
guardrails, a model gateway, more guardrails, and back out, with a bounded
repair loop in the middle.

It is a real, runnable system, not a sketch: **76 tests, no network required**,
a serving entry point, an observability plane, and an evaluation harness. It
runs against a local LLM (Ollama, llama.cpp, vLLM, LM Studio) or the Anthropic
API, with the same code either way.

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

---

## Why you would want this

Most RAG code starts as one function that embeds a query, stuffs some passages
into a prompt, and calls an API. That works until you need to answer a question
about it. This layout exists to make those questions answerable.

**Swap models without touching code.** Callers ask for a *route* (`default`,
`cheap`, `local`), never a model id. Changing which model serves a route, or
adding a fallback chain, is configuration. Cost accounting comes free because
the catalog knows the price of every route.

**Survive provider failures.** Retries with jittered backoff, then failover to
the next model in the route, all in one place and all visible in one trace. A
*refusal* is treated differently from an outage and is never routed around —
retrying a declined request on another model to get a different answer defeats
the refusal.

**Don't leak data between users.** Cache keys carry an isolation scope
(per-user by default), retrieval filters by tenant before scoring rather than
after, and chat history is keyed by tenant and session. These are enforced by
tests, not by convention.

**Don't let the model take irreversible actions on its own.** Write actions are
a separate subsystem with an approval gate, idempotency keys, dry runs and an
audit log — and nothing in the request path calls them.

**Know where the time and money went.** Every request produces a trace that
attributes latency by layer and by model, plus cost and token counts. Metrics
include histograms, so p95 is a real number rather than an average you squint at.

**Know whether a change helped.** Scorers run on live traffic and in an offline
harness over a dataset, sharing one sample type. `--fail-under` turns the
harness into a CI gate.

**Develop offline.** The whole test suite and the demo run with no API key and
no model server. Point it at a local model when you want real generations, or
export an API key when you want a frontier one.

---

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/piyooshsinha/proj-aie-arch
cd proj-aie-arch

uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
# or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

cp .env.example .env
```

### Run the tests

```bash
.venv/bin/python -m pytest -q      # 76 passed
```

No network, no API key, no model server. The suite uses an in-process
deterministic provider.

### Try it without a model

```bash
.venv/bin/python -m uvicorn examples.demo:app --port 8099
```

```bash
curl -X POST localhost:8099/query -H 'content-type: application/json' \
  -d '{"query":"How long do refunds take for EU orders?",
       "user_id":"alice","tenant_id":"acme","session_id":"s1"}'
```

```json
{
  "text": "Refunds take 14 days for EU orders [policy#d1].",
  "trace_id": "trace_b1d2c0e08f2b",
  "cached": false,
  "blocked": false,
  "model": "demo",
  "provider": "echo",
  "iterations": 1,
  "cost_usd": 0.0,
  "latency_ms": 0.64,
  "guardrails": [
    {"name": "pii_redaction", "action": "allow", "findings": [], "message": null},
    {"name": "prompt_injection", "action": "allow", "findings": [], "message": null},
    {"name": "non_empty", "action": "allow", "findings": [], "message": null},
    {"name": "structured_output", "action": "allow", "findings": [], "message": null},
    {"name": "pii_leak", "action": "allow", "findings": [], "message": null}
  ]
}
```

### Run it for real

Against a local model:

```bash
ollama serve && ollama pull llama3.1:8b        # in another terminal

export AIE_PRIMARY=local AIE_LOCAL_MODEL=llama3.1:8b
.venv/bin/python -m aie --port 8000
```

Against the Anthropic API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...            # or `ant auth login`
export AIE_PRIMARY=anthropic AIE_ANTHROPIC_MODEL=claude-opus-5
.venv/bin/python -m aie --port 8000
```

With an API key present, the `default` route uses Anthropic and falls back to
your local model if the API is unreachable. With no key, it degrades to
local-only automatically — `GET /health` tells you which state you are in.

```
.venv/bin/python -m aie --help
  --host HOST            default 127.0.0.1
  --port PORT            default 8000
  --log-level LOG_LEVEL  default INFO
  --plain-logs           human-readable logs instead of JSON
```

---

## Configuration

Everything is environment-driven; see `.env.example`. No setting is required —
the defaults produce a working local-only system.

| Variable | Default | What it does |
| --- | --- | --- |
| `AIE_PRIMARY` | `anthropic` | Which provider serves `default`. Falls back to `local` with no credentials |
| `ANTHROPIC_API_KEY` | — | Resolved by the SDK; `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile also work |
| `AIE_ANTHROPIC_MODEL` | `claude-opus-5` | Model for the `default` route |
| `AIE_EFFORT` | unset (`high`) | Thinking depth / token spend: `low`…`max` |
| `AIE_LOCAL_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible server |
| `AIE_LOCAL_MODEL` | `llama3.1:8b` | Local model name |
| `AIE_CACHE_SCOPE` | `user` | `user`, `tenant` or `global` — see [Safety properties](#safety-properties) |
| `AIE_CACHE_TTL_S` | `300` | Response cache TTL |
| `AIE_CACHE_ENABLED` | `true` | Turn the response cache off entirely |
| `AIE_TOP_K` | `4` | Passages retrieved per query |
| `AIE_MAX_ITERATIONS` | `2` | Bound on the guardrail repair loop |
| `AIE_SCORE_SAMPLE_RATE` | `1.0` | Fraction of responses scored by heuristics (free) |
| `AIE_JUDGE_SAMPLE_RATE` | `0.0` | Fraction graded by the model judge (costs a generation each) |

---

## Using it

### HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/query` | Ask a question |
| `POST` | `/documents` | Add a document to the corpus |
| `GET` | `/health` | Wiring, routes, credential state |
| `GET` | `/routes` | Route → model mapping with fallbacks |
| `GET` | `/traces` | Recent traces with cost/latency attribution |
| `GET` | `/traces/{id}` | One trace as `tree`, `summary` or `otlp` |
| `GET` | `/metrics` | Counters, histogram percentiles, cache stats |
| `GET` | `/metrics/prometheus` | Prometheus scrape endpoint |
| `GET` | `/scores` | Online scoring pass rates |
| `POST` | `/writes` | Propose a write action (does **not** execute it) |
| `GET` | `/writes/pending` | Proposals awaiting approval |
| `POST` | `/writes/{id}/approve` | Approve and execute |

Interactive docs at `/docs` once the server is running.

**Asking a question.** `user_id` and `tenant_id` are required — they key the
cache and filter retrieval, so there is no anonymous path that quietly shares
data. `session_id` is optional and enables follow-up questions.

```bash
curl -X POST localhost:8000/query -H 'content-type: application/json' -d '{
  "query": "How long does it take?",
  "user_id": "alice", "tenant_id": "acme", "session_id": "s1",
  "route": "cheap"
}'
```

**Structured output.** Pass a JSON schema and the platform constrains
generation, validates the result, and retries once with the validation error fed
back as a correction note:

```bash
curl -X POST localhost:8000/query -H 'content-type: application/json' -d '{
  "query": "refund policy for EU orders",
  "user_id": "alice", "tenant_id": "acme",
  "response_schema": {
    "type": "object",
    "required": ["answer", "sources"],
    "properties": {
      "answer":  {"type": "string"},
      "sources": {"type": "array", "items": {"type": "string"}}
    }
  }
}'
```

**Adding documents.** This also clears the caches, since new documents change
what retrieval returns and make cached answers stale:

```bash
curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"id":"d9","text":"Parking is free for staff.","source":"handbook"}'

# Tenant-scoped: only `acme` can retrieve this one.
curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"id":"d10","text":"Engineering salary bands.","source":"hr","tenants":["acme"]}'
```

### As a library

The HTTP layer is thin; everything works without it.

```python
import asyncio
from aie.config import Settings, build_platform
from aie.store.memory import Document
from aie.types import Query

platform = build_platform(
    Settings(primary="local", local_model="llama3.1:8b"),
    documents=[
        Document("d1", "Refunds are issued within 14 days for EU orders.", "policy"),
    ],
)

async def main():
    result = await platform.pipeline.run(
        Query(text="refund policy for EU orders", user_id="alice", tenant_id="acme")
    )
    print(result.text, result.cost_usd, result.trace_id)

asyncio.run(main())
```

`build_platform` is the composition root — the only module that knows how the
layers connect. Every layer takes its collaborators as constructor arguments, so
tests build their own platforms and never touch the environment.

### Extending it

Each of these is one class plus one line of wiring.

**A provider** — implement `generate`; no provider types cross the boundary:

```python
from aie.types import GenerationRequest, GenerationResult

class MyProvider:
    name = "mine"
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
    async def aclose(self) -> None: ...

platform = build_platform(providers={"mine": MyProvider()})
```

**A retriever** — swap BM25 for a vector index:

```python
from aie.types import RetrievedChunk

class VectorRetriever:
    name = "vector"
    def retrieve(self, query: str, *, tenant_id: str, k: int = 4) -> list[RetrievedChunk]:
        ...  # filter by tenant_id BEFORE scoring
```

**A guardrail** — declare its own policy:

```python
from aie.types import GuardrailAction, GuardrailResult

class ToxicityCheck:
    name = "toxicity"
    blocking = False     # shadow mode: records, never holds a response
    fail_closed = False  # if it throws, let traffic through

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult:
        return GuardrailResult(self.name, GuardrailAction.ALLOW, content)
```

**A scorer** — the same class runs online and in the offline harness:

```python
from aie.eval.types import EvalSample, Score

class AnswerLength:
    name = "answer_length"
    def score(self, sample: EvalSample) -> Score:
        words = len(sample.response.split())
        return Score(self.name, min(1.0, words / 50), 5 <= words <= 300, f"{words} words")
```

---

## Observability

The reference diagram has no telemetry plane at all. This is the largest
addition, and it lives in `aie/observe/`.

### Traces

Every layer opens a span. `summary()` attributes time by layer using **self**
time rather than wall time, so a slow `context.construct` is distinguishable
from one that is merely waiting on the retrieval beneath it. A request served by
`claude-opus-5` looks like this:

```bash
curl -s "localhost:8000/traces/trace_7f3a91c4e0d2?format=summary"
```

```json
{
  "trace_id": "trace_7f3a91c4e0d2", "tenant_id": "acme", "route": "default",
  "latency_ms": 2840.2, "cost_usd": 0.0135,
  "ms_by_layer": {"gateway": 2790.3, "action": 41.2, "guardrail": 5.1, "cache": 0.7},
  "ms_by_model": {"claude-opus-5": 2790.3},
  "tokens": {"input": 1200, "output": 300},
  "generation_attempts": 1,
  "error": false
}
```

Spans are timed with `perf_counter` (monotonic, immune to clock steps) but
exported against a wall-clock anchor taken at trace start, so each carries both
an accurate duration and a real timestamp. `?format=otlp` emits OpenTelemetry's
span shape — pointing this at a collector is a new exporter, not a rewrite.
`?format=tree` gives the full nested span tree with attributes.

`GET /traces?errors_only=true` narrows to failed requests. The store is a
bounded ring (default 200): an unbounded trace buffer is a memory leak with a
dashboard attached.

### Metrics

Counters *and* histograms — "what is p95 generation latency" is not a question a
counter can answer at any sampling rate. Bucket boundaries are tuned for LLM
ranges; the Prometheus defaults stop at 10s, which buries most generations in
`+Inf` where they are invisible.

```
pipeline_ok{route="default"} 1.0
gateway_latency_seconds_count{model="claude-opus-5",provider="anthropic",route="default"} 1
score_result{outcome="pass",route="default",scorer="groundedness"} 1.0
request_latency_seconds_bucket{le="2.0"} 1
```

`GET /metrics` returns the same data as JSON with p50/p95/p99 computed.

### Logs

JSON, with `trace_id` bound through a context variable — so every line emitted
during a request joins back to its trace, including lines from libraries that
know nothing about any of this.

```json
{"ts": "...", "level": "WARNING", "logger": "aie.eval.online",
 "message": "online scoring flagged a response", "trace_id": "trace_b1d2c0e",
 "tenant_id": "acme", "failed_scorers": ["groundedness"]}
```

Use `--plain-logs` for readable output in development.

---

## Evaluation

The reference diagram buries `Scoring` inside the model gateway, which conflates
online scoring with evaluation. Here it is its own layer (`aie/eval/`), with one
sample type shared by both paths — a scorer you only trust in CI is worthless.

### Scorers

| Scorer | What it catches |
| --- | --- |
| `Groundedness` | Answers sharing almost nothing with the retrieved passages |
| `CitationPresence` | No citation, or a citation to a passage never retrieved |
| `AnswerPresent` | Empty, blocked, or declined-when-the-answer-was-available |
| `LatencySLO` / `CostBudget` | Requests over their budget |
| `KeywordRecall` | Offline: expected content missing from the answer |
| `LLMJudge` | Everything lexical scorers miss — off by default, costs a generation |

### Online

Runs on live traffic after the answer is final. Three invariants, each with a
test: a scorer can never change a response, can never add latency to one, and a
scorer that throws produces a missing measurement rather than a failed request.
The model judge is dispatched as a background task at its own, much lower, rate.

```bash
curl -s localhost:8000/scores
```

```json
{"scorers": ["groundedness", "citation", "answer_present", "latency_slo", "cost_budget"],
 "sample_rate": 1.0, "judge_sample_rate": 0.0,
 "results": {"groundedness": {"pass": 41.0, "fail": 3.0, "pass_rate": 0.9318}}}
```

### Offline

Runs a JSONL dataset through the **whole** pipeline — cache, retrieval,
guardrails, gateway — because most regressions here come from retrieval or a
prompt change, and a harness that bypasses those layers cannot see them.

```bash
.venv/bin/python examples/run_eval.py                    # demo, no model server
.venv/bin/python -m aie.eval.runner cases.jsonl          # against your config
.venv/bin/python -m aie.eval.runner cases.jsonl --json --fail-under 0.9   # CI gate
.venv/bin/python -m aie.eval.runner cases.jsonl --judge  # add the model judge
```

Dataset format, one case per line:

```jsonl
{"id": "refund-eu", "query": "How long do refunds take for EU orders?", "expected_keywords": ["14 days"]}
{"id": "out-of-scope", "query": "What is the airspeed velocity of an unladen swallow?"}
```

```
  cases      3/5 passed
  cost       $0.0000
  latency    p50 0ms  p95 1ms  max 1ms

  groundedness     60.0%  mean 0.45  ############
  citation         80.0%  mean 0.80  ################
  answer_present  100.0%  mean 1.00  ####################

  2 failing case(s):
    out-of-scope  'What is the airspeed velocity of an unladen swal'
      -> groundedness(answered substantively with no retrieved context to support it),
         citation(cites ['ornithology#d99'] but nothing was retrieved)
```

Two things the scorers get right that are easy to get wrong:

- **Answering with no retrieved context is a failure, not a free pass.** The
  first version of `Groundedness` passed any answer when retrieval returned
  nothing — waving through exactly the confident-fabrication case. Running the
  harness against a deliberately-fabricating provider is what exposed it.
- **Declining is scored against whether the answer was available.** "I don't
  know" is correct when nothing was retrieved and a miss when the answer was
  sitting in the retrieved context.

The response cache is disabled for the duration of a run. Leave it on and the
second occurrence of a repeated question returns a cached answer in 0ms, which
quietly turns your eval into a measurement of your cache.

---

## Safety properties

These are the guarantees the layout is *for*, each with a test in
`tests/test_isolation.py`.

**Cache keys carry an isolation scope.** The diagram puts the cache in front of
permission-scoped retrieval, so a key built from query text alone serves user
A's answer to user B. Keys here always include a scope:

| `AIE_CACHE_SCOPE` | Shared between | Use when |
| --- | --- | --- |
| `user` (default) | Nobody | Retrieval is user-permissioned |
| `tenant` | Users in one tenant | Retrieval is tenant-scoped, not per-user |
| `global` | Everyone | The corpus has no access control at all |

Widening the scope is a decision you make, not one you fall into.

**Retrieval filters before it scores.** Documents carry a tenant set; filtering
happens before ranking, never after.

**Blocked responses are never cached or written to history.** Caching a refusal
makes it permanent for that user.

**Writes require approval.** The registry starts empty — nothing can be written
that an operator did not explicitly wire up:

```python
from aie.actions.write import WriteAction

platform.write_actions.register(
    WriteAction(
        name="send_email",
        description="Send an email to a customer",
        handler=lambda *, tenant_id, to, subject: email_client.send(to, subject),
        describe=lambda to, subject: f"Send an email to {to} with subject {subject!r}",
        requires_approval=True,   # the default
    )
)
```

`POST /writes` then returns a proposal with a human-readable preview and
executes nothing. Proposals are idempotent by key, so a replayed approval sends
one email, not two. Every transition is audited. The demo server registers this
action, so the flow below runs against `examples/demo.py` as written:

```bash
curl -X POST localhost:8000/writes -H 'content-type: application/json' \
  -d '{"action":"send_email","tenant_id":"acme","user_id":"alice",
       "arguments":{"to":"bob@example.com","subject":"Refund"}}'
# -> {"id":"write_c0db7dbe4822","status":"proposed",
#     "preview":"Send an email to bob@example.com with subject 'Refund'"}

curl -X POST "localhost:8000/writes/write_c0db7dbe4822/approve?approver=carol"
# -> {"id":"write_c0db7dbe4822","status":"executed","result":"sent"}
```

**PII is redacted before it reaches the model**, and PII in a response that was
not in the request is treated as a leak and blocked.

---

## Where each box lives

| Architecture box | Module | State |
| --- | --- | --- |
| Response cache | `aie/cache/response.py` | Built. Exact match, TTL, scope-keyed |
| Semantic cache | `aie/cache/semantic.py` | **Off by default**, refuses to enable without an eval |
| Context construction | `aie/context/construction.py` | Built. Lexical query rewriting, retrieval, assembly |
| Retrieval | `aie/context/retrieval.py` | Built. BM25 over the document store |
| Read-only actions | `aie/actions/read_only.py` | Built. Registry + TTL action cache |
| Databases | `aie/store/memory.py` | In-memory. Interfaces ready for Postgres/pgvector |
| Input guardrails | `aie/guardrails/input.py` | Built. PII redaction; injection heuristic in shadow mode |
| Model gateway | `aie/gateway/` | Built. Catalog, routing, retry, fallback, cost accounting |
| Output guardrails | `aie/guardrails/output.py` | Built. Schema validation, PII leak check, empty check |
| Write actions | `aie/actions/write.py` | Built, deliberately **not** wired to the request path |
| Observability | `aie/observe/` | Built. Traces, metrics, structured logs |
| Scoring | `aie/eval/` | Built. Online scorers + offline harness |

---

## Where this deviates from the reference diagram, and why

**An observability plane was added.** The diagram is all request path with no
telemetry — the layer you most regret omitting. `Scoring` in the diagram sits
inside the gateway, conflating online scoring with eval; here they are one layer
with two entry points, outside the gateway.

**The three "Cache" boxes are three different systems.** They are drawn
identically but have different keys, TTLs and failure modes. The response cache
and the action cache are separate classes. The semantic cache is a third thing
again and ships disabled — a threshold slightly too loose returns another
question's answer, fluently and silently. `SemanticCache` raises if you enable it
without naming a hit-quality eval.

**Cache keys carry an isolation scope**, as above.

**Write actions are a subsystem, not a box** — approval gate, idempotency keys,
dry run, audit log — and nothing in the pipeline calls them.

**Guardrails declare their own policy.** Each says whether it is `blocking` (can
hold a response) and `fail_closed` (what happens when it throws). A new check
goes in non-blocking, collects shadow data, and gets promoted on evidence.

**The feedback arrow carries information.** An output guardrail returning RETRY
appends its failure message as a `<correction>` note and re-enters context
construction, bounded by `max_iterations` — a model that has failed a schema
twice will not find it on the third pass.

---

## Design notes

**Why Python, and where Rust would fit.** Every box here is I/O-bound — waiting
on a model API or a database. Rust's throughput advantage is spent on a path
dominated by a 1–3s model call, and would cost two build systems and a network
hop. The gateway and guardrails are the parts that would eventually justify it
(narrow interfaces, CPU-bound scanning, standalone infra), so `Provider` and
`Guardrail` are wire-shaped protocols with no framework types crossing them.

**One process, service-shaped seams.** Every box is a module behind an
interface, not a microservice. Eight services on day one is eight deployments
and no product.

**Retry policy lives in the gateway**, not the provider SDKs (constructed with
`max_retries=0`). One trace shows every attempt, and the retry budget is enforced
across providers rather than per-provider.

**A refusal is not an outage.** `ProviderRefusal` is fatal and never routed
around. Transport failures fall back; refusals propagate.

**Anthropic provider specifics.** Uses the official SDK with server-side refusal
fallbacks enabled (`fallbacks: "default"`), so a safety-classifier decline routes
by category instead of returning a dead turn. Set `refusal_fallbacks=False` to
opt out.

---

## Testing

```bash
.venv/bin/python -m pytest -q                        # all 76
.venv/bin/python -m pytest tests/test_isolation.py   # permission boundaries
.venv/bin/python -m pytest -k "cache or leak" -v
```

| File | Tests | Covers |
| --- | --- | --- |
| `test_pipeline.py` | 5 | End-to-end request path, caching, history |
| `test_isolation.py` | 6 | Cross-user/tenant leaks, document permissions |
| `test_guardrails.py` | 7 | Redaction, leak blocking, repair loop bounds |
| `test_gateway.py` | 7 | Retry, fallback, refusals, cost accounting |
| `test_observability.py` | 18 | Span attribution, OTLP, metrics, log correlation |
| `test_eval.py` | 19 | Scorers, online invariants, offline harness |
| `test_api.py` | 14 | HTTP surface including the write-approval flow |

~4,200 lines of implementation, ~1,240 lines of tests.

---

## Not built yet

Streaming, real vector search, an agent loop over the action registry,
authentication, and durable stores. The interfaces for each are in place; see the
module table above.

The scorers are lexical. `Groundedness` is a term-overlap proxy, so a fluent
paraphrase scores lower than a copy-paste and a fabrication that reuses context
vocabulary scores high. What it reliably catches is the answer sharing almost
nothing with what was retrieved. `LLMJudge` exists for the rest and is off by
default because it costs a generation per call and disagrees with itself between
runs.

The stores are in-memory: documents, chat history and caches do not survive a
restart. `DocumentStore`, `ChatHistoryStore` and `ResponseCache` are the three
classes to reimplement against Postgres/pgvector/Redis.
