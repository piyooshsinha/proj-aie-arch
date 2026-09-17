"""HTTP surface.

Thin on purpose: it validates input, calls the pipeline, and shapes the
response. No business logic lives here, which is what keeps the platform
usable from a worker, a CLI or a test without a web server.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from aie.config import Platform, build_platform
from aie.gateway.gateway import GatewayError
from aie.observe.trace import METRICS
from aie.store.memory import Document
from aie.types import Query

logger = logging.getLogger("aie.api")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    session_id: str | None = None
    route: str = "default"
    response_schema: dict[str, Any] | None = None


class GuardrailView(BaseModel):
    name: str
    action: str
    findings: list[str] = []
    message: str | None = None


class QueryResponse(BaseModel):
    text: str
    trace_id: str
    cached: bool
    blocked: bool
    model: str | None = None
    provider: str | None = None
    iterations: int
    cost_usd: float
    latency_ms: float
    guardrails: list[GuardrailView]


class DocumentRequest(BaseModel):
    id: str
    text: str
    source: str = "docs"
    tenants: list[str] = []


class WriteProposalRequest(BaseModel):
    action: str
    tenant_id: str
    user_id: str
    arguments: dict[str, Any] = {}
    idempotency_key: str | None = None


def create_app(platform: Platform | None = None) -> FastAPI:
    app = FastAPI(
        title="AIE Platform",
        version="0.1.0",
        description="Reference implementation of the AI engineering platform architecture.",
    )
    state = platform or build_platform()
    app.state.platform = state

    def get_platform() -> Platform:
        return app.state.platform

    @app.get("/health")
    async def health(p: Platform = Depends(get_platform)) -> dict[str, Any]:
        return {
            "status": "ok",
            "routes": p.gateway.catalog.route_names(),
            "documents": len(p.documents),
            "anthropic_credentials": p.settings.has_anthropic_credentials,
            "local_base_url": p.settings.local_base_url,
        }

    @app.get("/routes")
    async def routes(p: Platform = Depends(get_platform)) -> dict[str, Any]:
        return p.gateway.catalog.describe()

    @app.get("/metrics")
    async def metrics(p: Platform = Depends(get_platform)) -> dict[str, Any]:
        return {
            "counters": METRICS.snapshot(),
            "response_cache": {
                "hits": p.cache.hits,
                "misses": p.cache.misses,
                "hit_rate": round(p.cache.hit_rate, 4),
                "scope": p.cache.scope,
            },
            "action_cache": {
                "hits": p.read_actions.cache.hits,
                "misses": p.read_actions.cache.misses,
            },
        }

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest, p: Platform = Depends(get_platform)) -> QueryResponse:
        try:
            parsed = Query(
                text=request.query,
                user_id=request.user_id,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                route=request.route,
                response_schema=request.response_schema,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            result = await p.pipeline.run(parsed)
        except KeyError as exc:  # unknown route
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GatewayError as exc:
            # Every model candidate failed. That is an upstream outage, not a
            # bad request -- say so with a 503 so callers retry correctly.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return QueryResponse(
            text=result.text,
            trace_id=result.trace_id,
            cached=result.cached,
            blocked=result.blocked,
            model=result.model,
            provider=result.provider,
            iterations=result.iterations,
            cost_usd=round(result.cost_usd, 6),
            latency_ms=round(result.latency_ms, 2),
            guardrails=[
                GuardrailView(
                    name=g.name, action=g.action.value, findings=g.findings, message=g.message
                )
                for g in result.guardrails
            ],
        )

    @app.post("/documents", status_code=201)
    async def add_document(
        request: DocumentRequest, p: Platform = Depends(get_platform)
    ) -> dict[str, Any]:
        p.documents.add(
            Document(
                id=request.id,
                text=request.text,
                source=request.source,
                tenants=set(request.tenants),
            )
        )
        # New documents change what retrieval returns, so answers cached
        # against the old corpus are now wrong.
        p.cache.clear()
        p.read_actions.cache.clear()
        return {"id": request.id, "documents": len(p.documents)}

    # --- Write actions: proposed here, never executed by the request path. ---

    @app.get("/writes/pending")
    async def pending_writes(p: Platform = Depends(get_platform)) -> list[dict[str, Any]]:
        return [
            {
                "id": proposal.id,
                "action": proposal.action,
                "preview": proposal.preview,
                "status": proposal.status.value,
                "tenant_id": proposal.tenant_id,
            }
            for proposal in p.write_actions.pending()
        ]

    @app.post("/writes", status_code=201)
    async def propose_write(
        request: WriteProposalRequest, p: Platform = Depends(get_platform)
    ) -> dict[str, Any]:
        try:
            proposal = p.write_actions.propose(
                request.action,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                idempotency_key=request.idempotency_key,
                **request.arguments,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"id": proposal.id, "status": proposal.status.value, "preview": proposal.preview}

    @app.post("/writes/{proposal_id}/approve")
    async def approve_write(
        proposal_id: str, approver: str, p: Platform = Depends(get_platform)
    ) -> dict[str, Any]:
        try:
            p.write_actions.approve(proposal_id, approver=approver)
            proposal = p.write_actions.execute(proposal_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": proposal.id, "status": proposal.status.value, "result": proposal.result}

    return app


app = create_app()
