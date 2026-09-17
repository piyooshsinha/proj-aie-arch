"""Routing, retry, fallback and cost accounting."""

from __future__ import annotations

import pytest

from aie.gateway.catalog import ANTHROPIC_MODELS, ModelCatalog, Route, local_model
from aie.gateway.gateway import GatewayError, ModelGateway
from aie.gateway.providers.base import ProviderError, ProviderRefusal
from aie.gateway.providers.echo import EchoProvider
from aie.observe.trace import Trace
from aie.types import GenerationRequest, GenerationResult, Message

MESSAGES = [Message("user", "hello")]


def catalog_with_fallback() -> ModelCatalog:
    catalog = ModelCatalog()
    catalog.register(
        Route("default", ANTHROPIC_MODELS["claude-opus-5"], [local_model("llama3.1:8b")])
    )
    return catalog


def gateway(primary, fallback=None, **kwargs) -> ModelGateway:
    return ModelGateway(
        catalog_with_fallback(),
        {"anthropic": primary, "local": fallback or EchoProvider("fallback answer")},
        base_backoff_s=0.0,
        **kwargs,
    )


async def test_retryable_failure_is_retried_on_the_same_model():
    primary = EchoProvider("recovered", fail_times=1)
    result = await gateway(primary).generate(MESSAGES)

    assert result.text == "recovered"
    assert primary.calls == 2


async def test_exhausted_retries_fall_back_to_the_next_model():
    primary = EchoProvider(fail_times=99)
    trace = Trace()
    result = await gateway(primary).generate(MESSAGES, trace=trace)

    assert result.text == "fallback answer"
    assert primary.calls == 2  # max_attempts_per_model
    attempts = [c["attributes"] for c in trace.to_dict()["children"]]
    assert [a["model"] for a in attempts] == [
        "claude-opus-5", "claude-opus-5", "llama3.1:8b",
    ]


async def test_fatal_failure_does_not_try_the_fallback():
    primary = EchoProvider(fail_times=1, retryable=False)
    primary_fatal = EchoProvider()

    class Fatal:
        name = "fatal"
        calls = 0

        async def generate(self, request):
            raise ProviderError("bad request", retryable=False, fatal=True)

        async def aclose(self):
            return None

    fallback = EchoProvider("should not be reached")
    with pytest.raises(GatewayError, match="fatally"):
        await gateway(Fatal(), fallback).generate(MESSAGES)
    assert fallback.calls == 0


async def test_a_refusal_is_not_routed_around():
    class Refuser:
        name = "refuser"

        async def generate(self, request):
            raise ProviderRefusal("declined", category="cyber")

        async def aclose(self):
            return None

    fallback = EchoProvider("different model, same question")
    with pytest.raises(GatewayError, match="declined"):
        await gateway(Refuser(), fallback).generate(MESSAGES)
    assert fallback.calls == 0, "retrying a refusal on another model defeats the refusal"


async def test_cost_is_computed_from_the_catalog_and_added_to_the_trace():
    class Fixed:
        name = "fixed"

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                text="hi", model=request.model, provider=self.name,
                input_tokens=1_000_000, output_tokens=1_000_000,
            )

        async def aclose(self):
            return None

    trace = Trace()
    result = await gateway(Fixed()).generate(MESSAGES, trace=trace)

    # Opus 5: $5/MTok in, $25/MTok out.
    assert result.cost_usd == pytest.approx(30.0)
    assert trace.cost_usd == pytest.approx(30.0)


async def test_local_models_cost_nothing():
    catalog = ModelCatalog()
    catalog.register(Route("local", local_model("llama3.1:8b")))
    gw = ModelGateway(catalog, {"local": EchoProvider("free")})

    result = await gw.generate(MESSAGES, route="local")
    assert result.cost_usd == 0.0


async def test_unknown_route_is_rejected():
    with pytest.raises(KeyError, match="no route named"):
        await gateway(EchoProvider()).generate(MESSAGES, route="nonexistent")
