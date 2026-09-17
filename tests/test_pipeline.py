"""End-to-end behaviour of the request path."""

from __future__ import annotations

import pytest

from aie.gateway.providers.echo import EchoProvider
from aie.types import Query
from tests.conftest import ScriptedProvider, make_platform


def q(text: str, **kwargs) -> Query:
    return Query(
        text=text,
        user_id=kwargs.pop("user_id", "alice"),
        tenant_id=kwargs.pop("tenant_id", "acme"),
        **kwargs,
    )


async def test_happy_path_retrieves_and_answers(echo_platform):
    result = await echo_platform.pipeline.run(q("What is the refund policy for EU orders?"))

    assert not result.blocked
    assert not result.cached
    assert result.iterations == 1
    assert result.text == "Refunds take 14 days [policy#d1]."
    assert result.trace_id.startswith("trace_")


async def test_retrieved_context_reaches_the_model(settings):
    provider = ScriptedProvider(["ok"])
    platform = make_platform(settings, provider)

    await platform.pipeline.run(q("refund policy for EU orders"))

    prompt = provider.last_prompt
    assert "<retrieved_context>" in prompt
    assert "14 days for EU orders" in prompt
    # The higher-scoring EU passage must outrank the US one.
    assert prompt.index("policy#d1") < prompt.index("policy#d2")


async def test_second_identical_query_is_served_from_cache(settings):
    provider = ScriptedProvider(["first answer", "second answer"])
    platform = make_platform(settings, provider)

    first = await platform.pipeline.run(q("When does the Berlin office open?"))
    second = await platform.pipeline.run(q("when does the BERLIN office   open?"))

    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text == "first answer"
    # The cache sits in front of the model: only one generation happened.
    assert len(provider.requests) == 1


async def test_history_makes_follow_ups_self_contained(settings):
    provider = ScriptedProvider(["14 days.", "Yes, 14 days."])
    platform = make_platform(settings, provider)

    await platform.pipeline.run(q("What is the refund policy for EU orders?", session_id="s1"))
    await platform.pipeline.run(q("how long does it take?", session_id="s1"))

    assert "refund policy for EU orders" in provider.last_prompt


async def test_gateway_outage_surfaces_as_an_error(settings):
    platform = make_platform(settings, EchoProvider(fail_times=99))

    with pytest.raises(Exception) as excinfo:
        await platform.pipeline.run(q("anything at all"))
    assert "every candidate failed" in str(excinfo.value)
