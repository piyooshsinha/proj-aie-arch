"""Permission boundaries: the failures a diagram review is supposed to catch."""

from __future__ import annotations

from aie.types import Query
from tests.conftest import ScriptedProvider, make_platform


async def test_cache_does_not_leak_across_users(settings):
    provider = ScriptedProvider(["alice's answer", "bob's answer"])
    platform = make_platform(settings, provider)

    alice = await platform.pipeline.run(
        Query(text="What are the salary bands?", user_id="alice", tenant_id="acme")
    )
    bob = await platform.pipeline.run(
        Query(text="What are the salary bands?", user_id="bob", tenant_id="acme")
    )

    assert alice.text == "alice's answer"
    assert bob.cached is False, "a cache hit crossed a user boundary"
    assert bob.text == "bob's answer"
    assert len(provider.requests) == 2


async def test_cache_does_not_leak_across_tenants(settings):
    provider = ScriptedProvider(["acme answer", "globex answer"])
    platform = make_platform(settings, provider)

    await platform.pipeline.run(
        Query(text="What is the refund policy?", user_id="u1", tenant_id="acme")
    )
    other = await platform.pipeline.run(
        Query(text="What is the refund policy?", user_id="u1", tenant_id="globex")
    )

    assert other.cached is False
    assert other.text == "globex answer"


async def test_tenant_scoped_cache_still_isolates_tenants(settings):
    settings.cache_scope = "tenant"
    provider = ScriptedProvider(["shared answer", "other tenant answer"])
    platform = make_platform(settings, provider)

    first = await platform.pipeline.run(
        Query(text="office hours?", user_id="alice", tenant_id="acme")
    )
    same_tenant = await platform.pipeline.run(
        Query(text="office hours?", user_id="bob", tenant_id="acme")
    )
    other_tenant = await platform.pipeline.run(
        Query(text="office hours?", user_id="bob", tenant_id="globex")
    )

    assert first.cached is False
    assert same_tenant.cached is True, "tenant scope should share within a tenant"
    assert other_tenant.cached is False, "tenant scope must not share across tenants"


async def test_retrieval_respects_document_permissions(settings):
    provider = ScriptedProvider(["ok", "ok"])
    platform = make_platform(settings, provider)

    await platform.pipeline.run(
        Query(text="engineering salary bands", user_id="u", tenant_id="acme")
    )
    assert "salary bands are confidential" in provider.last_prompt

    await platform.pipeline.run(
        Query(text="engineering salary bands", user_id="u", tenant_id="globex")
    )
    assert "salary bands are confidential" not in provider.last_prompt


async def test_chat_history_is_scoped_to_its_session(settings):
    provider = ScriptedProvider(["a", "b", "c"])
    platform = make_platform(settings, provider)

    await platform.pipeline.run(
        Query(text="What is the refund policy for EU orders?", user_id="u", tenant_id="acme", session_id="s1")
    )
    await platform.pipeline.run(
        Query(text="how long does it take?", user_id="u", tenant_id="acme", session_id="s2")
    )

    # A different session must not inherit s1's question during rewriting.
    assert "refund policy for EU orders" not in provider.last_prompt


async def test_blocked_responses_are_never_cached(settings):
    provider = ScriptedProvider(["Reach me at leaked@corp.com", "a clean answer"])
    platform = make_platform(settings, provider)

    blocked = await platform.pipeline.run(
        Query(text="who handles billing?", user_id="u", tenant_id="acme")
    )
    assert blocked.blocked is True

    retry = await platform.pipeline.run(
        Query(text="who handles billing?", user_id="u", tenant_id="acme")
    )
    assert retry.cached is False, "a refusal was cached and became permanent"
    assert retry.text == "a clean answer"
