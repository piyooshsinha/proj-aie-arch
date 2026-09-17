from __future__ import annotations

import pytest

from aie.config import Settings, build_platform
from aie.gateway.providers.echo import EchoProvider
from aie.observe.trace import METRICS, TRACES
from aie.store.memory import Document
from aie.types import GenerationRequest, GenerationResult

DOCUMENTS = [
    Document("d1", "Refunds are issued within 14 days for EU orders.", "policy"),
    Document("d2", "US orders are refunded within 30 days of purchase.", "policy"),
    Document("d3", "The Berlin office opens at 9am on weekdays.", "handbook"),
    Document("d4", "Engineering salary bands are confidential.", "hr", tenants={"acme"}),
]


class ScriptedProvider:
    """Returns a pre-set sequence of replies; records what it was sent."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        reply = self._replies.pop(0) if self._replies else "(exhausted)"
        return GenerationResult(
            text=reply, model=request.model, provider=self.name,
            input_tokens=10, output_tokens=5,
        )

    async def aclose(self) -> None:
        return None

    @property
    def last_prompt(self) -> str:
        return self.requests[-1].messages[-1].content


@pytest.fixture(autouse=True)
def reset_observability():
    # METRICS and TRACES are process-global; a leaked trace or counter from one
    # test makes another test's assertions lie.
    METRICS.reset()
    TRACES.clear()
    yield
    METRICS.reset()
    TRACES.clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(primary="local", local_model="test-model", cache_ttl_s=60.0)


def make_platform(settings: Settings, provider):
    return build_platform(settings, documents=list(DOCUMENTS), providers={"local": provider})


@pytest.fixture
def echo_platform(settings):
    return make_platform(settings, EchoProvider("Refunds take 14 days [policy#d1]."))
