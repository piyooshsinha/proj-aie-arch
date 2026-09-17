"""Model catalog: logical route names -> concrete provider models.

Callers ask for ``"default"`` or ``"cheap"``, never for a provider model id.
That indirection is the whole point of the catalog -- swapping which model
serves a route is a config change, not a code change, and cost accounting has
one place to live.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    # USD per million tokens. Local models are free to run, hence 0.0.
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    max_output_tokens: int = 16000

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


@dataclass
class Route:
    """A logical route: a primary model plus ordered fallbacks."""

    name: str
    primary: ModelSpec
    fallbacks: list[ModelSpec] = field(default_factory=list)

    def candidates(self) -> list[ModelSpec]:
        return [self.primary, *self.fallbacks]


# First-party Anthropic API pricing, USD per million tokens.
ANTHROPIC_MODELS: dict[str, ModelSpec] = {
    "claude-opus-5": ModelSpec("anthropic", "claude-opus-5", 5.00, 25.00, 16000),
    "claude-sonnet-5": ModelSpec("anthropic", "claude-sonnet-5", 2.00, 10.00, 16000),
    "claude-haiku-4-5": ModelSpec("anthropic", "claude-haiku-4-5", 1.00, 5.00, 16000),
}


def local_model(model_id: str, max_output_tokens: int = 4096) -> ModelSpec:
    """A model served by a local runtime (Ollama, llama.cpp, vLLM). No per-token cost."""
    return ModelSpec("local", model_id, 0.0, 0.0, max_output_tokens)


class ModelCatalog:
    def __init__(self, routes: dict[str, Route] | None = None) -> None:
        self._routes: dict[str, Route] = dict(routes or {})

    def register(self, route: Route) -> None:
        self._routes[route.name] = route

    def resolve(self, name: str) -> Route:
        try:
            return self._routes[name]
        except KeyError:
            raise UnknownRoute(
                f"no route named {name!r}; known routes: {sorted(self._routes)}"
            ) from None

    def route_names(self) -> list[str]:
        return sorted(self._routes)

    def describe(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "primary": f"{r.primary.provider}:{r.primary.model_id}",
                "fallbacks": [f"{m.provider}:{m.model_id}" for m in r.fallbacks],
            }
            for name, r in self._routes.items()
        }


class UnknownRoute(KeyError):
    pass
