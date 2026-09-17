"""Composition root: env -> a wired platform.

Every layer above takes its collaborators by constructor argument, so this is
the only module that knows how the pieces are actually connected. Tests build
their own platforms directly and never touch the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from aie.actions.read_only import ActionCache, ReadOnlyActionRegistry, vector_search_action
from aie.actions.write import WriteActionRegistry
from aie.cache.response import ResponseCache, Scope
from aie.context.construction import ContextConfig, ContextConstructor
from aie.context.retrieval import KeywordRetriever
from aie.gateway.catalog import ANTHROPIC_MODELS, ModelCatalog, Route, local_model
from aie.gateway.gateway import ModelGateway
from aie.gateway.providers.base import Provider
from aie.guardrails.base import GuardrailChain
from aie.guardrails.input import PIIRedaction, PromptInjectionHeuristic
from aie.guardrails.output import NonEmptyOutput, PIILeakCheck, StructuredOutput
from aie.pipeline import Pipeline, PipelineConfig
from aie.store.memory import ChatHistoryStore, Document, DocumentStore
from aie.types import Query


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Which model serves the "default" route: "anthropic" or "local".
    primary: str = field(default_factory=lambda: os.getenv("AIE_PRIMARY", "anthropic"))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("AIE_ANTHROPIC_MODEL", "claude-opus-5")
    )
    local_base_url: str = field(
        default_factory=lambda: os.getenv("AIE_LOCAL_BASE_URL", "http://localhost:11434/v1")
    )
    local_model: str = field(
        default_factory=lambda: os.getenv("AIE_LOCAL_MODEL", "llama3.1:8b")
    )
    effort: str | None = field(default_factory=lambda: os.getenv("AIE_EFFORT") or None)
    cache_scope: Scope = field(
        default_factory=lambda: os.getenv("AIE_CACHE_SCOPE", "user")  # type: ignore[return-value]
    )
    cache_ttl_s: float = field(
        default_factory=lambda: float(os.getenv("AIE_CACHE_TTL_S", "300"))
    )
    cache_enabled: bool = field(default_factory=lambda: _env_bool("AIE_CACHE_ENABLED", True))
    top_k: int = field(default_factory=lambda: int(os.getenv("AIE_TOP_K", "4")))
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("AIE_MAX_ITERATIONS", "2"))
    )

    @property
    def has_anthropic_credentials(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


@dataclass
class Platform:
    """Everything the API layer needs, assembled."""

    pipeline: Pipeline
    gateway: ModelGateway
    documents: DocumentStore
    history: ChatHistoryStore
    cache: ResponseCache
    read_actions: ReadOnlyActionRegistry
    write_actions: WriteActionRegistry
    settings: Settings


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Construct only the providers we can actually reach.

    Importing the Anthropic provider without credentials is fine; constructing
    its client is not, so we skip it and let the catalog fall back to local.
    """
    providers: dict[str, Provider] = {}

    from aie.gateway.providers.local import LocalProvider

    providers["local"] = LocalProvider(settings.local_base_url)

    if settings.has_anthropic_credentials:
        from aie.gateway.providers.anthropic_provider import AnthropicProvider

        providers["anthropic"] = AnthropicProvider(effort=settings.effort)

    return providers


def build_catalog(settings: Settings) -> ModelCatalog:
    anthropic_spec = ANTHROPIC_MODELS.get(
        settings.anthropic_model
    ) or ANTHROPIC_MODELS["claude-opus-5"]
    local_spec = local_model(settings.local_model)

    catalog = ModelCatalog()
    if settings.primary == "local" or not settings.has_anthropic_credentials:
        # No credentials: local is the only thing that can serve a request.
        catalog.register(Route("default", local_spec))
        catalog.register(Route("cheap", local_spec))
    else:
        catalog.register(Route("default", anthropic_spec, [local_spec]))
        catalog.register(
            Route("cheap", ANTHROPIC_MODELS["claude-haiku-4-5"], [local_spec])
        )
    catalog.register(Route("local", local_spec))
    return catalog


def output_guardrails_for(query: Query) -> GuardrailChain:
    """Bind the output chain to this request (the schema comes from the query)."""
    return GuardrailChain(
        [
            NonEmptyOutput(),
            StructuredOutput(query.response_schema),
            PIILeakCheck(),
        ],
        stage="output",
    )


def build_platform(
    settings: Settings | None = None,
    *,
    documents: list[Document] | None = None,
    providers: dict[str, Provider] | None = None,
) -> Platform:
    settings = settings or Settings()

    document_store = DocumentStore(documents or [])
    history = ChatHistoryStore()
    read_actions = ReadOnlyActionRegistry(ActionCache(ttl_s=60.0))
    read_actions.register(vector_search_action(KeywordRetriever(document_store)))

    context = ContextConstructor(
        read_actions, history, ContextConfig(top_k=settings.top_k)
    )
    cache = ResponseCache(
        ttl_s=settings.cache_ttl_s,
        scope=settings.cache_scope,
        prompt_version=context.config.prompt_version,
        enabled=settings.cache_enabled,
    )
    gateway = ModelGateway(build_catalog(settings), providers or build_providers(settings))

    pipeline = Pipeline(
        gateway=gateway,
        context=context,
        input_guardrails=GuardrailChain(
            [PIIRedaction(), PromptInjectionHeuristic()], stage="input"
        ),
        output_guardrails_for=output_guardrails_for,
        cache=cache,
        history=history,
        config=PipelineConfig(max_iterations=settings.max_iterations),
    )

    return Platform(
        pipeline=pipeline,
        gateway=gateway,
        documents=document_store,
        history=history,
        cache=cache,
        read_actions=read_actions,
        write_actions=WriteActionRegistry(),
        settings=settings,
    )
