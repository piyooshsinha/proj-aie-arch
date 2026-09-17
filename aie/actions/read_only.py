"""Read-only actions -- the safe half of the tool surface.

Everything here can be retried, run in parallel, and cached, because none of it
changes state. That is the whole reason the architecture separates these from
write actions: the safety properties differ, so the plumbing should too.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from aie.context.retrieval import Retriever
from aie.observe.trace import METRICS, Trace
from aie.types import RetrievedChunk


@dataclass
class _Entry:
    value: Any
    expires_at: float


class ActionCache:
    """TTL cache for read-only action results.

    This is the diagram's middle 'Cache' box, and it is a different animal from
    the response cache: keyed by (action, args, tenant), much shorter TTL, and
    it caches *inputs* to generation rather than generation output.
    """

    def __init__(self, ttl_s: float = 60.0) -> None:
        self.ttl_s = ttl_s
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(action: str, tenant_id: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{action}:{tenant_id}:{digest}"

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.expires_at <= time.time():
                if entry is not None:
                    del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = _Entry(value, time.time() + self.ttl_s)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


@dataclass
class ReadOnlyAction:
    name: str
    description: str
    handler: Callable[..., Any]
    cacheable: bool = True


class ReadOnlyActionRegistry:
    def __init__(self, cache: ActionCache | None = None) -> None:
        self._actions: dict[str, ReadOnlyAction] = {}
        self.cache = cache or ActionCache()

    def register(self, action: ReadOnlyAction) -> None:
        self._actions[action.name] = action

    def names(self) -> list[str]:
        return sorted(self._actions)

    def call(
        self, name: str, *, tenant_id: str, trace: Trace | None = None, **kwargs: Any
    ) -> Any:
        action = self._actions.get(name)
        if action is None:
            raise KeyError(f"no read-only action named {name!r}; have {self.names()}")

        payload = repr(sorted(kwargs.items()))
        key = ActionCache.key(name, tenant_id, payload)
        if action.cacheable:
            cached = self.cache.get(key)
            if cached is not None:
                METRICS.incr("action.cache_hit", action=name)
                if trace is not None:
                    with trace.span(f"action.{name}", cached=True):
                        pass
                return cached

        span = trace.span(f"action.{name}", cached=False) if trace is not None else None
        if span is not None:
            with span:
                result = action.handler(tenant_id=tenant_id, **kwargs)
        else:
            result = action.handler(tenant_id=tenant_id, **kwargs)

        if action.cacheable:
            self.cache.set(key, result)
        METRICS.incr("action.call", action=name)
        return result


def vector_search_action(retriever: Retriever) -> ReadOnlyAction:
    def handler(*, tenant_id: str, query: str, k: int = 4) -> list[RetrievedChunk]:
        return retriever.retrieve(query, tenant_id=tenant_id, k=k)

    return ReadOnlyAction(
        name="search_documents",
        description="Search the document corpus for passages relevant to a query.",
        handler=handler,
    )
