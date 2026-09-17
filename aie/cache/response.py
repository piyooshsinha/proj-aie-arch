"""Exact-match response cache.

The reference diagram draws this as one box in front of context construction,
which hides a real hazard: the cache sits *before* permission-scoped retrieval,
so a key computed from query text alone will serve user A's answer to user B.
The cache key here therefore always includes an isolation scope, and the
default scope is per-user. Widening it to a whole tenant is a deliberate
decision a caller has to make, not the default they fall into.

The key also covers the route and a prompt version, because a cached answer is
only valid for the prompt and model that produced it. Bump ``prompt_version``
whenever the system prompt changes, or you will serve yesterday's behaviour.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Literal

from aie.types import Query

Scope = Literal["user", "tenant", "global"]

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def cache_key(query: Query, *, scope: Scope, prompt_version: str) -> str:
    if scope == "user":
        isolation = f"user:{query.tenant_id}:{query.user_id}"
    elif scope == "tenant":
        isolation = f"tenant:{query.tenant_id}"
    elif scope == "global":
        isolation = "global"
    else:  # pragma: no cover - guarded by the Literal, kept for runtime callers
        raise ValueError(f"unknown cache scope {scope!r}")

    material = "\x1f".join(
        [isolation, query.route, prompt_version, normalize(query.text)]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    value: str
    expires_at: float
    model: str | None = None


class ResponseCache:
    """In-process TTL cache.

    Deliberately small and swappable -- the interface is what matters, and the
    Redis implementation replaces this class without touching a caller.
    """

    def __init__(
        self,
        *,
        ttl_s: float = 300.0,
        scope: Scope = "user",
        prompt_version: str = "v1",
        max_entries: int = 10_000,
        enabled: bool = True,
    ) -> None:
        self.ttl_s = ttl_s
        self.scope: Scope = scope
        self.prompt_version = prompt_version
        self.max_entries = max_entries
        self.enabled = enabled
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def key_for(self, query: Query) -> str:
        return cache_key(query, scope=self.scope, prompt_version=self.prompt_version)

    def get(self, query: Query) -> CacheEntry | None:
        if not self.enabled:
            return None
        key = self.key_for(query)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry

    def set(self, query: Query, value: str, *, model: str | None = None) -> None:
        if not self.enabled:
            return
        key = self.key_for(query)
        with self._lock:
            if len(self._store) >= self.max_entries:
                self._evict_locked()
            self._store[key] = CacheEntry(
                value=value, expires_at=time.time() + self.ttl_s, model=model
            )

    def _evict_locked(self) -> None:
        """Drop expired entries; if none are expired, drop the nearest to expiry."""
        now = time.time()
        expired = [k for k, e in self._store.items() if e.expires_at <= now]
        if expired:
            for k in expired:
                del self._store[k]
            return
        oldest = min(self._store, key=lambda k: self._store[k].expires_at)
        del self._store[oldest]

    def invalidate(self, query: Query) -> None:
        with self._lock:
            self._store.pop(self.key_for(query), None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
