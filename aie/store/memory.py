"""In-memory document and chat-history stores.

Stand-ins for the "Databases" cylinder in the architecture -- documents,
tables, chat history, vectorDB. They implement the interfaces the rest of the
platform depends on so that swapping in Postgres, pgvector or Qdrant is a
constructor change.

Chat history is tenant-and-session scoped at the API, not by convention: an
accidental cross-session read is the same class of bug as the cache leak.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from aie.types import Message


@dataclass
class Document:
    id: str
    text: str
    source: str = "docs"
    # Documents readable only by specific tenants; empty means all tenants.
    tenants: set[str] = field(default_factory=set)

    def visible_to(self, tenant_id: str) -> bool:
        return not self.tenants or tenant_id in self.tenants


class DocumentStore:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self._documents: list[Document] = list(documents or [])
        self._lock = threading.Lock()

    def add(self, document: Document) -> None:
        with self._lock:
            self._documents.append(document)

    def all_visible(self, tenant_id: str) -> list[Document]:
        with self._lock:
            return [d for d in self._documents if d.visible_to(tenant_id)]

    def __len__(self) -> int:
        return len(self._documents)


class ChatHistoryStore:
    def __init__(self, max_turns: int = 10) -> None:
        self._sessions: dict[tuple[str, str], list[Message]] = {}
        self._lock = threading.Lock()
        self.max_turns = max_turns

    def _key(self, tenant_id: str, session_id: str) -> tuple[str, str]:
        return (tenant_id, session_id)

    def get(self, tenant_id: str, session_id: str | None) -> list[Message]:
        if not session_id:
            return []
        with self._lock:
            return list(self._sessions.get(self._key(tenant_id, session_id), []))

    def append(self, tenant_id: str, session_id: str | None, *messages: Message) -> None:
        if not session_id:
            return
        with self._lock:
            key = self._key(tenant_id, session_id)
            history = self._sessions.setdefault(key, [])
            history.extend(messages)
            # Keep the tail: a run-away session must not grow the prompt forever.
            if len(history) > self.max_turns * 2:
                del history[: len(history) - self.max_turns * 2]

    def clear(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(self._key(tenant_id, session_id), None)
