"""Context construction: query rewriting, retrieval, prompt assembly.

This is the box that changes daily, which is the main argument for keeping the
platform in one language with a fast edit-run loop.

Query rewriting here is deliberately lexical, not model-based. A model-based
rewrite adds a whole extra generation to every request, and on a first pass you
do not yet know whether it earns that latency. Resolving pronouns against the
last turn does most of the work for follow-up questions. When you do promote it
to a model call, it goes through the gateway like everything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aie.actions.read_only import ReadOnlyActionRegistry
from aie.observe.trace import Trace
from aie.store.memory import ChatHistoryStore
from aie.types import Context, Message, Query, RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions using only the passages in <retrieved_context>. "
    "Cite the passage ids you used. If the passages do not contain the answer, "
    "say so plainly instead of guessing."
)

_FOLLOW_UP = re.compile(r"\b(it|that|they|them|this|those|these|he|she)\b", re.I)


@dataclass
class ContextConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    top_k: int = 4
    min_score: float = 0.0
    include_history: bool = True
    rewrite_follow_ups: bool = True
    # Prompt version: changing the system prompt must invalidate cached answers.
    prompt_version: str = "v1"


class ContextConstructor:
    def __init__(
        self,
        actions: ReadOnlyActionRegistry,
        history: ChatHistoryStore,
        config: ContextConfig | None = None,
    ) -> None:
        self._actions = actions
        self._history = history
        self.config = config or ContextConfig()

    def build(
        self,
        query: Query,
        *,
        trace: Trace | None = None,
        repair_notes: list[str] | None = None,
    ) -> Context:
        if trace is None:
            return self._build(query, None, repair_notes)
        with trace.span("context.construct") as span:
            context = self._build(query, trace, repair_notes)
            span.set(
                rewritten=context.rewritten_query != query.text,
                chunks=len(context.chunks),
                history_turns=len(context.history),
            )
            return context

    def _build(
        self, query: Query, trace: Trace | None, repair_notes: list[str] | None
    ) -> Context:
        history = (
            self._history.get(query.tenant_id, query.session_id)
            if self.config.include_history
            else []
        )
        rewritten = self._rewrite(query.text, history)
        chunks = self._retrieve(rewritten, query, trace)
        return Context(
            query=query,
            rewritten_query=rewritten,
            chunks=chunks,
            history=history,
            system_prompt=self.config.system_prompt,
            repair_notes=list(repair_notes or []),
        )

    def _rewrite(self, text: str, history: list[Message]) -> str:
        """Make a follow-up self-contained by prepending the last question.

        Retrieval on "how long does it take?" returns nothing useful; retrieval
        on "Refund policy EU / how long does it take?" returns the right passage.
        """
        if not self.config.rewrite_follow_ups or not history:
            return text
        if not _FOLLOW_UP.search(text) and len(text.split()) > 6:
            return text
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        if not previous:
            return text
        return f"{previous.strip()} / {text.strip()}"

    def _retrieve(
        self, query_text: str, query: Query, trace: Trace | None
    ) -> list[RetrievedChunk]:
        if "search_documents" not in self._actions.names():
            return []
        chunks: list[RetrievedChunk] = self._actions.call(
            "search_documents",
            tenant_id=query.tenant_id,
            trace=trace,
            query=query_text,
            k=self.config.top_k,
        )
        return [c for c in chunks if c.score >= self.config.min_score]
