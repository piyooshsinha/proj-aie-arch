"""Core data types shared across the platform layers.

Deliberately stdlib-only: every layer below ``aie.api`` must stay importable
without a web framework installed, so the gateway and guardrails can later be
lifted out of this process without dragging the HTTP stack along.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class Query:
    """A single inbound request, before any processing."""

    text: str
    user_id: str
    tenant_id: str
    session_id: str | None = None
    # Logical model name resolved by the gateway catalog, not a provider id.
    route: str = "default"
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text must not be empty")
        if not self.user_id or not self.tenant_id:
            raise ValueError("query must carry user_id and tenant_id")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str


@dataclass
class Context:
    """Everything assembled for the model, plus how we got there."""

    query: Query
    rewritten_query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    # Feedback appended when an output guardrail sends us back around the loop.
    repair_notes: list[str] = field(default_factory=list)

    def to_messages(self) -> list[Message]:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message("system", self.system_prompt))
        messages.extend(self.history)

        parts: list[str] = []
        if self.chunks:
            retrieved = "\n\n".join(
                f"[{c.source}#{c.id}] {c.text}" for c in self.chunks
            )
            parts.append(f"<retrieved_context>\n{retrieved}\n</retrieved_context>")
        parts.append(self.rewritten_query)
        for note in self.repair_notes:
            parts.append(f"<correction>{note}</correction>")

        messages.append(Message("user", "\n\n".join(parts)))
        return messages


@dataclass
class GenerationRequest:
    messages: list[Message]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.2
    stop: list[str] = field(default_factory=list)
    timeout_s: float = 60.0
    # JSON schema constraining generation, when the provider supports it.
    response_format: dict[str, Any] | None = None


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str = "stop"


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"      # content was rewritten (e.g. PII redacted)
    RETRY = "retry"        # send it back around the loop with feedback
    BLOCK = "block"        # refuse, do not return model output


@dataclass
class GuardrailResult:
    name: str
    action: GuardrailAction
    content: str
    findings: list[str] = field(default_factory=list)
    message: str | None = None

    @property
    def blocked(self) -> bool:
        return self.action is GuardrailAction.BLOCK


@dataclass
class PipelineResponse:
    text: str
    trace_id: str
    cached: bool = False
    model: str | None = None
    provider: str | None = None
    blocked: bool = False
    guardrails: list[GuardrailResult] = field(default_factory=list)
    iterations: int = 1
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    structured: Any | None = None


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> float:
    return time.perf_counter() * 1000.0
