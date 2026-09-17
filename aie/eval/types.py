"""Shared types for scoring, online and offline.

One sample type for both paths on purpose: a scorer you trust in an offline
eval is worthless if it cannot also run against live traffic, and vice versa.
The only difference between the two is whether ``expected`` is populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aie.types import RetrievedChunk


@dataclass
class EvalSample:
    query: str
    response: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    expected: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    blocked: bool = False
    cached: bool = False
    route: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    name: str
    value: float          # normalized 0..1 where the scorer has a natural range
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "passed": self.passed,
            "detail": self.detail,
        }
