"""Semantic cache -- present, disabled, and documented as to why.

An embedding-similarity cache returns a *different* question's answer whenever
the similarity threshold is slightly too loose, and it fails silently: the user
gets a fluent, confident, wrong answer and nothing in the logs says "cache".
"What is our refund policy for EU orders" and "...for US orders" sit well above
0.95 cosine on most embedding models.

So this ships off by default. Turn it on only behind a measured hit-quality
eval: sample live hits, have a judge score whether the cached answer actually
answers the new question, and pick the threshold from that curve rather than
from a blog post. ``min_similarity`` below is a placeholder, not a
recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("aie.cache.semantic")


@dataclass
class SemanticCacheConfig:
    enabled: bool = False
    min_similarity: float = 0.97
    ttl_s: float = 300.0
    # Set once you have hit-quality numbers; blocks enabling until then.
    hit_quality_eval: str | None = None


@dataclass
class SemanticCache:
    config: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)

    def __post_init__(self) -> None:
        if self.config.enabled and not self.config.hit_quality_eval:
            raise ValueError(
                "refusing to enable the semantic cache without hit_quality_eval set: "
                "a mis-tuned threshold returns another question's answer silently. "
                "Measure the hit quality first, then name the eval here."
            )

    def get(self, _query_embedding: list[float]) -> str | None:
        if not self.config.enabled:
            return None
        raise NotImplementedError(
            "semantic cache lookup is not implemented yet -- wire it to the vector "
            "store once a hit-quality eval exists"
        )

    def set(self, _query_embedding: list[float], _value: str) -> None:
        if not self.config.enabled:
            return
        raise NotImplementedError("semantic cache write is not implemented yet")
