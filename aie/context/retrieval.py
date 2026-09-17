"""Retrieval: the Retriever contract plus a dependency-free keyword implementation.

The keyword retriever is not a toy stand-in for embeddings -- BM25-style
lexical search is a genuinely strong baseline, and it is the thing a vector
index should have to beat before you take on the operational cost of one.
Swapping it out means implementing ``Retriever`` and changing one constructor.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from aie.store.memory import DocumentStore
from aie.types import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "of", "on", "or", "our", "that", "the",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "you",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, *, tenant_id: str, k: int = 4) -> list[RetrievedChunk]: ...


class KeywordRetriever:
    """BM25 lexical retrieval over the document store."""

    name = "keyword"

    def __init__(self, store: DocumentStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._store = store
        self._k1 = k1
        self._b = b

    def retrieve(self, query: str, *, tenant_id: str, k: int = 4) -> list[RetrievedChunk]:
        # Permission filtering happens here, before scoring -- never after.
        documents = self._store.all_visible(tenant_id)
        if not documents:
            return []

        terms = tokenize(query)
        if not terms:
            return []

        tokenized = [tokenize(d.text) for d in documents]
        lengths = [len(t) for t in tokenized]
        avg_length = sum(lengths) / len(lengths) or 1.0
        doc_frequency = Counter(term for tokens in tokenized for term in set(tokens))
        total = len(documents)

        scored: list[tuple[float, RetrievedChunk]] = []
        for document, tokens, length in zip(documents, tokenized, lengths):
            counts = Counter(tokens)
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1 + (total - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / avg_length
                )
                score += idf * (frequency * (self._k1 + 1)) / denominator
            if score > 0:
                scored.append(
                    (score, RetrievedChunk(document.id, document.text, score, document.source))
                )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]
