"""Offline eval demo: runs the dataset without a model server.

The canned provider answers three of the five cases correctly, invents a
citation on one, and fabricates an answer on another -- so the report shows the
scorers discriminating rather than a wall of green.

    .venv/bin/python examples/run_eval.py
"""

from __future__ import annotations

import asyncio

from aie.config import Settings, build_platform
from aie.eval.runner import load_cases, run_eval
from aie.observe.logging import configure_logging
from aie.store.memory import Document
from aie.types import GenerationRequest, GenerationResult

ANSWERS = {
    "refund": "Refunds for EU orders are issued within 14 days [policy#d1].",
    "us order": "US orders are refunded within 30 days of purchase [policy#d2].",
    "berlin": "The Berlin office opens at 9am on weekdays [handbook#d3].",
    "swallow": "African swallows migrate at roughly 24 miles per hour [ornithology#d99].",
}
FALLBACK = "Our llamas hold a standing meeting about that on Thursdays."


class CannedProvider:
    name = "canned"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = request.messages[-1].content.lower()
        text = next((a for k, a in ANSWERS.items() if k in prompt), FALLBACK)
        return GenerationResult(
            text=text, model=request.model, provider=self.name,
            input_tokens=400, output_tokens=40, latency_ms=120.0,
        )

    async def aclose(self) -> None:
        return None


async def main() -> None:
    configure_logging(json_format=False)
    platform = build_platform(
        Settings(primary="local", local_model="canned"),
        documents=[
            Document("d1", "Refunds are issued within 14 days for EU orders.", "policy"),
            Document("d2", "US orders are refunded within 30 days of purchase.", "policy"),
            Document("d3", "The Berlin office opens at 9am on weekdays.", "handbook"),
        ],
        providers={"local": CannedProvider()},
    )
    report = await run_eval(platform, load_cases("examples/eval_cases.jsonl"))
    print(report.render())


if __name__ == "__main__":
    asyncio.run(main())
