"""Local LLM provider, via any OpenAI-compatible chat-completions endpoint.

Works unchanged against Ollama (``http://localhost:11434/v1``), llama.cpp's
server, vLLM and LM Studio -- they all expose ``POST /chat/completions``. This
is raw HTTP on purpose: these are not Anthropic endpoints, so there is no SDK
to prefer, and the surface we use is three fields wide.
"""

from __future__ import annotations

import httpx

from aie.gateway.providers.base import ProviderError
from aie.types import GenerationRequest, GenerationResult, now_ms

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class LocalProvider:
    name = "local"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str = "not-needed",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = request.stop
        if request.response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.response_format},
            }

        started = now_ms()
        try:
            response = await self._client.post(
                "/chat/completions", json=payload, timeout=request.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"local: timeout talking to {self._base_url}: {exc}", retryable=True) from exc
        except httpx.ConnectError as exc:
            # The usual cause is "the runtime isn't running" -- say so plainly.
            raise ProviderError(
                f"local: cannot reach an OpenAI-compatible server at {self._base_url} "
                f"(is Ollama/vLLM running?): {exc}",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"local: transport error: {exc}", retryable=True) from exc

        if response.status_code == 404:
            raise ProviderError(
                f"local: model {request.model!r} not found at {self._base_url} "
                f"(pull it first, e.g. `ollama pull {request.model}`)",
                fatal=True,
            )
        if response.status_code >= 500:
            raise ProviderError(f"local: http {response.status_code}: {response.text[:200]}", retryable=True)
        if response.status_code >= 400:
            raise ProviderError(f"local: http {response.status_code}: {response.text[:200]}", fatal=True)

        latency_ms = now_ms() - started
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason") or "stop"
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"local: unreadable response body: {exc}") from exc

        usage = body.get("usage") or {}
        return GenerationResult(
            text=text,
            model=body.get("model", request.model),
            provider=self.name,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
