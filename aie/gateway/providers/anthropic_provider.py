"""Anthropic Messages API provider.

Uses the official ``anthropic`` SDK (async client). Credentials resolve from the
environment the way the SDK does it -- ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
or an `ant auth login` profile -- so the gateway never holds a key itself.

Server-side refusal fallbacks are enabled by default: if a safety classifier
declines the request, the API routes to a fallback model by refusal category
rather than handing us a dead turn. Set ``refusal_fallbacks=False`` to opt out
and surface the refusal to our own routing layer instead.
"""

from __future__ import annotations

import anthropic

from aie.gateway.providers.base import ProviderError, ProviderRefusal
from aie.types import GenerationRequest, GenerationResult, now_ms

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        client: "anthropic.AsyncAnthropic | None" = None,
        *,
        api_key: str | None = None,
        effort: str | None = None,
        refusal_fallbacks: bool = True,
        max_retries: int = 0,
    ) -> None:
        # max_retries=0: the gateway owns retry policy so that backoff, budget
        # and fallback decisions live in one place and show up in one trace.
        self._client = client or anthropic.AsyncAnthropic(
            **({"api_key": api_key} if api_key else {}),
            max_retries=max_retries,
        )
        self._effort = effort
        self._refusal_fallbacks = refusal_fallbacks

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        if not messages:
            raise ProviderError("request has no user or assistant messages", fatal=True)

        kwargs: dict[str, object] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        output_config: dict[str, object] = {}
        if self._effort:
            output_config["effort"] = self._effort
        if request.response_format:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.response_format,
            }
        if output_config:
            kwargs["output_config"] = output_config

        started = now_ms()
        try:
            if self._refusal_fallbacks:
                response = await self._client.beta.messages.create(
                    betas=[REFUSAL_FALLBACK_BETA],
                    fallbacks="default",
                    timeout=request.timeout_s,
                    **kwargs,
                )
            else:
                response = await self._client.messages.create(
                    timeout=request.timeout_s, **kwargs
                )
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"anthropic: unknown model {request.model!r}: {exc}", fatal=True) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(f"anthropic: rate limited: {exc}", retryable=True) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderError(f"anthropic: timeout: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500 or exc.status_code in (408, 409)
            raise ProviderError(
                f"anthropic: http {exc.status_code}: {exc}",
                retryable=retryable,
                fatal=exc.status_code in (400, 401, 403),
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"anthropic: connection error: {exc}", retryable=True) from exc

        latency_ms = now_ms() - started

        # Guard stop_reason before reading content -- a refused turn is a 200.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise ProviderRefusal(
                f"anthropic: model refused the request ({getattr(details, 'explanation', 'no explanation')})",
                category=getattr(details, "category", None),
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return GenerationResult(
            text=text,
            model=response.model,
            provider=self.name,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=latency_ms,
            finish_reason=getattr(response, "stop_reason", "stop") or "stop",
        )

    async def aclose(self) -> None:
        await self._client.close()
