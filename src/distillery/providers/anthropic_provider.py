"""Anthropic Messages API with native JSON-schema structured output (`output_config.format`)."""

from __future__ import annotations

import time

import anthropic
from anthropic import AsyncAnthropic

from distillery.config import ModelConfig
from distillery.providers.base import ChatRequest, ChatResponse, ProviderError, RetryableError


class AnthropicProvider:
    def __init__(self, model_config: ModelConfig, *, api_key: str | None) -> None:
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.config = model_config
        # SDK retries are off: LLMClient retries, so every request is visible in the call ledger.
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0, timeout=model_config.timeout_seconds)

    @property
    def label(self) -> str:
        return f"anthropic/{self.config.model}"

    @property
    def is_remote(self) -> bool:
        return True

    async def complete(self, request: ChatRequest) -> ChatResponse:
        started = time.monotonic()
        try:
            # No sampling parameters and no prefill: current Claude models reject them. The schema
            # constrains decoding, so the first text block is the JSON answer.
            message = await self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config={"format": {"type": "json_schema", "schema": request.json_schema}},
            )
        except anthropic.RateLimitError as exc:
            raise RetryableError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in {408, 409, 529}:
                raise RetryableError(f"upstream {exc.status_code}: {exc.message}") from exc
            raise ProviderError(f"{exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RetryableError(f"connection: {exc}") from exc

        if message.stop_reason == "refusal":
            raise ProviderError("the model declined the request (stop_reason=refusal)")
        if message.stop_reason == "max_tokens":
            raise ProviderError(f"output truncated at max_tokens={self.config.max_tokens}")
        text = next((block.text for block in message.content if block.type == "text"), "")
        if not text.strip():
            raise RetryableError("empty content")
        return ChatResponse(
            text=text,
            model=message.model,
            latency_s=round(time.monotonic() - started, 3),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
