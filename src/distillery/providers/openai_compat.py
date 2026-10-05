"""OpenAI and OpenRouter (OpenAI-compatible) chat completions with JSON-schema structured output."""

from __future__ import annotations

import time
from typing import Any, Literal

import openai
from openai import AsyncOpenAI

from distillery.config import ModelConfig, ensure_free
from distillery.providers.base import ChatRequest, ChatResponse, ProviderError, RetryableError

# Optional OpenRouter attribution headers (shown in the OpenRouter dashboard).
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/gelevanog/llm-distillation-pipeline",
    "X-Title": "Distillery",
}


class OpenAICompatibleProvider:
    """`kind="openrouter"` adds the `models` fallback list, reasoning effort and the free-only guard."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        kind: Literal["openai", "openrouter"],
        api_key: str | None,
        base_url: str | None = None,
        require_free: bool = False,
    ) -> None:
        if not api_key:
            env = "OPENROUTER_API_KEY" if kind == "openrouter" else "OPENAI_API_KEY"
            raise ProviderError(f"{env} is not set")
        if kind == "openrouter" and require_free:
            ensure_free(model_config.all_models)
        self.config = model_config
        self.kind = kind
        self.require_free = require_free
        # SDK retries are off: LLMClient retries, so every request is visible in the call ledger.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=model_config.timeout_seconds,
            default_headers=OPENROUTER_HEADERS if kind == "openrouter" else None,
        )

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.config.model}"

    @property
    def is_remote(self) -> bool:
        return True

    def _extra_body(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.kind == "openrouter":
            if self.config.fallback_models:
                extra["models"] = self.config.all_models
            if self.config.reasoning_effort:
                extra["reasoning"] = {"effort": self.config.reasoning_effort, "exclude": True}
            # Only route to upstream providers that honour response_format.
            extra["provider"] = {"require_parameters": True}
        return extra

    async def complete(self, request: ChatRequest) -> ChatResponse:
        started = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "strict": True, "schema": request.json_schema},
            },
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.kind == "openai" and self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise RetryableError(f"rate limited: {_error_text(exc)}", retry_after=_retry_after(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in {408, 409}:
                raise RetryableError(f"upstream {exc.status_code}: {_error_text(exc)}") from exc
            raise ProviderError(f"{exc.status_code}: {_error_text(exc)}") from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise RetryableError(f"connection: {exc}") from exc

        served_model = completion.model or self.config.model
        if self.kind == "openrouter" and self.require_free and not served_model.endswith(":free"):
            # Defence in depth: never accept (or cache) an answer that a paid model produced.
            raise ProviderError(f"OpenRouter served non-free model {served_model!r}; refusing the answer")
        if not completion.choices:
            raise RetryableError("empty response (no choices)")
        choice = completion.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            if choice.finish_reason == "content_filter":
                # The upstream moderation blocks this exact input; retrying the same request will not help.
                raise ProviderError("empty content (finish_reason=content_filter)")
            raise RetryableError(f"empty content (finish_reason={choice.finish_reason})")
        usage = completion.usage
        return ChatResponse(
            text=text,
            model=served_model,
            latency_s=round(time.monotonic() - started, 3),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


def _error_text(exc: openai.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            metadata = error.get("metadata")
            raw = metadata.get("raw") if isinstance(metadata, dict) else None
            return str(raw or error.get("message") or error)[:300]
    return str(exc)[:300]


def _retry_after(exc: openai.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(value) if value else None
    except ValueError:
        return None
