"""Provider interface: one JSON-producing chat call."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Task = Literal["generate", "label", "repair"]


class ChatRequest(BaseModel):
    """A single structured-output request.

    `payload` is a machine-readable copy of the items in the prompt. Real providers ignore it; the
    offline fake provider answers from it, so the fake run exercises the same prompt -> JSON ->
    validation -> repair path as a real model.
    """

    task: Task
    system: str
    user: str
    json_schema: dict[str, Any]
    schema_name: str
    payload: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    text: str
    model: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class ProviderError(RuntimeError):
    """A request failed and should not be retried (bad request, auth, policy violation)."""


class RetryableError(ProviderError):
    """Rate limit, upstream overload, timeout or an empty reply: worth retrying with backoff."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceededError(ProviderError):
    """The run hit `teacher.max_calls`."""


class LLMProvider(Protocol):
    """Anything that can answer a `ChatRequest` with JSON text."""

    @property
    def label(self) -> str:
        """Provider and model, e.g. "openrouter/nvidia/nemotron-3-super-120b-a12b:free"."""
        ...

    @property
    def is_remote(self) -> bool:
        """True for real APIs (counted against the call budget, throttled and cached)."""
        ...

    async def complete(self, request: ChatRequest) -> ChatResponse: ...
