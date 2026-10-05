"""A provider that replays scripted answers (for client/label/router tests)."""

from __future__ import annotations

from collections.abc import Callable

from distillery.providers.base import ChatRequest, ChatResponse, ProviderError

Reply = str | ProviderError | Callable[[ChatRequest], str]


class ScriptedProvider:
    def __init__(self, replies: list[Reply], *, remote: bool = True, model: str = "scripted:free") -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []
        self.remote = remote
        self.model = model

    @property
    def label(self) -> str:
        return f"scripted/{self.model}"

    @property
    def is_remote(self) -> bool:
        return self.remote

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if not self.replies:
            raise ProviderError("no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, ProviderError):
            raise reply
        text = reply(request) if callable(reply) else reply
        return ChatResponse(text=text, model=self.model, latency_s=0.5, input_tokens=100, output_tokens=50)
