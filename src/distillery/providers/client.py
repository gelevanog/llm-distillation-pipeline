"""LLMClient: disk cache + throttle + retries with backoff + a hard call budget + a call ledger.

Every real request (retries included) is appended to `calls.jsonl` in the run directory, so the
README numbers about "how many API calls did this cost" come from a file, not from memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.logging_config import get_logger
from distillery.providers.base import (
    BudgetExceededError,
    ChatRequest,
    ChatResponse,
    LLMProvider,
    ProviderError,
    RetryableError,
)

log = get_logger(__name__)


class DiskCache:
    """One JSON file per request hash. Re-running a stage replays answers instead of paying again."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(provider_label: str, request: ChatRequest, extra: dict[str, Any] | None = None) -> str:
        material = json.dumps(
            {
                "provider": provider_label,
                "system": request.system,
                "user": request.user,
                "schema": request.json_schema,
                "extra": extra or {},
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> ChatResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return ChatResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, response: ChatResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.model_dump_json(), encoding="utf-8")


class Throttle:
    """Spaces request *starts* at least `min_interval` seconds apart (shared by all clients of a run)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_start - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start = max(now, self._next_start) + self.min_interval


class CallLedger:
    """Counts real API requests against a hard budget and logs each one to JSONL."""

    def __init__(self, path: Path | None, max_calls: int) -> None:
        self.path = path
        self.max_calls = max_calls
        # The budget covers the whole run directory: requests made by earlier invocations count too.
        self.calls = 0
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as handle:
                self.calls = sum(1 for line in handle if line.strip())

    def reserve(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceededError(f"call budget of {self.max_calls} real requests reached")
        self.calls += 1

    def record(self, **entry: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class LLMClient:
    """Wraps a provider with caching, throttling, retries and budget accounting."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        ledger: CallLedger,
        throttle: Throttle | None = None,
        cache: DiskCache | None = None,
        semaphore: asyncio.Semaphore | None = None,
        max_retries: int = 4,
        retry_base_seconds: float = 5.0,
        stage: str = "",
    ) -> None:
        self.provider = provider
        self.ledger = ledger
        self.throttle = throttle
        self.cache = cache
        self.semaphore = semaphore or asyncio.Semaphore(4)
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.stage = stage

    @property
    def label(self) -> str:
        return self.provider.label

    async def complete(self, request: ChatRequest) -> ChatResponse:
        key = DiskCache.key(self.provider.label, request) if self.cache else ""
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit.model_copy(update={"cached": True})

        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            async with self.semaphore:
                if self.provider.is_remote:
                    self.ledger.reserve()
                    if self.throttle:
                        await self.throttle.wait()
                started = time.monotonic()
                try:
                    response = await self.provider.complete(request)
                except RetryableError as exc:
                    last_error = exc
                    self._record(request, status="retryable_error", error=str(exc), started=started)
                except ProviderError as exc:
                    self._record(request, status="error", error=str(exc), started=started)
                    raise
                else:
                    self._record(request, status="ok", started=started, response=response)
                    if self.cache:
                        self.cache.put(key, response)
                    return response
            if attempt < self.max_retries:
                delay = self._backoff(attempt, last_error)
                log.warning(
                    "llm.retry",
                    provider=self.label,
                    task=request.task,
                    attempt=attempt + 1,
                    delay=round(delay, 1),
                    error=str(last_error)[:200],
                )
                await asyncio.sleep(delay)
        raise last_error or ProviderError("request failed")

    def _backoff(self, attempt: int, error: ProviderError | None) -> float:
        retry_after = error.retry_after if isinstance(error, RetryableError) else None
        if retry_after:
            return min(retry_after, 120.0)
        if not self.provider.is_remote:
            return 0.0
        delay: float = min(self.retry_base_seconds * 2.0**attempt, 90.0) * (0.75 + random.random() / 2)
        return delay

    def _record(
        self,
        request: ChatRequest,
        *,
        status: str,
        started: float,
        error: str | None = None,
        response: ChatResponse | None = None,
    ) -> None:
        if not self.provider.is_remote:
            return
        self.ledger.record(
            stage=self.stage,
            task=request.task,
            provider=self.label,
            served_model=response.model if response else None,
            status=status,
            latency_s=round(time.monotonic() - started, 2),
            input_tokens=response.input_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            items=len(request.payload),
            error=error[:300] if error else None,
        )
