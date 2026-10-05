"""Build providers and clients from config + environment."""

from __future__ import annotations

import asyncio
from pathlib import Path

from distillery.config import ModelConfig, PipelineConfig, Settings
from distillery.providers.base import LLMProvider
from distillery.providers.client import CallLedger, DiskCache, LLMClient, Throttle


def build_provider(model_config: ModelConfig, settings: Settings, *, require_free: bool = False) -> LLMProvider:
    match model_config.provider:
        case "fake":
            from distillery.providers.fake import FakeProvider

            return FakeProvider(model_config)
        case "openrouter":
            from distillery.providers.openai_compat import OpenAICompatibleProvider

            return OpenAICompatibleProvider(
                model_config,
                kind="openrouter",
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                require_free=require_free,
            )
        case "openai":
            from distillery.providers.openai_compat import OpenAICompatibleProvider

            return OpenAICompatibleProvider(
                model_config, kind="openai", api_key=settings.openai_api_key, base_url=settings.openai_base_url
            )
        case "anthropic":
            from distillery.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(model_config, api_key=settings.anthropic_api_key)


class ClientFactory:
    """Shares one throttle, one concurrency limit and one call budget across every client of a run."""

    def __init__(self, config: PipelineConfig, settings: Settings, ledger_path: Path | None) -> None:
        self.config = config
        self.settings = settings
        teacher = config.teacher
        self.ledger = CallLedger(ledger_path, teacher.max_calls)
        self.throttle = Throttle(teacher.min_seconds_between_requests)
        self.semaphore = asyncio.Semaphore(max(1, teacher.concurrency))
        self.cache = DiskCache(teacher.cache_dir)

    def client(self, model_config: ModelConfig, stage: str) -> LLMClient:
        provider = build_provider(model_config, self.settings, require_free=self.config.teacher.require_free_models)
        return LLMClient(
            provider,
            ledger=self.ledger,
            throttle=self.throttle,
            cache=self.cache if provider.is_remote else None,
            semaphore=self.semaphore,
            max_retries=self.config.teacher.max_retries,
            retry_base_seconds=self.config.teacher.retry_base_seconds,
            stage=stage,
        )
