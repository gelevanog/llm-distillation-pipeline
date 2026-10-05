from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from distillery.config import ConfigError, ModelConfig, PipelineConfig, Settings, load_config
from distillery.providers.base import BudgetExceededError, ChatRequest, ProviderError, RetryableError
from distillery.providers.client import CallLedger, DiskCache, LLMClient, Throttle
from distillery.providers.factory import ClientFactory, build_provider
from distillery.providers.fake import FakeProvider, write_fake_ticket
from distillery.providers.openai_compat import OpenAICompatibleProvider
from tests.scripted import ScriptedProvider

REQUEST = ChatRequest(task="label", system="sys", user="user", json_schema={"type": "object"}, schema_name="s")


def _openrouter(model: str, *fallbacks: str) -> dict[str, Any]:
    return {"provider": "openrouter", "model": model, "fallback_models": list(fallbacks)}


# ---------- Free-only guard ----------


def test_free_guard_accepts_free_models() -> None:
    config = PipelineConfig.model_validate(
        {
            "teacher": {
                "require_free_models": True,
                "generator": _openrouter("qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"),
                "labelers": [_openrouter("nvidia/nemotron-3-super-120b-a12b:free")],
            }
        }
    )
    assert config.teacher.require_free_models


@pytest.mark.parametrize(
    "labeler",
    [
        _openrouter("anthropic/claude-sonnet-5"),
        _openrouter("nvidia/nemotron-3-super-120b-a12b:free", "openai/gpt-5.4-mini"),
        {"provider": "anthropic", "model": "claude-sonnet-5"},
    ],
)
def test_free_guard_rejects_paid_models(labeler: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="require_free_models"):
        PipelineConfig.model_validate({"teacher": {"require_free_models": True, "labelers": [labeler]}})


def test_free_guard_can_be_turned_off_for_clients() -> None:
    config = PipelineConfig.model_validate({"teacher": {"labelers": [_openrouter("anthropic/claude-sonnet-5")]}})
    assert not config.teacher.require_free_models


def test_shipped_configs_load_and_real_config_is_free_only() -> None:
    demo = load_config("configs/demo.yaml")
    real = load_config("configs/openrouter-free.yaml")
    assert demo.student.backend == "fake"
    assert real.teacher.require_free_models
    assert all(model.endswith(":free") for cfg in real.all_teacher_models() for model in cfg.all_models)


def test_openrouter_provider_refuses_paid_model_and_missing_key() -> None:
    with pytest.raises(ConfigError):
        OpenAICompatibleProvider(
            ModelConfig(**_openrouter("openai/gpt-5.4-mini")), kind="openrouter", api_key="k", require_free=True
        )
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        OpenAICompatibleProvider(ModelConfig(**_openrouter("x:free")), kind="openrouter", api_key=None)


class _FakeCompletions:
    def __init__(self, served_model: str, content: str | None = '{"items": []}') -> None:
        self.served_model = served_model
        self.content = content
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            model=self.served_model,
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=34),
        )


def _patched_provider(
    served_model: str, content: str | None = '{"items": []}'
) -> tuple[OpenAICompatibleProvider, _FakeCompletions]:
    config = ModelConfig(
        **_openrouter("nvidia/nemotron-3-super-120b-a12b:free", "qwen/qwen3.8-27b:free"), reasoning_effort="low"
    )
    provider = OpenAICompatibleProvider(config, kind="openrouter", api_key="test-key", require_free=True)
    completions = _FakeCompletions(served_model, content)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return provider, completions


def test_openrouter_request_shape() -> None:
    provider, completions = _patched_provider("qwen/qwen3.8-27b:free")
    response = asyncio.run(provider.complete(REQUEST))
    assert response.model == "qwen/qwen3.8-27b:free"
    assert (response.input_tokens, response.output_tokens) == (12, 34)
    extra = completions.kwargs["extra_body"]
    assert extra["models"] == ["nvidia/nemotron-3-super-120b-a12b:free", "qwen/qwen3.8-27b:free"]
    assert extra["reasoning"]["effort"] == "low"
    assert completions.kwargs["response_format"]["json_schema"]["strict"] is True


def test_openrouter_refuses_answer_from_non_free_model() -> None:
    provider, _ = _patched_provider("nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(ProviderError, match="non-free"):
        asyncio.run(provider.complete(REQUEST))


def test_empty_content_is_retryable() -> None:
    provider, _ = _patched_provider("qwen/qwen3.8-27b:free", content="")
    with pytest.raises(RetryableError):
        asyncio.run(provider.complete(REQUEST))


# ---------- Client: cache, retries, budget, ledger ----------


def test_client_retries_then_caches(tmp_path: Path) -> None:
    provider = ScriptedProvider([RetryableError("429 rate limited", retry_after=0.01), '{"ok": true}'])
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=10)
    client = LLMClient(provider, ledger=ledger, cache=DiskCache(tmp_path / "cache"), max_retries=2)
    first = asyncio.run(client.complete(REQUEST))
    second = asyncio.run(client.complete(REQUEST))
    assert first.text == second.text == '{"ok": true}'
    assert not first.cached and second.cached
    assert len(provider.requests) == 2  # one failure + one success; the repeat came from the cache
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["retryable_error", "ok"]


def test_client_does_not_retry_permanent_errors() -> None:
    provider = ScriptedProvider([ProviderError("400 bad request"), '{"never": "reached"}'])
    client = LLMClient(provider, ledger=CallLedger(None, 10), max_retries=3)
    with pytest.raises(ProviderError, match="400"):
        asyncio.run(client.complete(REQUEST))
    assert len(provider.requests) == 1


def test_budget_counts_previous_invocations(tmp_path: Path) -> None:
    ledger_path = tmp_path / "calls.jsonl"
    ledger_path.write_text('{"status": "ok"}\n{"status": "ok"}\n')
    client = LLMClient(ScriptedProvider(['{"a": 1}']), ledger=CallLedger(ledger_path, max_calls=2), max_retries=0)
    with pytest.raises(BudgetExceededError):
        asyncio.run(client.complete(REQUEST))


def test_fake_provider_is_not_budgeted_or_ledgered(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=0)
    client = LLMClient(FakeProvider(ModelConfig()), ledger=ledger)
    request = REQUEST.model_copy(update={"payload": [{"id": "1", "text": "Where is my order BL-123456?"}]})
    response = asyncio.run(client.complete(request))
    assert json.loads(response.text)["items"][0]["order_id"] == "BL-123456"
    assert not (tmp_path / "calls.jsonl").exists()


def test_throttle_spaces_request_starts() -> None:
    async def run() -> float:
        throttle = Throttle(0.05)
        started = time.monotonic()
        await asyncio.gather(*(throttle.wait() for _ in range(3)))
        return time.monotonic() - started

    assert asyncio.run(run()) >= 0.09


def test_factory_builds_each_provider(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, openrouter_api_key="k", openai_api_key="k", anthropic_api_key="k")  # type: ignore[call-arg]
    for provider, model in [
        ("fake", "f"),
        ("openrouter", "a/b:free"),
        ("openai", "gpt-5-mini"),
        ("anthropic", "claude-sonnet-5"),
    ]:
        built = build_provider(ModelConfig(provider=provider, model=model), settings)  # type: ignore[arg-type]
        assert built.label.endswith(model)
    config = PipelineConfig.model_validate({"teacher": {"cache_dir": str(tmp_path)}})
    factory = ClientFactory(config, settings, ledger_path=None)
    assert factory.client(config.teacher.labelers[0], stage="t").cache is None  # fake: no cache


def test_fake_ticket_writer_is_deterministic() -> None:
    seed = {
        "id": "s1",
        "topic": "refund_request",
        "product": "camera",
        "tone": "angry",
        "length": "medium",
        "style": "several typos",
        "order_number": True,
        "contact_details": True,
    }
    assert write_fake_ticket(seed) == write_fake_ticket(dict(seed))
    assert "@example.com" in write_fake_ticket(seed)


def test_content_filter_is_not_retried() -> None:
    provider, completions = _patched_provider("qwen/qwen3.8-27b:free", content="")

    async def blocked(**kwargs: Any) -> Any:
        message = SimpleNamespace(content="")
        return SimpleNamespace(
            model="qwen/qwen3.8-27b:free",
            choices=[SimpleNamespace(message=message, finish_reason="content_filter")],
            usage=None,
        )

    completions.create = blocked  # type: ignore[method-assign]
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.complete(REQUEST))
    assert not isinstance(info.value, RetryableError)
