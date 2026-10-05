"""Configuration: one YAML file per pipeline run plus environment variables for secrets and serving.

The YAML file (see `configs/`) describes *what* to run: teacher models, dataset size, filters,
student, training and eval settings. API keys never go in YAML; they come from the environment
(or a `.env` file) through `Settings`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["fake", "openai", "anthropic", "openrouter"]


class ConfigError(ValueError):
    """Raised for configurations that must not run (e.g. a paid model under the free-only guard)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    """One teacher model endpoint."""

    provider: ProviderName = "fake"
    model: str = "fake-teacher-a"
    # OpenRouter only: tried in order when the primary model errors or is rate-limited.
    fallback_models: list[str] = Field(default_factory=list)
    max_tokens: int = 8000
    # OpenRouter/OpenAI reasoning effort for reasoning models ("low" keeps free models fast); None = provider default.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    timeout_seconds: float = 180.0

    @property
    def all_models(self) -> list[str]:
        return [self.model, *self.fallback_models]


class TeacherConfig(_Strict):
    generator: ModelConfig = Field(default_factory=ModelConfig)
    # Two labelers = self-consistency check. The same model twice also works (agreement across samples).
    labelers: list[ModelConfig] = Field(
        default_factory=lambda: [ModelConfig(model="fake-teacher-a"), ModelConfig(model="fake-teacher-b")]
    )
    # Refuse any OpenRouter model id that does not end in ":free" (on by default in the demo config).
    require_free_models: bool = False
    min_seconds_between_requests: float = 3.0
    concurrency: int = 3
    max_retries: int = 4
    retry_base_seconds: float = 5.0
    # Hard cap on real (non-cached) API requests per pipeline invocation, retries included.
    max_calls: int = 250
    cache_dir: Path = Path(".cache/llm")

    @model_validator(mode="after")
    def _check(self) -> TeacherConfig:
        if not self.labelers:
            raise ValueError("teacher.labelers needs at least one model")
        return self


class GenerateConfig(_Strict):
    num_tickets: int = 400
    tickets_per_call: int = 12
    # Optional real, unlabeled tickets (CSV with a `text` column, or JSONL with a `text` field).
    import_path: Path | None = None
    synthetic: bool = True


class LabelConfig(_Strict):
    batch_size: int = 12
    max_repair_attempts: int = 1
    # Replace emails/phones with placeholders before tickets are sent to the teacher.
    scrub_pii_before_teacher: bool = True


class FilterConfig(_Strict):
    min_chars: int = 20
    max_chars: int = 1500
    require_agreement: bool = True
    agreement_fields: list[str] = Field(
        default_factory=lambda: ["intent", "urgency", "sentiment", "product_area", "order_id"]
    )
    dedup_threshold: float = 0.7
    dedup_num_perm: int = 128
    dedup_shingle_size: int = 5
    max_per_intent: int = 80
    scrub_pii: bool = True


class BuildConfig(_Strict):
    val_fraction: float = Field(default=0.12, ge=0.0, lt=0.5)


class StudentConfig(_Strict):
    # hf = Hugging Face model + LoRA adapter; fake = deterministic keyword student (no downloads, CI/demo).
    backend: Literal["fake", "hf"] = "hf"
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 120
    batch_size: int = 8
    torch_threads: int | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class TrainConfig(_Strict):
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | Literal["all-linear"] = "all-linear"
    warmup_ratio: float = 0.05
    bf16: bool = False
    gradient_checkpointing: bool = False
    merge_adapter: bool = False
    seed: int = 42


class RouterConfig(_Strict):
    # Confidence threshold below which a student answer goes to the teacher. "auto" = calibrate on the
    # validation split as the lowest threshold whose accepted answers reach `target_exact_match`.
    min_confidence: float | Literal["auto"] = "auto"
    target_exact_match: float = 0.9


class EvalConfig(_Strict):
    gold_path: Path = Path("data/gold/gold.jsonl")
    teacher_batch_size: int = 8
    # Extra single-ticket teacher calls to measure per-request latency and tokens (0 = skip).
    teacher_latency_probe: int = 5
    # Which labeler is "the teacher" in the comparison (index into teacher.labelers).
    teacher_index: int = 0
    limit: int | None = None


class PriceConfig(_Strict):
    name: str
    input_per_million: float
    output_per_million: float
    source: str = ""


class CostConfig(_Strict):
    # API list prices to price the teacher's measured token usage (the free models cost $0).
    api_prices: list[PriceConfig] = Field(default_factory=list)
    # Self-hosted student: hourly price of the machine that ran the measured CPU latency.
    cpu_instance: str = "16 vCPU cloud VM"
    cpu_hourly_usd: float = 0.68
    cpu_parallel_workers: int = 1
    # GPU estimate is an assumption (not measured here): throughput of a batched GPU server.
    gpu_instance: str = "1x NVIDIA L4 (24 GB)"
    gpu_hourly_usd: float = 0.80
    gpu_assumed_items_per_second: float = 20.0


class PipelineConfig(_Strict):
    name: str = "demo"
    output_dir: Path = Path("runs/demo")
    # Read train/val from here instead of <output_dir>/dataset (train/eval only), e.g. a committed dataset.
    dataset_dir: Path | None = None
    seed: int = 42
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    cost: CostConfig = Field(default_factory=CostConfig)

    @model_validator(mode="after")
    def _check_free_models(self) -> PipelineConfig:
        check_free_models(self)
        return self

    def all_teacher_models(self) -> list[ModelConfig]:
        return [self.teacher.generator, *self.teacher.labelers]


def check_free_models(config: PipelineConfig) -> None:
    """The free-only guard: with `require_free_models`, every OpenRouter model id must end in ":free"."""
    if not config.teacher.require_free_models:
        return
    for model_config in config.all_teacher_models():
        if model_config.provider == "openrouter":
            ensure_free(model_config.all_models)
        elif model_config.provider in {"openai", "anthropic"}:
            raise ConfigError(
                f"require_free_models is on, but provider {model_config.provider!r} is always paid; "
                "use openrouter with ':free' models or turn the guard off"
            )


def ensure_free(models: list[str]) -> None:
    paid = [model for model in models if not model.endswith(":free")]
    if paid:
        raise ConfigError(f"require_free_models is on; refusing non-free model ids: {', '.join(paid)}")


def load_config(path: Path | str) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(raw)


class Settings(BaseSettings):
    """Environment: API keys and the serving/dashboard runtime."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Serving + dashboard
    distillery_config: Path = Path("configs/demo.yaml")
    # Run directory whose reports the dashboard shows (defaults to the config's output_dir).
    distillery_run_dir: Path | None = None
    # Override the config's student backend for serving (fake = no downloads; hf = base model + adapter).
    student_backend: Literal["fake", "hf"] | None = None
    # Adapter to serve (defaults to <run dir>/student/adapter).
    student_adapter_path: Path | None = None
    # Answer with the teacher when the student falls back (needs the teacher's API key unless fake).
    router_teacher_enabled: bool = True
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
