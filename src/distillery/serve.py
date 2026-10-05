"""FastAPI service: `POST /triage` (student first, teacher fallback), the dashboard and the playground."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from distillery import __version__
from distillery.config import PipelineConfig, Settings, load_config
from distillery.dashboard.data import load_view
from distillery.dashboard.render import render
from distillery.io import read_json, read_jsonl
from distillery.label import label_batch
from distillery.logging_config import configure_logging, get_logger
from distillery.providers.client import LLMClient
from distillery.providers.factory import ClientFactory
from distillery.records import GoldItem, RunPaths
from distillery.router import Router
from distillery.schema import TicketTriage
from distillery.student import Student, StudentPrediction, load_student

log = get_logger(__name__)


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000, description="The customer's ticket text")


class TriageResponse(BaseModel):
    triage: TicketTriage | None
    source: str = Field(description="student, teacher, or none (student failed and no teacher configured)")
    reason: str
    student_confidence: float
    student_field_confidence: dict[str, float]
    teacher_model: str | None
    latency_ms: float


class Runtime:
    """Lazily loads the student (a model load can take seconds) and wires the router."""

    def __init__(self, settings: Settings, config: PipelineConfig) -> None:
        self.settings = settings
        self.config = config
        self.run_dir = settings.distillery_run_dir or config.output_dir
        self.paths = RunPaths(root=self.run_dir, dataset_override=config.dataset_dir)
        self.backend = settings.student_backend or config.student.backend
        self.adapter = settings.student_adapter_path or self.paths.adapter_dir
        self.threshold = self._threshold()
        self.factory = ClientFactory(config, settings, ledger_path=None)
        self.teacher: LLMClient | None = None
        self.teacher_error: str | None = None
        if settings.router_teacher_enabled:
            try:
                self.teacher = self.factory.client(config.teacher.labelers[config.eval.teacher_index], stage="serve")
            except Exception as exc:  # missing key etc.: serve the student alone and say why
                self.teacher_error = str(exc)
                log.warning("serve.teacher_disabled", error=str(exc))
        self._student: Student | None = None
        self._lock = asyncio.Lock()

    def _threshold(self) -> float:
        """A fixed threshold from the config wins; else the one calibrated for this student in the eval report."""
        setting = self.config.router.min_confidence
        if isinstance(setting, float):
            return setting
        if self.paths.eval_report.exists():
            report = read_json(self.paths.eval_report)
            finetuned = (report.get("systems") or {}).get("student_finetuned") or {}
            calibrated_for_fake = finetuned.get("model") == "fake-student"
            router = report.get("router") or {}
            if "threshold" in router and calibrated_for_fake == (self.backend == "fake"):
                return float(router["threshold"])
        return setting if isinstance(setting, float) else 0.5

    @property
    def student_label(self) -> str:
        if self._student is not None:
            return self._student.label
        return (
            "fake-student" if self.backend == "fake" else f"{self.config.student.base_model}+lora (loads on first use)"
        )

    async def student(self) -> Student:
        async with self._lock:
            if self._student is None:
                adapter = (
                    self.adapter if self.backend == "hf" and (self.adapter / "adapter_config.json").exists() else None
                )
                if self.backend == "hf" and adapter is None:
                    log.warning("serve.no_adapter", path=str(self.adapter), note="serving the base model without LoRA")
                cfg = self.config.student
                self._student = await asyncio.to_thread(
                    load_student,
                    self.backend,
                    base_model=cfg.base_model,
                    adapter_path=adapter,
                    max_new_tokens=cfg.max_new_tokens,
                    device=cfg.device,
                    torch_threads=cfg.torch_threads,
                )
            return self._student

    async def router(self) -> Router:
        return Router(await self.student(), self.teacher, self.threshold)

    def examples(self, limit: int = 12) -> list[GoldItem]:
        path = self.config.eval.gold_path
        if not path.exists():
            return []
        items = read_jsonl(path, GoldItem)
        step = max(1, len(items) // limit)
        return items[::step][:limit]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)
    config = load_config(settings.distillery_config)
    runtime = Runtime(settings, config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info(
            "serve.start",
            run_dir=str(runtime.run_dir),
            student=runtime.backend,
            threshold=runtime.threshold,
            teacher=runtime.teacher.label if runtime.teacher else None,
        )
        yield

    app = FastAPI(
        title="Distillery",
        version=__version__,
        description="Support-ticket triage with a distilled student model and a teacher fallback.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "student_backend": runtime.backend,
            "student": runtime.student_label,
            "teacher": runtime.teacher.label if runtime.teacher else None,
            "threshold": runtime.threshold,
            "run_dir": str(runtime.run_dir),
        }

    @app.post("/triage", response_model=TriageResponse)
    async def triage(body: TriageRequest) -> TriageResponse:
        router = await runtime.router()
        try:
            answer = await router.triage(body.text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"triage failed: {exc}") from exc
        return TriageResponse(
            triage=answer.triage,
            source=answer.source,
            reason=answer.reason
            if not answer.teacher_error
            else f"{answer.reason}; teacher error: {answer.teacher_error}",
            student_confidence=answer.student.confidence,
            student_field_confidence=answer.student.field_confidence,
            teacher_model=answer.teacher_model,
            latency_ms=round(answer.latency_s * 1000, 1),
        )

    @app.get("/api/report")
    async def report() -> dict[str, Any]:
        if not runtime.paths.eval_report.exists():
            raise HTTPException(status_code=404, detail="no eval report in the run directory")
        data: dict[str, Any] = read_json(runtime.paths.eval_report)
        return data

    @app.get("/", response_class=HTMLResponse)
    async def overview() -> str:
        return render(
            "overview.html", view=load_view(runtime.run_dir), config_name=config.name, live=True, page="overview"
        )

    def playground_context(text: str | None = None, result: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "view": None,
            "live": True,
            "page": "playground",
            "examples": runtime.examples(),
            "student_label": runtime.student_label,
            "teacher_label": runtime.teacher.label if runtime.teacher else None,
            "threshold": runtime.threshold,
            "text": text,
            "result": result,
        }

    @app.get("/playground", response_class=HTMLResponse)
    async def playground() -> str:
        return render("playground.html", **playground_context())

    @app.post("/playground", response_class=HTMLResponse)
    async def playground_run(request: Request, text: str = Form(...)) -> str:
        result = await compare(runtime, text)
        context = playground_context(text, result)
        if request.headers.get("HX-Request"):
            return render("partials/result.html", **context)
        return render("playground.html", **context)

    return app


async def compare(runtime: Runtime, text: str) -> dict[str, Any]:
    """Student and teacher on the same ticket (teacher always runs here, unlike in /triage)."""
    router = await runtime.router()
    routed = await router.triage(text)
    student: StudentPrediction = routed.student
    teacher_dict: dict[str, Any] | None = None
    teacher_model: str | None = None
    teacher_latency: float | None = None
    teacher_error: str | None = runtime.teacher_error
    if routed.source == "teacher" and routed.triage is not None:
        teacher_dict = routed.triage.model_dump(mode="json")
        teacher_model = routed.teacher_model
        teacher_latency = routed.latency_s - student.latency_s
    elif runtime.teacher is not None:
        started = time.perf_counter()
        labels, _ = await label_batch(runtime.teacher, [("ticket", text)], max_repair_attempts=1)
        label = labels["ticket"]
        teacher_latency = time.perf_counter() - started
        log.info(
            "playground.teacher", model=label.model, ok=label.triage is not None, latency_s=round(teacher_latency, 2)
        )
        teacher_model = label.model
        teacher_error = label.error
        teacher_dict = label.triage.model_dump(mode="json") if label.triage else None
    return {
        "routed": routed,
        "student": student,
        "teacher_dict": teacher_dict,
        "teacher_model": teacher_model,
        "teacher_latency": teacher_latency,
        "teacher_error": teacher_error,
    }
