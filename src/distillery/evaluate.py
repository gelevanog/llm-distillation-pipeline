"""Stage 6, evaluate: teacher vs student (zero-shot) vs student (fine-tuned) on the hand-labeled gold set.

Scores per system: JSON validity, schema validity, per-field accuracy, macro-F1 for the enum fields,
exact match on all structured fields, a ROUGE-L signal for the free-text summary, latency, and the
router simulation + cost per 1k tickets. Invalid answers count as wrong everywhere.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from distillery.build import record_to_example
from distillery.config import PipelineConfig
from distillery.cost import cost_table, routed_cost_per_1k
from distillery.io import iter_jsonl, read_json, read_jsonl, write_json, write_jsonl
from distillery.label import CallStat, label_with_teacher
from distillery.logging_config import get_logger
from distillery.metrics import accuracy, macro_f1, percentile, rouge_l
from distillery.prompts import zero_shot_system_prompt
from distillery.providers.factory import ClientFactory
from distillery.records import GoldItem, RunPaths
from distillery.router import calibrate_threshold, exact_match, simulate, tradeoff_curve
from distillery.schema import ENUM_FIELDS, STRUCTURED_FIELDS, TicketTriage
from distillery.student import Student, StudentPrediction, load_student, predict_all, resolve_device

log = get_logger(__name__)
INVALID = "__invalid__"


class SystemScores(BaseModel):
    system: str
    model: str
    items: int
    json_valid_rate: float
    schema_valid_rate: float
    exact_match: float
    field_accuracy: dict[str, float]
    macro_f1: dict[str, float]
    summary_rouge_l: float
    latency_s_per_item: dict[str, float]
    latency_note: str = ""
    intent_confusions: list[dict[str, Any]] = Field(default_factory=list)


def score_system(
    system: str,
    model: str,
    gold: Sequence[GoldItem],
    predicted: Sequence[TicketTriage | None],
    json_valid: Sequence[bool],
    latencies: Sequence[float],
    latency_note: str = "",
) -> SystemScores:
    items = len(gold)
    gold_structured = [item.label.structured() for item in gold]
    pred_structured = [answer.structured() if answer else None for answer in predicted]

    def column(rows: Sequence[dict[str, str | None] | None], field: str) -> list[str | None]:
        return [row[field] if row is not None else INVALID for row in rows]

    field_accuracy = {
        field: round(accuracy(column(gold_structured, field), column(pred_structured, field)), 4)
        for field in STRUCTURED_FIELDS
    }
    f1 = {
        field: round(
            macro_f1(
                [str(value) for value in column(gold_structured, field)],
                [str(value) for value in column(pred_structured, field)],
                labels=[member.value for member in enum],
            ),
            4,
        )
        for field, enum in ENUM_FIELDS.items()
    }
    rouge = [
        rouge_l(item.label.summary, answer.summary) if answer is not None else 0.0
        for item, answer in zip(gold, predicted, strict=True)
    ]
    confusions: Counter[tuple[str, str]] = Counter(
        (item.label.intent.value, answer.intent.value if answer else INVALID)
        for item, answer in zip(gold, predicted, strict=True)
        if answer is None or answer.intent != item.label.intent
    )
    return SystemScores(
        system=system,
        model=model,
        items=items,
        json_valid_rate=round(sum(json_valid) / items, 4) if items else 0.0,
        schema_valid_rate=round(sum(1 for answer in predicted if answer is not None) / items, 4) if items else 0.0,
        exact_match=round(
            sum(exact_match(answer, item.label) for answer, item in zip(predicted, gold, strict=True)) / items, 4
        )
        if items
        else 0.0,
        field_accuracy=field_accuracy,
        macro_f1=f1,
        summary_rouge_l=round(sum(rouge) / items, 4) if items else 0.0,
        latency_s_per_item={
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
        },
        latency_note=latency_note,
        intent_confusions=[
            {"gold": gold_intent, "predicted": predicted_intent, "count": count}
            for (gold_intent, predicted_intent), count in confusions.most_common(8)
        ],
    )


def load_gold(config: PipelineConfig) -> list[GoldItem]:
    gold = read_jsonl(config.eval.gold_path, GoldItem)
    return gold[: config.eval.limit] if config.eval.limit else gold


async def run_teacher_eval(
    config: PipelineConfig, gold: Sequence[GoldItem], factory: ClientFactory
) -> tuple[list[TicketTriage | None], str, list[CallStat], list[CallStat]]:
    labeler = config.teacher.labelers[config.eval.teacher_index]
    client = factory.client(labeler, stage="eval.teacher")
    items = [(item.id, item.text) for item in gold]
    run = await label_with_teacher(client, items, config.eval.teacher_batch_size, config.label.max_repair_attempts)
    answers = [run.labels[item.id].triage if item.id in run.labels else None for item in gold]
    served = Counter(label.model for label in run.labels.values() if label.triage is not None)
    model = served.most_common(1)[0][0] if served else client.label
    probe: list[CallStat] = []
    if config.eval.teacher_latency_probe:
        probe_items = items[: config.eval.teacher_latency_probe]
        probe_run = await label_with_teacher(client, probe_items, 1, 0)
        probe = probe_run.calls
    return answers, model, run.calls, probe


def _student_rows(gold: Sequence[GoldItem], predictions: Sequence[StudentPrediction]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "gold": item.label.model_dump(mode="json"),
            "predicted": prediction.triage.model_dump(mode="json") if prediction.triage else None,
            "raw": prediction.raw,
            "error": prediction.error,
            "confidence": prediction.confidence,
            "field_confidence": prediction.field_confidence,
            "latency_s": round(prediction.latency_s, 4),
            "exact_match": exact_match(prediction.triage, item.label),
        }
        for item, prediction in zip(gold, predictions, strict=True)
    ]


def _latency_probe(student: Student, texts: Sequence[str]) -> list[float]:
    return [student.predict_batch([text])[0].latency_s for text in texts]


async def run_eval(
    config: PipelineConfig, factory: ClientFactory, paths: RunPaths, *, systems: Sequence[str]
) -> dict[str, Any]:
    student_backend = config.student.backend
    gold = load_gold(config)
    texts = [item.text for item in gold]
    report: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config.name,
        "gold_items": len(gold),
        "gold_path": str(config.eval.gold_path),
        "systems": {},
    }
    teacher_answers: list[TicketTriage | None] | None = None
    teacher_tokens = (0.0, 0.0)

    if "teacher" in systems:
        answers, model, calls, probe = await run_teacher_eval(config, gold, factory)
        teacher_answers = answers
        batch_items = sum(call.items for call in calls) or 1
        per_item = [call.latency_s / call.items for call in calls for _ in range(call.items)]
        report["systems"]["teacher"] = score_system(
            "teacher",
            model,
            gold,
            answers,
            [answer is not None for answer in answers],
            per_item,
            latency_note=f"batched requests of {config.eval.teacher_batch_size}, latency divided by batch size",
        ).model_dump()
        source = probe or calls
        source_items = sum(call.items for call in source) or 1
        teacher_tokens = (
            sum(call.input_tokens for call in source) / source_items,
            sum(call.output_tokens for call in source) / source_items,
        )
        report["teacher_usage"] = {
            "batched": {
                "calls": len(calls),
                "input_tokens_per_item": round(sum(call.input_tokens for call in calls) / batch_items, 1),
                "output_tokens_per_item": round(sum(call.output_tokens for call in calls) / batch_items, 1),
            },
            "single_request_probe": {
                "calls": len(probe),
                "latency_s": {
                    "p50": round(percentile([call.latency_s for call in probe], 50), 2),
                    "p95": round(percentile([call.latency_s for call in probe], 95), 2),
                },
                "input_tokens_per_item": round(teacher_tokens[0], 1),
                "output_tokens_per_item": round(teacher_tokens[1], 1),
            }
            if probe
            else None,
        }
        write_jsonl(
            paths.predictions("teacher"),
            [
                {
                    "id": item.id,
                    "gold": item.label.model_dump(mode="json"),
                    "predicted": answer.model_dump(mode="json") if answer else None,
                    "exact_match": exact_match(answer, item.label),
                }
                for item, answer in zip(gold, answers, strict=True)
            ],
        )

    student_cfg = config.student
    finetuned: list[StudentPrediction] | None = None
    finetuned_student: Student | None = None
    student_seconds: float | None = None
    for system in systems:
        if system == "teacher":
            continue
        if system == "student_zero_shot":
            if student_backend == "fake":
                log.info("eval.skip", system=system, reason="the fake student has no zero-shot variant")
                continue
            student = load_student(
                "hf",
                base_model=student_cfg.base_model,
                adapter_path=None,
                max_new_tokens=student_cfg.max_new_tokens + 40,
                device=student_cfg.device,
                torch_threads=student_cfg.torch_threads,
                system_prompt=zero_shot_system_prompt(),
                name=f"{student_cfg.base_model} (zero-shot: labeling guide + format)",
            )
        elif system == "student_finetuned":
            adapter = paths.adapter_dir if student_backend == "hf" else None
            if student_backend == "hf" and not (paths.adapter_dir / "adapter_config.json").exists():
                log.warning(
                    "eval.skip", system=system, reason=f"no adapter at {paths.adapter_dir}; run `distillery train`"
                )
                continue
            student = load_student(
                student_backend,
                base_model=student_cfg.base_model,
                adapter_path=adapter,
                max_new_tokens=student_cfg.max_new_tokens,
                device=student_cfg.device,
                torch_threads=student_cfg.torch_threads,
            )
        else:
            raise ValueError(f"unknown system {system!r}")
        log.info("eval.student", system=system, model=student.label, items=len(texts))
        predictions = predict_all(student, texts, student_cfg.batch_size)
        probe_latency = _latency_probe(student, texts[: min(5, len(texts))])
        scores = score_system(
            system,
            student.label,
            gold,
            [prediction.triage for prediction in predictions],
            [prediction.json_valid for prediction in predictions],
            [prediction.latency_s for prediction in predictions],
            latency_note=f"batched generation ({student_cfg.batch_size} per batch); single-ticket p50 "
            f"{percentile(probe_latency, 50):.2f}s",
        ).model_dump()
        scores["single_request_latency_s_p50"] = round(percentile(probe_latency, 50), 3)
        report["systems"][system] = scores
        write_jsonl(paths.predictions(system), _student_rows(gold, predictions))
        if system == "student_finetuned":
            finetuned, finetuned_student = predictions, student
            student_seconds = scores["latency_s_per_item"]["mean"]

    if finetuned is not None and finetuned_student is not None:
        report["router"] = _router_section(config, paths, gold, finetuned, finetuned_student, teacher_answers)

    report["cost"] = _cost_section(config, report, teacher_tokens, student_seconds, student_backend)
    if paths.train_metrics.exists():
        report["training"] = {key: value for key, value in read_json(paths.train_metrics).items() if key != "history"}
    write_json(paths.eval_report, report)
    return report


def _router_section(
    config: PipelineConfig,
    paths: RunPaths,
    gold: Sequence[GoldItem],
    predictions: Sequence[StudentPrediction],
    student: Student,
    teacher_answers: Sequence[TicketTriage | None] | None,
) -> dict[str, Any]:
    section: dict[str, Any] = {"target_exact_match": config.router.target_exact_match}
    threshold_setting = config.router.min_confidence
    if threshold_setting == "auto":
        val_examples = [record_to_example(record) for record in iter_jsonl(paths.val)] if paths.val.exists() else []
        if val_examples:
            val_predictions = predict_all(
                student, [example.text for example in val_examples], config.student.batch_size
            )
            threshold = calibrate_threshold(
                [prediction.confidence for prediction in val_predictions],
                [
                    exact_match(prediction.triage, example.triage)
                    for prediction, example in zip(val_predictions, val_examples, strict=True)
                ],
                config.router.target_exact_match,
            )
            section["calibration"] = {
                "split": "val (teacher labels)",
                "items": len(val_examples),
                "student_exact_match": round(
                    sum(exact_match(p.triage, e.triage) for p, e in zip(val_predictions, val_examples, strict=True))
                    / len(val_examples),
                    4,
                ),
            }
        else:
            threshold = 0.5
            section["calibration"] = {"split": "none (no val split), default 0.5"}
    else:
        threshold = threshold_setting
        section["calibration"] = {"split": "fixed in config"}
    section["threshold"] = threshold
    gold_labels = [item.label for item in gold]
    if teacher_answers is not None:
        section["gold"] = simulate(predictions, teacher_answers, gold_labels, threshold).model_dump()
        section["curve"] = tradeoff_curve(predictions, teacher_answers, gold_labels)
    else:
        # No teacher run: report what the student would handle alone at this threshold.
        section["gold"] = simulate(predictions, [None] * len(gold), gold_labels, threshold).model_dump()
    return section


def _cost_section(
    config: PipelineConfig,
    report: dict[str, Any],
    teacher_tokens: tuple[float, float],
    student_seconds: float | None,
    student_backend: str,
) -> dict[str, Any]:
    measured_seconds = student_seconds if student_backend == "hf" else None
    device = resolve_device(config.student.device) if measured_seconds is not None else "cpu"
    lines = cost_table(
        config.cost,
        teacher_input_tokens=teacher_tokens[0],
        teacher_output_tokens=teacher_tokens[1],
        student_seconds_per_item=measured_seconds,
        student_device=device,
    )
    section: dict[str, Any] = {
        "per_1k_tickets": [line.model_dump() for line in lines],
        "assumptions": {
            "cpu_instance": config.cost.cpu_instance,
            "cpu_hourly_usd": config.cost.cpu_hourly_usd,
            "gpu_instance": config.cost.gpu_instance,
            "gpu_hourly_usd": config.cost.gpu_hourly_usd,
            "gpu_assumed_items_per_second": config.cost.gpu_assumed_items_per_second,
        },
    }
    router = report.get("router", {}).get("gold")
    student_line = next((line for line in lines if line.name.startswith("Student") and line.measured), None)
    if router and student_line is not None:
        section["routed"] = [
            {
                "teacher": line.name.removeprefix("Teacher via API: "),
                "student": "GPU" if device == "cuda" else "CPU",
                "usd_per_1k": round(
                    routed_cost_per_1k(student_line.usd_per_1k, line.usd_per_1k, router["offload_rate"]), 4
                ),
                "offload_rate": router["offload_rate"],
            }
            for line in lines
            if line.name.startswith("Teacher")
        ]
    return section
