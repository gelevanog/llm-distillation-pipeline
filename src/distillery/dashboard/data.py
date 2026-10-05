"""Turn a run directory into the view model the dashboard and the static report render."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from distillery.io import iter_jsonl, read_json
from distillery.records import RunPaths

SYSTEM_LABELS = {
    "teacher": "Teacher",
    "student_zero_shot": "Student, zero-shot",
    "student_finetuned": "Student, fine-tuned",
}
SYSTEM_SLOTS = {"teacher": 1, "student_zero_shot": 2, "student_finetuned": 3}
FIELD_LABELS = {
    "intent": "Intent",
    "urgency": "Urgency",
    "sentiment": "Sentiment",
    "product_area": "Product area",
    "order_id": "Order id",
}


def _load(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def _funnel_view(funnel: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not funnel:
        return []
    top = max(step["kept"] + step.get("dropped", 0) for step in funnel) or 1
    return [
        {
            **step,
            "width": round(100 * step["kept"] / top, 2),
            "dropped_width": round(100 * step.get("dropped", 0) / top, 2),
            "details": ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in step["reasons"].items()),
        }
        for step in funnel
    ]


def _distribution_view(stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not stats:
        return []
    fields = []
    for field, counts in stats["distribution"]["all"].items():
        total = sum(counts.values()) or 1
        top = max(counts.values(), default=1) or 1
        fields.append(
            {
                "field": FIELD_LABELS.get(field, field),
                "rows": [
                    {"label": label, "count": count, "share": count / total, "width": round(100 * count / top, 2)}
                    for label, count in counts.items()
                ],
            }
        )
    return fields


def _metric_rows(systems: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per metric, one cell per system (value 0..1 plus bar width)."""
    metrics: list[tuple[str, str, tuple[str, ...]]] = [
        ("JSON valid", "Output parses as JSON", ("json_valid_rate",)),
        ("Schema valid", "Passes the Pydantic schema", ("schema_valid_rate",)),
        ("Exact match", "All 5 structured fields correct", ("exact_match",)),
        *[(f"{label} accuracy", "", ("field_accuracy", field)) for field, label in FIELD_LABELS.items()],
        ("Intent macro-F1", "Unweighted mean over 11 intents", ("macro_f1", "intent")),
        ("Urgency macro-F1", "", ("macro_f1", "urgency")),
        ("Summary ROUGE-L", "Word overlap with the hand-written summary", ("summary_rouge_l",)),
    ]
    rows = []
    for name, hint, keys in metrics:
        cells = []
        for system, scores in systems.items():
            value: Any = scores
            for key in keys:
                value = value[key]
            cells.append(
                {
                    "system": system,
                    "slot": SYSTEM_SLOTS.get(system, 1),
                    "value": float(value),
                    "width": round(100 * value, 2),
                }
            )
        rows.append({"name": name, "hint": hint, "cells": cells})
    return rows


def _curve_view(
    router: dict[str, Any] | None, width: int = 560, height: int = 240, pad: int = 36
) -> dict[str, Any] | None:
    if not router or not router.get("curve"):
        return None
    points = router["curve"]
    inner_w, inner_h = width - 2 * pad, height - 2 * pad

    def xy(threshold: float, value: float) -> tuple[float, float]:
        return round(pad + threshold * inner_w, 1), round(pad + (1 - value) * inner_h, 1)

    def path(key: str) -> str:
        segments = []
        for index, point in enumerate(points):
            x, y = xy(point["threshold"], point[key])
            segments.append(f"{'M' if index == 0 else 'L'}{x},{y}")
        return " ".join(segments)

    chosen = min(router["threshold"], 1.0)
    gold = router["gold"]
    marker_x, offload_y = xy(chosen, gold["offload_rate"])
    _, exact_y = xy(chosen, gold["exact_match"])
    return {
        "width": width,
        "height": height,
        "pad": pad,
        "offload_path": path("offload_rate"),
        "exact_path": path("exact_match"),
        "points": [
            {
                "x": xy(point["threshold"], 0)[0],
                "offload_y": xy(point["threshold"], point["offload_rate"])[1],
                "exact_y": xy(point["threshold"], point["exact_match"])[1],
                **point,
            }
            for point in points
        ],
        "grid": [{"y": xy(0, value)[1], "label": f"{value:.0%}"} for value in (0, 0.25, 0.5, 0.75, 1.0)],
        "xticks": [{"x": xy(value, 0)[0], "label": f"{value:.1f}"} for value in (0, 0.2, 0.4, 0.6, 0.8, 1.0)],
        "marker": {"x": marker_x, "offload_y": offload_y, "exact_y": exact_y, "threshold": chosen},
        "baseline_y": xy(0, 0)[1],
    }


def _calls_view(paths: RunPaths) -> dict[str, Any] | None:
    if not paths.calls.exists():
        return None
    rows = list(iter_jsonl(paths.calls))
    by_status = Counter(row["status"] for row in rows)
    by_model = Counter(row.get("served_model") or "-" for row in rows if row["status"] == "ok")
    by_stage = Counter(row["stage"] for row in rows)
    return {
        "total": len(rows),
        "ok": by_status.get("ok", 0),
        "retried": by_status.get("retryable_error", 0),
        "failed": by_status.get("error", 0),
        "models": by_model.most_common(),
        "stages": by_stage.most_common(),
        "all_free": all((row.get("provider") or "").endswith(":free") for row in rows)
        and all((row.get("served_model") or ":free").endswith(":free") for row in rows),
    }


def load_view(run_dir: Path) -> dict[str, Any]:
    paths = RunPaths(root=run_dir)
    stats = _load(paths.dataset_stats)
    report = _load(paths.eval_report) or {}
    systems = report.get("systems", {})
    router = report.get("router")
    teacher_usage = report.get("teacher_usage")
    return {
        "run_dir": str(run_dir),
        "has_data": bool(stats or report),
        "stats": stats,
        "funnel": _funnel_view(_load(paths.funnel)),
        "label_stats": _load(paths.label_stats),
        "distribution": _distribution_view(stats),
        "report": report,
        "systems": [
            {"key": key, "label": SYSTEM_LABELS.get(key, key), "slot": SYSTEM_SLOTS.get(key, 1), **scores}
            for key, scores in systems.items()
        ],
        "metric_rows": _metric_rows(systems),
        "router": router,
        "curve": _curve_view(router),
        "cost": report.get("cost"),
        "teacher_usage": teacher_usage,
        "training": report.get("training") or _load(paths.train_metrics),
        "calls": _calls_view(paths),
    }
