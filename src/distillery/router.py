"""Stage 7, route: answer with the student when it is valid and confident, else fall back to the teacher.

`decide()` is the whole policy (pure function, unit-tested). `calibrate_threshold()` picks the
confidence threshold on the validation split, and `simulate()` replays the policy offline on the
gold set using already-computed student and teacher predictions, so the offload numbers in the
eval cost no extra API calls.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from distillery.label import label_batch
from distillery.providers.client import LLMClient
from distillery.schema import STRUCTURED_FIELDS, TicketTriage
from distillery.student import Student, StudentPrediction

NEVER = 1.01  # a threshold no confidence reaches: everything goes to the teacher


class RouteDecision(BaseModel):
    source: Literal["student", "teacher"]
    reason: str


def decide(prediction: StudentPrediction, threshold: float) -> RouteDecision:
    if not prediction.json_valid:
        return RouteDecision(source="teacher", reason="student output is not valid JSON")
    if prediction.triage is None:
        return RouteDecision(source="teacher", reason=f"student output failed validation: {prediction.error}")
    if prediction.confidence < threshold:
        return RouteDecision(
            source="teacher", reason=f"low confidence {prediction.confidence:.2f} < threshold {threshold:.2f}"
        )
    return RouteDecision(source="student", reason=f"confidence {prediction.confidence:.2f} >= {threshold:.2f}")


def exact_match(predicted: TicketTriage | None, gold: TicketTriage) -> bool:
    return predicted is not None and predicted.structured() == gold.structured()


def calibrate_threshold(
    confidences: Sequence[float], correct: Sequence[bool], target: float, min_accepted: int = 5
) -> float:
    """Lowest threshold whose accepted answers reach `target` exact-match (most offload at that quality).

    `confidences` must be 0 for invalid answers. Returns `NEVER` when no threshold reaches the target.
    """
    pairs = sorted(zip(confidences, correct, strict=True), key=lambda pair: pair[0])
    for index, (threshold, _) in enumerate(pairs):
        if threshold <= 0:
            continue
        accepted = [ok for _, ok in pairs[index:]]
        if len(accepted) >= min_accepted and sum(accepted) / len(accepted) >= target:
            return threshold
    return NEVER


class RouterSimulation(BaseModel):
    threshold: float
    items: int
    offload_rate: float
    exact_match: float
    field_accuracy: dict[str, float]
    student_exact_match_on_offloaded: float | None


def simulate(
    student: Sequence[StudentPrediction],
    teacher: Sequence[TicketTriage | None],
    gold: Sequence[TicketTriage],
    threshold: float,
) -> RouterSimulation:
    routed: list[TicketTriage | None] = []
    offloaded_correct: list[bool] = []
    for prediction, teacher_answer, gold_answer in zip(student, teacher, gold, strict=True):
        if decide(prediction, threshold).source == "student":
            routed.append(prediction.triage)
            offloaded_correct.append(exact_match(prediction.triage, gold_answer))
        else:
            routed.append(teacher_answer)
    items = len(gold)
    field_accuracy = {
        field: round(
            sum(
                1
                for answer, gold_answer in zip(routed, gold, strict=True)
                if answer is not None and answer.structured()[field] == gold_answer.structured()[field]
            )
            / items,
            4,
        )
        for field in STRUCTURED_FIELDS
    }
    return RouterSimulation(
        threshold=threshold,
        items=items,
        offload_rate=round(len(offloaded_correct) / items, 4) if items else 0.0,
        exact_match=round(sum(exact_match(answer, g) for answer, g in zip(routed, gold, strict=True)) / items, 4)
        if items
        else 0.0,
        field_accuracy=field_accuracy,
        student_exact_match_on_offloaded=round(sum(offloaded_correct) / len(offloaded_correct), 4)
        if offloaded_correct
        else None,
    )


def tradeoff_curve(
    student: Sequence[StudentPrediction],
    teacher: Sequence[TicketTriage | None],
    gold: Sequence[TicketTriage],
    steps: int = 20,
) -> list[dict[str, float]]:
    """Offload rate vs routed exact-match for thresholds 0..1 (for the dashboard chart)."""
    points = []
    for step in range(steps + 1):
        threshold = round(step / steps, 3)
        result = simulate(student, teacher, gold, threshold)
        points.append({"threshold": threshold, "offload_rate": result.offload_rate, "exact_match": result.exact_match})
    return points


class RoutedAnswer(BaseModel):
    triage: TicketTriage | None
    source: Literal["student", "teacher", "none"]
    reason: str
    student: StudentPrediction
    teacher_model: str | None = None
    teacher_error: str | None = None
    latency_s: float


class Router:
    """Online router used by the API: student first, teacher on fallback."""

    def __init__(self, student: Student, teacher: LLMClient | None, threshold: float) -> None:
        self.student = student
        self.teacher = teacher
        self.threshold = threshold

    async def triage(self, text: str) -> RoutedAnswer:
        started = time.perf_counter()
        prediction = self.student.predict_batch([text])[0]
        decision = decide(prediction, self.threshold)
        if decision.source == "student" or self.teacher is None:
            source: Literal["student", "teacher", "none"] = "student" if decision.source == "student" else "none"
            reason = decision.reason if self.teacher is not None else decision.reason + "; no teacher configured"
            return RoutedAnswer(
                triage=prediction.triage if source == "student" else None,
                source=source,
                reason=reason,
                student=prediction,
                latency_s=time.perf_counter() - started,
            )
        labels, _ = await label_batch(self.teacher, [("ticket", text)], max_repair_attempts=1)
        label = labels["ticket"]
        return RoutedAnswer(
            triage=label.triage,
            source="teacher",
            reason=decision.reason,
            student=prediction,
            teacher_model=label.model,
            teacher_error=label.error,
            latency_s=time.perf_counter() - started,
        )
