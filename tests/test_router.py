from __future__ import annotations

import asyncio
import json

import pytest

from distillery.providers.client import CallLedger, LLMClient
from distillery.router import NEVER, Router, calibrate_threshold, decide, simulate, tradeoff_curve
from distillery.student import StudentPrediction
from tests.conftest import make_triage
from tests.scripted import ScriptedProvider


def _prediction(
    confidence: float = 0.9, *, valid: bool = True, json_valid: bool = True, **overrides: str
) -> StudentPrediction:
    triage = make_triage(**overrides) if valid else None
    return StudentPrediction(
        raw=triage.to_json() if triage else "{oops",
        triage=triage,
        json_valid=json_valid,
        error=None if valid else "intent: invalid",
        confidence=confidence if valid else 0.0,
    )


def test_decide() -> None:
    assert decide(_prediction(0.9), 0.5).source == "student"
    assert decide(_prediction(0.5), 0.5).source == "student"
    low = decide(_prediction(0.3), 0.5)
    assert low.source == "teacher" and "low confidence" in low.reason
    assert "not valid JSON" in decide(_prediction(valid=False, json_valid=False), 0.0).reason
    assert "failed validation" in decide(_prediction(valid=False), 0.0).reason


def test_calibrate_threshold_picks_lowest_threshold_meeting_target() -> None:
    confidences = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.0]
    correct = [True, True, True, True, True, False, False, False]
    # Accepting >= 0.7 keeps five answers, all correct; >= 0.6 drops to 5/6.
    assert calibrate_threshold(confidences, correct, target=1.0, min_accepted=3) == 0.7
    assert calibrate_threshold(confidences, correct, target=0.8, min_accepted=3) == 0.6
    assert calibrate_threshold(confidences, [False] * 8, target=0.5, min_accepted=1) == NEVER


def test_simulate_routes_and_scores() -> None:
    gold = [make_triage(), make_triage(intent="refund_request"), make_triage(intent="other")]
    student = [_prediction(0.9), _prediction(0.2, intent="other"), _prediction(valid=False)]
    teacher = [make_triage(), make_triage(intent="refund_request"), None]
    result = simulate(student, teacher, gold, threshold=0.5)
    assert result.offload_rate == pytest.approx(1 / 3, abs=1e-3)
    assert result.exact_match == pytest.approx(2 / 3, abs=1e-3)
    assert result.student_exact_match_on_offloaded == 1.0
    assert result.field_accuracy["intent"] == pytest.approx(2 / 3, abs=1e-3)

    curve = tradeoff_curve(student, teacher, gold, steps=4)
    assert curve[0]["offload_rate"] == pytest.approx(2 / 3, abs=1e-3)  # threshold 0: every valid answer
    assert curve[-1]["offload_rate"] == 0.0


class _StubStudent:
    def __init__(self, prediction: StudentPrediction) -> None:
        self.prediction = prediction

    @property
    def label(self) -> str:
        return "stub"

    def predict_batch(self, texts: list[str]) -> list[StudentPrediction]:
        return [self.prediction for _ in texts]


def test_router_uses_student_when_confident() -> None:
    teacher = ScriptedProvider([])
    router = Router(_StubStudent(_prediction(0.95)), LLMClient(teacher, ledger=CallLedger(None, 10)), 0.5)
    answer = asyncio.run(router.triage("where is my order"))
    assert answer.source == "student" and answer.triage is not None
    assert teacher.requests == []


def test_router_falls_back_to_teacher() -> None:
    reply = json.dumps({"items": [{"id": "ticket", **make_triage(intent="billing_issue").model_dump(mode="json")}]})
    teacher = ScriptedProvider([reply])
    router = Router(_StubStudent(_prediction(0.1)), LLMClient(teacher, ledger=CallLedger(None, 10)), 0.5)
    answer = asyncio.run(router.triage("I was charged twice"))
    assert answer.source == "teacher"
    assert answer.triage is not None and answer.triage.intent.value == "billing_issue"
    assert answer.teacher_model == "scripted:free"


def test_router_without_teacher_reports_no_answer() -> None:
    router = Router(_StubStudent(_prediction(valid=False)), None, 0.5)
    answer = asyncio.run(router.triage("???"))
    assert answer.source == "none" and answer.triage is None
    assert "no teacher configured" in answer.reason
