from __future__ import annotations

import pytest

from distillery.config import CostConfig, PriceConfig
from distillery.cost import api_cost_per_1k, cost_table, routed_cost_per_1k, self_hosted_cost_per_1k


def test_api_cost_per_1k() -> None:
    price = PriceConfig(name="m", input_per_million=2.0, output_per_million=10.0)
    # 1000 tickets x (500 in + 200 out) tokens = 0.5M in ($1) + 0.2M out ($2)
    assert api_cost_per_1k(price, 500, 200) == pytest.approx(3.0)


def test_self_hosted_cost() -> None:
    # 1.8 s/ticket -> 0.5 h per 1k tickets at $1/h
    assert self_hosted_cost_per_1k(1.8, 1.0) == pytest.approx(0.5)
    assert self_hosted_cost_per_1k(1.8, 1.0, parallel_workers=2) == pytest.approx(0.25)


def test_routed_cost() -> None:
    assert routed_cost_per_1k(0.2, 3.0, offload_rate=0.75) == pytest.approx(0.95)
    assert routed_cost_per_1k(0.2, 3.0, offload_rate=0.0) == pytest.approx(3.2)


def test_cost_table_marks_measured_and_assumed() -> None:
    config = CostConfig(api_prices=[PriceConfig(name="paid", input_per_million=1, output_per_million=1)])
    lines = cost_table(config, teacher_input_tokens=1000, teacher_output_tokens=1000, student_seconds_per_item=0.5)
    assert [line.measured for line in lines] == [True, True, False]
    assert lines[0].usd_per_1k == pytest.approx(2.0)
    assert "CPU" in lines[1].name
    assert "assumption" in lines[-1].basis
    without_student = cost_table(config, teacher_input_tokens=1, teacher_output_tokens=1, student_seconds_per_item=None)
    assert len(without_student) == 2
    no_teacher = cost_table(config, teacher_input_tokens=0, teacher_output_tokens=0, student_seconds_per_item=None)
    assert [line.measured for line in no_teacher] == [False]


def test_cost_table_measured_gpu_replaces_the_assumption() -> None:
    lines = cost_table(
        CostConfig(),
        teacher_input_tokens=0,
        teacher_output_tokens=0,
        student_seconds_per_item=0.05,
        student_device="cuda",
    )
    assert len(lines) == 1 and lines[0].measured and "GPU" in lines[0].name
