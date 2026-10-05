"""Cost per 1,000 tickets: teacher via API list prices vs a self-hosted student.

Inputs are measured where possible (teacher tokens per ticket, student CPU latency per ticket) and
explicit assumptions where not (instance prices, GPU throughput). Every number in the output says
which one it is.
"""

from __future__ import annotations

from pydantic import BaseModel

from distillery.config import CostConfig, PriceConfig


class CostLine(BaseModel):
    name: str
    usd_per_1k: float
    basis: str
    measured: bool


def api_cost_per_1k(price: PriceConfig, input_tokens_per_item: float, output_tokens_per_item: float) -> float:
    per_item = (
        input_tokens_per_item * price.input_per_million + output_tokens_per_item * price.output_per_million
    ) / 1_000_000
    return per_item * 1000


def self_hosted_cost_per_1k(seconds_per_item: float, hourly_usd: float, parallel_workers: int = 1) -> float:
    hours = seconds_per_item * 1000 / 3600 / max(1, parallel_workers)
    return hours * hourly_usd


def routed_cost_per_1k(student_per_1k: float, teacher_per_1k: float, offload_rate: float) -> float:
    """The student always runs first; the teacher is paid only for the share it handles."""
    return student_per_1k + (1 - offload_rate) * teacher_per_1k


def cost_table(
    config: CostConfig,
    *,
    teacher_input_tokens: float,
    teacher_output_tokens: float,
    student_seconds_per_item: float | None,
    student_device: str = "cpu",
) -> list[CostLine]:
    """API rows need measured teacher tokens; the student row needs a measured latency on `student_device`."""
    lines: list[CostLine] = []
    if teacher_input_tokens or teacher_output_tokens:
        lines += [
            CostLine(
                name=f"Teacher via API: {price.name}",
                usd_per_1k=round(api_cost_per_1k(price, teacher_input_tokens, teacher_output_tokens), 4),
                basis=(
                    f"measured {teacher_input_tokens:.0f} input + {teacher_output_tokens:.0f} output tokens per ticket "
                    f"(single-ticket requests) x list price ${price.input_per_million}/${price.output_per_million} "
                    "per 1M" + (f" ({price.source})" if price.source else "")
                ),
                measured=True,
            )
            for price in config.api_prices
        ]
    on_gpu = student_device == "cuda"
    if student_seconds_per_item is not None:
        instance, hourly = (
            (config.gpu_instance, config.gpu_hourly_usd) if on_gpu else (config.cpu_instance, config.cpu_hourly_usd)
        )
        workers = 1 if on_gpu else config.cpu_parallel_workers
        lines.append(
            CostLine(
                name=f"Student self-hosted, {'GPU' if on_gpu else 'CPU'} ({instance})",
                usd_per_1k=round(self_hosted_cost_per_1k(student_seconds_per_item, hourly, workers), 4),
                basis=(
                    f"measured {student_seconds_per_item:.2f} s/ticket (batched generation); "
                    f"assumed ${hourly}/h instance price"
                ),
                measured=True,
            )
        )
    if not (on_gpu and student_seconds_per_item is not None):
        lines.append(
            CostLine(
                name=f"Student self-hosted, GPU ({config.gpu_instance})",
                usd_per_1k=round(
                    self_hosted_cost_per_1k(1 / config.gpu_assumed_items_per_second, config.gpu_hourly_usd), 4
                ),
                basis=(
                    f"assumption, not measured: {config.gpu_assumed_items_per_second:g} tickets/s batched on a GPU "
                    f"server at ${config.gpu_hourly_usd}/h"
                ),
                measured=False,
            )
        )
    return lines
