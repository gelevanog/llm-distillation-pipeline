from __future__ import annotations

from collections import Counter

from distillery.build import (
    dataset_card,
    dataset_stats,
    record_to_example,
    stratified_split,
    to_chat_record,
)
from distillery.prompts import STUDENT_SYSTEM_PROMPT
from distillery.records import FunnelStep
from tests.conftest import make_example

INTENTS = ["order_status"] * 20 + ["refund_request"] * 10 + ["other"] * 2 + ["feedback"]


def _examples() -> list:
    return [make_example(f"e{index:03d}", intent) for index, intent in enumerate(INTENTS)]


def test_stratified_split_keeps_label_mix() -> None:
    train, val = stratified_split(_examples(), 0.2, seed=3)
    assert len(train) + len(val) == len(INTENTS)
    val_counts = Counter(example.triage.intent.value for example in val)
    assert val_counts == {"order_status": 4, "refund_request": 2, "other": 1}
    assert not {example.id for example in train} & {example.id for example in val}


def test_stratified_split_is_deterministic() -> None:
    first = stratified_split(_examples(), 0.2, seed=3)
    second = stratified_split(list(reversed(_examples())), 0.2, seed=3)
    assert [example.id for example in first[1]] == [example.id for example in second[1]]


def test_zero_val_fraction() -> None:
    train, val = stratified_split(_examples(), 0.0, seed=1)
    assert val == [] and len(train) == len(INTENTS)


def test_chat_record_roundtrip() -> None:
    example = make_example("x1", "refund_request", text="I want my money back for order BL-123456")
    record = to_chat_record(example)
    roles = [message["role"] for message in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == STUDENT_SYSTEM_PROMPT
    assert record["messages"][2]["content"] == example.triage.to_json()
    assert record_to_example(record) == example


def test_dataset_stats_and_card() -> None:
    train, val = stratified_split(_examples(), 0.2, seed=3)
    funnel = [
        FunnelStep(name="input", description="in", kept=40),
        FunnelStep(name="dedup", description="d", kept=33, dropped=7, reasons={"near_duplicate": 7}),
    ]
    stats = dataset_stats(train, val, funnel, {}, ["fake/a"])
    assert stats["sizes"] == {"train": len(train), "val": len(val), "total": 33}
    assert stats["distribution"]["all"]["intent"]["order_status"] == 20
    card = dataset_card(stats, "unit")
    assert "# Dataset card: unit" in card
    assert "| dedup | d | 33 | 7 | near_duplicate: 7 |" in card
