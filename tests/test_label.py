from __future__ import annotations

import asyncio
import json
from typing import Any

from distillery.label import agreement_stats, label_batch, labels_agree, parse_batch
from distillery.providers.client import CallLedger, LLMClient
from distillery.records import TeacherLabel
from tests.conftest import make_labeled, make_triage
from tests.scripted import ScriptedProvider


def _item(item_id: str, **overrides: Any) -> dict[str, Any]:
    return {"id": item_id, **make_triage(**overrides).model_dump(mode="json")}


def _client(provider: ScriptedProvider) -> LLMClient:
    return LLMClient(provider, ledger=CallLedger(None, 100), max_retries=0)


def test_parse_batch_splits_valid_invalid_and_missing() -> None:
    bad = _item("b")
    bad["intent"] = "refund"
    text = json.dumps({"items": [_item("a"), bad, _item("zzz"), _item("a", intent="other")]})
    valid, errors = parse_batch(text, ["a", "b", "c"])
    assert set(valid) == {"a"}
    assert valid["a"].intent.value == "order_status"  # the first answer for an id wins
    assert "intent" in errors["b"]
    assert errors["c"] == "missing from the answer"


def test_parse_batch_invalid_json_fails_every_item() -> None:
    valid, errors = parse_batch("not json at all", ["a", "b"])
    assert valid == {}
    assert set(errors) == {"a", "b"}


def test_parse_batch_accepts_bare_list() -> None:
    valid, _ = parse_batch(json.dumps([_item("a")]), ["a"])
    assert "a" in valid


def test_label_batch_repairs_invalid_items_once() -> None:
    bad = _item("t2")
    bad["urgency"] = "urgent"
    provider = ScriptedProvider(
        [
            json.dumps({"items": [_item("t1"), bad]}),
            json.dumps({"items": [_item("t2", urgency="high")]}),
        ]
    )
    labels, stats = asyncio.run(label_batch(_client(provider), [("t1", "first"), ("t2", "second")], 1))
    assert labels["t1"].triage is not None and not labels["t1"].repaired
    assert labels["t2"].triage is not None and labels["t2"].repaired
    assert labels["t2"].triage.urgency.value == "high"
    assert len(stats) == 2
    repair_request = provider.requests[1]
    assert repair_request.task == "repair"
    assert "rejected by the validator" in repair_request.user
    assert "t2" in repair_request.user and "t1" not in repair_request.user.split("Previous answer")[0]


def test_label_batch_gives_up_after_repair_budget() -> None:
    provider = ScriptedProvider(["garbage", "still garbage"])
    labels, _ = asyncio.run(label_batch(_client(provider), [("t1", "text")], 1))
    assert labels["t1"].triage is None
    assert "invalid JSON" in (labels["t1"].error or "")


def test_label_batch_request_failure_marks_items_failed() -> None:
    from distillery.providers.base import ProviderError

    provider = ScriptedProvider([ProviderError("401 unauthorized")])
    labels, stats = asyncio.run(label_batch(_client(provider), [("t1", "text")], 1))
    assert labels["t1"].triage is None and "request failed" in (labels["t1"].error or "")
    assert stats == []


def test_labels_agree() -> None:
    a = TeacherLabel(model="a", triage=make_triage())
    b = TeacherLabel(model="b", triage=make_triage(summary="A different but valid summary text."))
    c = TeacherLabel(model="c", triage=make_triage(urgency="high"))
    fields = ["intent", "urgency", "sentiment", "product_area", "order_id"]
    assert labels_agree([a, b], fields) == (True, [])
    assert labels_agree([a, c], fields) == (False, ["urgency"])
    assert labels_agree([a, c], ["intent"]) == (True, [])
    assert labels_agree([a, TeacherLabel(model="x", error="bad")], fields) == (False, ["invalid"])


def test_agreement_stats() -> None:
    tickets = [
        make_labeled("1", "x", make_triage(), make_triage()),
        make_labeled("2", "y", make_triage(), make_triage(sentiment="negative")),
        make_labeled("3", "z", make_triage(), None),
    ]
    stats = agreement_stats(tickets)
    assert stats["both_valid"] == 2
    assert stats["all_fields_agreement"] == 0.5
    assert stats["fields"]["intent"]["agreement"] == 1.0
    assert stats["fields"]["sentiment"]["agreement"] == 0.5
    assert stats["teachers"][1]["invalid"] == 1
