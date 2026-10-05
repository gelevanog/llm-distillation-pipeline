from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from distillery.schema import (
    ALL_FIELDS,
    TicketTriage,
    batch_json_schema,
    extract_json,
    normalize_order_id,
    parse_triage,
    triage_json_schema,
)
from tests.conftest import make_triage


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BL-482913", "BL-482913"),
        ("#BL-482913", "BL-482913"),
        ("bl482913", "BL-482913"),
        ("bl 482913", "BL-482913"),
        ("482913", "BL-482913"),
        ("  #482913 ", "BL-482913"),
        ("null", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_order_id(raw: str | None, expected: str | None) -> None:
    assert normalize_order_id(raw) == expected


def test_order_id_is_normalized_on_validation() -> None:
    assert make_triage(order_id="#bl 204815").order_id == "BL-204815"


@pytest.mark.parametrize("bad", ["BL-12345", "ORDER-123456", "12345678"])
def test_invalid_order_id_rejected(bad: str) -> None:
    with pytest.raises(ValidationError, match="order_id"):
        make_triage(order_id=bad)


def test_unknown_enum_and_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        make_triage(intent="refund")
    with pytest.raises(ValidationError):
        TicketTriage.model_validate({**make_triage().model_dump(mode="json"), "language": "en"})


def test_summary_word_limit_and_whitespace() -> None:
    assert make_triage(summary="  Customer   asks\nabout bulbs. ").summary == "Customer asks about bulbs."
    with pytest.raises(ValidationError, match="30 words"):
        make_triage(summary=" ".join(["word"] * 31))


def test_to_json_is_compact_and_in_schema_order() -> None:
    text = make_triage().to_json()
    assert list(json.loads(text)) == list(ALL_FIELDS)
    assert text.startswith('{"intent": "order_status", "urgency": "medium"')


def test_json_schema_is_strict_mode_compatible() -> None:
    schema = triage_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(ALL_FIELDS)
    assert "refund_request" in schema["properties"]["intent"]["enum"]
    batch = batch_json_schema(schema)
    item = batch["properties"]["items"]["items"]
    assert item["required"][0] == "id" and item["additionalProperties"] is False


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here is the JSON:\n{"a": 1}\nHope this helps.',
        '  {"a": 1}  trailing',
    ],
)
def test_extract_json_tolerates_fences_and_prose(text: str) -> None:
    assert extract_json(text) == {"a": 1}


def test_extract_json_raises_without_json() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        extract_json("I cannot help with that.")


def test_parse_triage_outcomes() -> None:
    good = parse_triage(make_triage().to_json())
    assert good.ok and good.json_valid

    broken = parse_triage('{"intent": "order_status", "urgency":')
    assert not broken.json_valid and not broken.ok

    not_object = parse_triage("[1, 2]")
    assert not_object.json_valid and not not_object.ok and "object" in (not_object.error or "")

    schema_error = parse_triage(json.dumps({**make_triage().model_dump(mode="json"), "urgency": "urgent"}))
    assert schema_error.json_valid and not schema_error.ok and "urgency" in (schema_error.error or "")


def test_salvage_truncated_batch_answer() -> None:
    from distillery.schema import batch_items

    truncated = '{"items": [{"id": "a", "text": "first"}, {"id": "b", "text": "sec\\"ond"}, {"id": "c", "text": "cut of'
    assert batch_items(truncated) == [{"id": "a", "text": "first"}, {"id": "b", "text": 'sec"ond'}]
    assert batch_items('{"items": []}') == []
    assert batch_items("nothing here") is None


def test_zero_shot_prompt_contains_guide_and_exact_format() -> None:
    from distillery.prompts import LABELING_GUIDE, zero_shot_system_prompt

    prompt = zero_shot_system_prompt()
    assert prompt.startswith(LABELING_GUIDE)
    assert '{"intent": one of [' in prompt and '"summary": one sentence' in prompt
