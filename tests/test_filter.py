from __future__ import annotations

import pytest

from distillery.config import FilterConfig
from distillery.dedup import find_near_duplicates, normalize
from distillery.filter import cap_per_class, run_filter
from distillery.pii import scrub_pii
from tests.conftest import make_example, make_labeled, make_triage

# ---------- PII ----------


@pytest.mark.parametrize(
    ("text", "expected", "kind"),
    [
        ("Write to jane.doe+shop@example.co.uk please", "Write to [EMAIL] please", "email"),
        ("Call me at +1 415 555 0134 tonight", "Call me at [PHONE] tonight", "phone"),
        ("my number is (030) 1234-5678", "my number is [PHONE]", "phone"),
        ("card 4111 1111 1111 1111 was charged", "card [CARD] was charged", "card"),
    ],
)
def test_scrub_pii(text: str, expected: str, kind: str) -> None:
    result = scrub_pii(text)
    assert result.text == expected
    assert result.counts == {kind: 1}


@pytest.mark.parametrize(
    "text",
    [
        "Order #BL-482913 has not arrived",
        "order 482913 is late",
        "It is 14°C inside, firmware 3.2.1, app v5.4.1",
        "I paid $249.00 on 2026-09-28",
    ],
)
def test_scrub_pii_leaves_task_data_alone(text: str) -> None:
    result = scrub_pii(text)
    assert result.text == text
    assert not result.changed


# ---------- Dedup ----------


def test_normalize_masks_digits_and_punctuation() -> None:
    assert normalize("Order #BL-482913 LATE!!") == normalize("order bl 000000 late")


def test_near_duplicates_found_and_first_kept() -> None:
    texts = [
        "My Aura thermostat stopped heating after the firmware update, please help.",
        "Do the Glow bulbs work with Google Home?",
        "My Aura thermostat stopped heating after the firmware update. Please help!!",
        "my aura thermostat stopped heating after the firmware update please help",
    ]
    duplicates = find_near_duplicates(texts, threshold=0.7)
    assert set(duplicates) == {2, 3}
    assert duplicates[2][0] == 0 and duplicates[3][0] == 0
    assert duplicates[2][1] >= 0.7


def test_order_numbers_do_not_make_duplicates_unique() -> None:
    texts = ["Where is order BL-111111? It is late.", "Where is order BL-999999? It is late."]
    assert 1 in find_near_duplicates(texts, threshold=0.9)


def test_distinct_texts_are_kept() -> None:
    texts = [
        "I was charged twice for my Protect plan this month.",
        "The doorbell battery dies after five days.",
        "How do I pair a new light strip with the app?",
    ]
    assert find_near_duplicates(texts, threshold=0.7) == {}


# ---------- Balance ----------


def test_cap_per_class_is_deterministic() -> None:
    examples = [make_example(f"o{i}", "order_status") for i in range(10)] + [make_example("r1", "refund_request")]
    kept, removed = cap_per_class(examples, cap=3, seed=1)
    assert sum(1 for example in kept if example.triage.intent.value == "order_status") == 3
    assert any(example.id == "r1" for example in kept)
    assert len(removed) == 7
    again, _ = cap_per_class(list(reversed(examples)), cap=3, seed=1)
    assert {example.id for example in again} == {example.id for example in kept}


# ---------- Full funnel ----------


def test_run_filter_funnel() -> None:
    agree = make_triage()
    tickets = [
        make_labeled("ok1", "My Glow bulbs order BL-123456 has not arrived after two weeks.", agree, agree),
        make_labeled("short", "hi", agree, agree),
        make_labeled("invalid", "Where is my order? It has been ten days now, please check.", agree, None),
        make_labeled(
            "disagree",
            "The thermostat display shows an error code E42 since Monday.",
            agree,
            make_triage(urgency="high"),
        ),
        make_labeled("dup", "My Glow bulbs order BL-654321 has not arrived after two weeks!", agree, agree),
        make_labeled("pii", "Please email me at jane@example.com about my camera refund status.", agree, agree),
    ]
    config = FilterConfig(min_chars=10, max_per_intent=100)
    result = run_filter(config, tickets, requested=8)
    steps = {step.name: step for step in result.funnel}
    assert steps["requested"].kept == 8
    assert steps["generated"].dropped == 2
    assert steps["length"].reasons == {"too_short": 1}
    assert steps["schema_valid"].reasons == {"teacher_2_invalid": 1}
    assert steps["teacher_agreement"].reasons == {"disagree_urgency": 1}
    assert steps["dedup"].dropped == 1
    assert steps["pii_scrub"].modified == 1 and steps["pii_scrub"].reasons == {"email": 1}
    assert [example.id for example in result.examples] == ["ok1", "pii"]
    assert "[EMAIL]" in result.examples[1].text
    assert {dropped.stage for dropped in result.dropped} == {"length", "schema", "agreement", "dedup"}


def test_run_filter_without_agreement_requirement_keeps_disagreements() -> None:
    tickets = [
        make_labeled("d", "The thermostat display shows an error code.", make_triage(), make_triage(urgency="high"))
    ]
    result = run_filter(FilterConfig(require_agreement=False), tickets)
    assert [example.id for example in result.examples] == ["d"]
    assert result.funnel[0].name == "input"
