from __future__ import annotations

from typing import Any

import pytest

from distillery.records import Example, LabeledTicket, TeacherLabel
from distillery.schema import TicketTriage


def make_triage(**overrides: Any) -> TicketTriage:
    data: dict[str, Any] = {
        "intent": "order_status",
        "urgency": "medium",
        "sentiment": "neutral",
        "product_area": "lighting",
        "order_id": "BL-123456",
        "summary": "Customer asks where their bulb order is.",
    }
    data.update(overrides)
    return TicketTriage.model_validate(data)


def make_example(example_id: str, intent: str = "order_status", text: str | None = None) -> Example:
    return Example(id=example_id, text=text or f"Ticket {example_id} about {intent}", triage=make_triage(intent=intent))


def make_labeled(ticket_id: str, text: str, *labels: TicketTriage | None) -> LabeledTicket:
    return LabeledTicket(
        id=ticket_id,
        text=text,
        labels=[
            TeacherLabel(model=f"teacher-{index}", triage=label, error=None if label else "invalid")
            for index, label in enumerate(labels)
        ],
    )


@pytest.fixture
def triage() -> TicketTriage:
    return make_triage()
