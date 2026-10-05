"""Stage 2, label: structured teacher labels with validation, a repair retry and self-consistency.

Each ticket is labeled by every configured labeler (two different teacher models by default).
Answers are validated with Pydantic; items that fail are sent back to the same teacher once with the
validator's error messages ("repair"). Agreement between labelers is measured per field and is used
by the filter stage.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from distillery.config import PipelineConfig
from distillery.logging_config import get_logger
from distillery.metrics import cohen_kappa
from distillery.pii import scrub_pii
from distillery.prompts import label_batch_user_prompt, label_system_prompt, repair_user_prompt
from distillery.providers.base import ChatRequest, ChatResponse
from distillery.providers.client import LLMClient
from distillery.providers.factory import ClientFactory
from distillery.records import LabeledTicket, RawTicket, TeacherLabel
from distillery.schema import STRUCTURED_FIELDS, batch_items, batch_json_schema, triage_json_schema, validate_triage

log = get_logger(__name__)


class CallStat(BaseModel):
    latency_s: float
    input_tokens: int
    output_tokens: int
    items: int
    cached: bool


class LabelRun(BaseModel):
    labels: dict[str, TeacherLabel] = Field(default_factory=dict)
    calls: list[CallStat] = Field(default_factory=list)


def _request(task: str, items: Sequence[tuple[str, str]], user: str) -> ChatRequest:
    return ChatRequest(
        task="repair" if task == "repair" else "label",
        system=label_system_prompt(),
        user=user,
        json_schema=batch_json_schema(triage_json_schema()),
        schema_name="ticket_triage_batch",
        payload=[{"id": item_id, "text": text} for item_id, text in items],
    )


def parse_batch(text: str, expected_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Split a batch answer into valid items (id -> TicketTriage) and errors (id -> message)."""
    valid: dict[str, Any] = {}
    errors: dict[str, str] = {}
    raw_items = batch_items(text)
    if raw_items is None:
        return {}, dict.fromkeys(expected_ids, "invalid JSON: expected {'items': [...]}")
    expected = set(expected_ids)
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id", ""))
        if item_id not in expected or item_id in valid:
            continue
        outcome = validate_triage({key: value for key, value in raw.items() if key != "id"})
        if outcome.triage is not None:
            valid[item_id] = outcome.triage
            errors.pop(item_id, None)
        else:
            errors[item_id] = outcome.error or "invalid"
    for item_id in expected_ids:
        if item_id not in valid and item_id not in errors:
            errors[item_id] = "missing from the answer"
    return valid, errors


async def label_batch(
    client: LLMClient, items: Sequence[tuple[str, str]], max_repair_attempts: int
) -> tuple[dict[str, TeacherLabel], list[CallStat]]:
    ids = [item_id for item_id, _ in items]
    texts = dict(items)
    user = label_batch_user_prompt(items)
    stats: list[CallStat] = []
    try:
        response = await client.complete(_request("label", items, user))
    except Exception as exc:
        log.warning("label.batch_failed", teacher=client.label, items=len(items), error=str(exc)[:300])
        return {item_id: TeacherLabel(model=client.label, error=f"request failed: {exc}"[:300]) for item_id in ids}, []
    stats.append(_stat(response, len(items)))
    valid, errors = parse_batch(response.text, ids)
    labels = {item_id: TeacherLabel(model=response.model, triage=triage) for item_id, triage in valid.items()}

    last_output = response.text
    for _ in range(max_repair_attempts):
        if not errors:
            break
        failed = [(item_id, texts[item_id]) for item_id in ids if item_id in errors]
        repair_prompt = repair_user_prompt(
            label_batch_user_prompt(failed),
            last_output,
            [f"ticket {item_id}: {error}" for item_id, error in errors.items()],
        )
        try:
            repair = await client.complete(_request("repair", failed, repair_prompt))
        except Exception as exc:
            log.warning("label.repair_failed", teacher=client.label, items=len(failed), error=str(exc)[:300])
            break
        stats.append(_stat(repair, len(failed)))
        fixed, errors = parse_batch(repair.text, [item_id for item_id, _ in failed])
        labels.update(
            {
                item_id: TeacherLabel(model=repair.model, triage=triage, repaired=True)
                for item_id, triage in fixed.items()
            }
        )
        last_output = repair.text
    for item_id, error in errors.items():
        labels[item_id] = TeacherLabel(model=client.label, error=error[:300])
    return labels, stats


def _stat(response: ChatResponse, items: int) -> CallStat:
    return CallStat(
        latency_s=response.latency_s,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        items=items,
        cached=response.cached,
    )


async def label_with_teacher(
    client: LLMClient, items: Sequence[tuple[str, str]], batch_size: int, max_repair_attempts: int
) -> LabelRun:
    batches = [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    results = await asyncio.gather(*(label_batch(client, batch, max_repair_attempts) for batch in batches))
    run = LabelRun()
    for labels, stats in results:
        run.labels.update(labels)
        run.calls.extend(stats)
    return run


async def label_tickets(
    config: PipelineConfig, tickets: Sequence[RawTicket], factory: ClientFactory
) -> list[LabeledTicket]:
    scrub = config.label.scrub_pii_before_teacher
    prepared = [(ticket, scrub_pii(ticket.text).text if scrub else ticket.text) for ticket in tickets]
    items = [(ticket.id, text) for ticket, text in prepared]
    runs: list[LabelRun] = []
    for index, labeler in enumerate(config.teacher.labelers):
        client = factory.client(labeler, stage=f"label[{index}]")
        log.info("label.start", teacher=client.label, tickets=len(items), batch_size=config.label.batch_size)
        run = await label_with_teacher(client, items, config.label.batch_size, config.label.max_repair_attempts)
        ok = sum(1 for label in run.labels.values() if label.triage is not None)
        repaired = sum(1 for label in run.labels.values() if label.repaired)
        log.info("label.done", teacher=client.label, valid=ok, repaired=repaired, invalid=len(items) - ok)
        runs.append(run)
    return [
        # The stored text stays original; the filter stage scrubs it for the training set. With
        # scrub_pii_before_teacher (default) the teacher only ever saw the scrubbed version.
        LabeledTicket(
            id=ticket.id,
            text=ticket.text,
            source=ticket.source,
            seed=ticket.seed,
            labels=[run.labels.get(ticket.id, TeacherLabel(model="?", error="not labeled")) for run in runs],
        )
        for ticket, _ in prepared
    ]


def labels_agree(labels: Sequence[TeacherLabel], fields: Sequence[str]) -> tuple[bool, list[str]]:
    """True when every label is valid and all labels match on `fields`. Returns the disagreeing fields."""
    triages = [label.triage for label in labels]
    if any(triage is None for triage in triages):
        return False, ["invalid"]
    values = [triage.structured() for triage in triages if triage is not None]
    disagreeing = [field for field in fields if len({str(value[field]) for value in values}) > 1]
    return not disagreeing, disagreeing


def agreement_stats(tickets: Sequence[LabeledTicket]) -> dict[str, Any]:
    """Validity and repairs per teacher, plus per-field agreement and Cohen's kappa between teachers 1 and 2."""
    if not tickets:
        return {}
    teachers = len(tickets[0].labels)
    per_teacher = []
    for index in range(teachers):
        labels = [ticket.labels[index] for ticket in tickets]
        models: dict[str, int] = {}
        for label in labels:
            models[label.model] = models.get(label.model, 0) + 1
        per_teacher.append(
            {
                "models": models,
                "valid": sum(1 for label in labels if label.triage is not None),
                "repaired": sum(1 for label in labels if label.repaired),
                "invalid": sum(1 for label in labels if label.triage is None),
            }
        )
    stats: dict[str, Any] = {"tickets": len(tickets), "teachers": per_teacher}
    if teachers < 2:
        return stats
    both = [
        (ticket.labels[0].triage, ticket.labels[1].triage)
        for ticket in tickets
        if ticket.labels[0].triage is not None and ticket.labels[1].triage is not None
    ]
    fields: dict[str, Any] = {}
    for field in STRUCTURED_FIELDS:
        first = [str(a.structured()[field]) for a, _ in both if a is not None]
        second = [str(b.structured()[field]) for _, b in both if b is not None]
        agree = sum(1 for x, y in zip(first, second, strict=True) if x == y)
        fields[field] = {
            "agreement": round(agree / len(both), 4) if both else 0.0,
            "kappa": round(cohen_kappa(first, second), 4) if both else 0.0,
        }
    all_agree = sum(1 for a, b in both if a is not None and b is not None and a.structured() == b.structured())
    stats["both_valid"] = len(both)
    stats["fields"] = fields
    stats["all_fields_agreement"] = round(all_agree / len(both), 4) if both else 0.0
    return stats
