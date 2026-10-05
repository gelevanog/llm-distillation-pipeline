"""Stage 3, filter: deterministic quality gates with a funnel that explains every dropped ticket.

Order: length bounds -> schema validity (every teacher) -> teacher agreement -> near-duplicates ->
PII scrubbing (rewrites text, drops nothing) -> per-intent balance cap. Everything here is plain
code, so the same input always gives the same dataset and each rule is unit-tested.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel

from distillery.config import FilterConfig
from distillery.dedup import find_near_duplicates
from distillery.label import labels_agree
from distillery.pii import scrub_pii
from distillery.records import DroppedExample, Example, FunnelStep, LabeledTicket


class FilterResult(BaseModel):
    examples: list[Example]
    dropped: list[DroppedExample]
    funnel: list[FunnelStep]


def _stable_order(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def cap_per_class(examples: Sequence[Example], cap: int, seed: int) -> tuple[list[Example], list[Example]]:
    """Keep at most `cap` examples per intent (a deterministic pseudo-random subset)."""
    ordered = sorted(examples, key=lambda example: _stable_order(example.id, seed))
    counts: Counter[str] = Counter()
    kept_ids: set[str] = set()
    for example in ordered:
        intent = example.triage.intent.value
        if counts[intent] < cap:
            counts[intent] += 1
            kept_ids.add(example.id)
    kept = [example for example in examples if example.id in kept_ids]
    removed = [example for example in examples if example.id not in kept_ids]
    return kept, removed


def run_filter(
    config: FilterConfig,
    tickets: Sequence[LabeledTicket],
    *,
    requested: int | None = None,
    seed: int = 42,
) -> FilterResult:
    funnel: list[FunnelStep] = []
    dropped: list[DroppedExample] = []

    if requested is not None and requested >= len(tickets):
        funnel.append(FunnelStep(name="requested", description="Seed specs sent to the generator", kept=requested))
        funnel.append(
            FunnelStep(
                name="generated",
                description="Tickets the generator returned",
                kept=len(tickets),
                dropped=requested - len(tickets),
                reasons={"missing_from_teacher_output": requested - len(tickets)} if requested > len(tickets) else {},
            )
        )
    else:
        funnel.append(FunnelStep(name="input", description="Tickets entering the filter", kept=len(tickets)))

    def drop(ticket: LabeledTicket, stage: str, reason: str) -> None:
        dropped.append(DroppedExample(id=ticket.id, stage=stage, reason=reason, text=ticket.text))

    # 1. Length bounds
    survivors: list[LabeledTicket] = []
    reasons: Counter[str] = Counter()
    for ticket in tickets:
        length = len(ticket.text)
        if length < config.min_chars:
            reasons["too_short"] += 1
            drop(ticket, "length", f"too_short ({length} chars)")
        elif length > config.max_chars:
            reasons["too_long"] += 1
            drop(ticket, "length", f"too_long ({length} chars)")
        else:
            survivors.append(ticket)
    funnel.append(
        FunnelStep(
            name="length",
            description=f"{config.min_chars}-{config.max_chars} characters",
            kept=len(survivors),
            dropped=sum(reasons.values()),
            reasons=dict(reasons),
        )
    )

    # 2. Schema validity: every teacher must have produced a valid TicketTriage
    tickets_in, survivors, reasons = survivors, [], Counter()
    for ticket in tickets_in:
        invalid = [index for index, label in enumerate(ticket.labels) if label.triage is None]
        if invalid or not ticket.labels:
            reason = f"teacher_{invalid[0] + 1}_invalid" if invalid else "no_labels"
            reasons[reason] += 1
            first_error = ticket.labels[invalid[0]].error if invalid else None
            drop(ticket, "schema", f"{reason}: {first_error or ''}"[:300])
        else:
            survivors.append(ticket)
    funnel.append(
        FunnelStep(
            name="schema_valid",
            description="Every teacher answer validates against the Pydantic schema (after one repair retry)",
            kept=len(survivors),
            dropped=sum(reasons.values()),
            reasons=dict(reasons),
        )
    )

    # 3. Teacher agreement (self-consistency)
    if config.require_agreement and survivors and len(survivors[0].labels) > 1:
        tickets_in, survivors, reasons = survivors, [], Counter()
        for ticket in tickets_in:
            agree, fields = labels_agree(ticket.labels, config.agreement_fields)
            if agree:
                survivors.append(ticket)
            else:
                for field in fields:
                    reasons[f"disagree_{field}"] += 1
                drop(ticket, "agreement", "teachers disagree on " + ", ".join(fields))
        funnel.append(
            FunnelStep(
                name="teacher_agreement",
                description="Teachers agree on " + ", ".join(config.agreement_fields),
                kept=len(survivors),
                dropped=len(tickets_in) - len(survivors),
                reasons=dict(reasons),
            )
        )

    # 4. Near-duplicates
    duplicates = find_near_duplicates(
        [ticket.text for ticket in survivors],
        threshold=config.dedup_threshold,
        num_perm=config.dedup_num_perm,
        shingle_size=config.dedup_shingle_size,
    )
    kept_after_dedup = []
    for index, ticket in enumerate(survivors):
        if index in duplicates:
            original, similarity = duplicates[index]
            drop(ticket, "dedup", f"near-duplicate of {survivors[original].id} (jaccard {similarity})")
        else:
            kept_after_dedup.append(ticket)
    funnel.append(
        FunnelStep(
            name="dedup",
            description=f"MinHash near-duplicates (Jaccard >= {config.dedup_threshold}) removed",
            kept=len(kept_after_dedup),
            dropped=len(duplicates),
            reasons={"near_duplicate": len(duplicates)} if duplicates else {},
        )
    )

    # 5. PII scrubbing (rewrites, never drops)
    examples: list[Example] = []
    pii_counts: Counter[str] = Counter()
    modified = 0
    for ticket in kept_after_dedup:
        text = ticket.text
        if config.scrub_pii:
            result = scrub_pii(text)
            if result.changed:
                modified += 1
                pii_counts.update(result.counts)
            text = result.text
        triage = ticket.labels[0].triage
        assert triage is not None  # guaranteed by the schema step
        examples.append(Example(id=ticket.id, text=text, triage=triage, source=ticket.source))
    funnel.append(
        FunnelStep(
            name="pii_scrub",
            description="Emails, phone and card numbers replaced with placeholders",
            kept=len(examples),
            modified=modified,
            reasons=dict(pii_counts),
        )
    )

    # 6. Class balance cap
    kept, removed = cap_per_class(examples, config.max_per_intent, seed)
    by_id = {ticket.id: ticket for ticket in kept_after_dedup}
    removed_reasons: Counter[str] = Counter()
    for example in removed:
        removed_reasons[f"cap_{example.triage.intent.value}"] += 1
        drop(by_id[example.id], "balance", f"over the per-intent cap of {config.max_per_intent}")
    funnel.append(
        FunnelStep(
            name="balance",
            description=f"At most {config.max_per_intent} examples per intent",
            kept=len(kept),
            dropped=len(removed),
            reasons=dict(removed_reasons),
        )
    )
    return FilterResult(examples=kept, dropped=dropped, funnel=funnel)
