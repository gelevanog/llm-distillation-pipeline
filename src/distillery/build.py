"""Stage 4, build: stratified train/val split, chat-format JSONL for SFT, and a dataset card."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from distillery.metrics import percentile
from distillery.prompts import STUDENT_SYSTEM_PROMPT
from distillery.records import Example, FunnelStep
from distillery.schema import ENUM_FIELDS, TicketTriage


def stratified_split(
    examples: Sequence[Example], val_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Split per intent so train and val have the same label mix. Every intent with 2+ examples gets 1+ val example."""
    rng = random.Random(seed)
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.triage.intent.value].append(example)
    train: list[Example] = []
    val: list[Example] = []
    for intent in sorted(groups):
        group = sorted(groups[intent], key=lambda example: example.id)
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction) if val_fraction > 0 else 0
        if val_fraction > 0 and n_val == 0 and len(group) >= 2:
            n_val = 1
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def chat_messages(text: str, system_prompt: str = STUDENT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Prompt messages for the student (no answer): used for training and inference alike."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]


def to_chat_record(example: Example) -> dict[str, Any]:
    return {
        "id": example.id,
        "messages": [*chat_messages(example.text), {"role": "assistant", "content": example.triage.to_json()}],
    }


def record_to_example(record: dict[str, Any]) -> Example:
    """Inverse of `to_chat_record` (reads train/val JSONL back for calibration and tests)."""
    messages = record["messages"]
    user = next(message["content"] for message in messages if message["role"] == "user")
    answer = next(message["content"] for message in messages if message["role"] == "assistant")
    return Example(id=record["id"], text=user, triage=TicketTriage.model_validate_json(answer))


def distribution(examples: Sequence[Example]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field in ENUM_FIELDS:
        counts = Counter(getattr(example.triage, field).value for example in examples)
        result[field] = dict(counts.most_common())
    result["order_id"] = {
        "present": sum(1 for example in examples if example.triage.order_id),
        "null": sum(1 for example in examples if not example.triage.order_id),
    }
    return result


def dataset_stats(
    train: Sequence[Example],
    val: Sequence[Example],
    funnel: Sequence[FunnelStep],
    label_stats: dict[str, Any] | None,
    teachers: Sequence[str],
) -> dict[str, Any]:
    everything = [*train, *val]
    chars = [len(example.text) for example in everything]
    words = [len(example.text.split()) for example in everything]
    return {
        "created": datetime.now(UTC).date().isoformat(),
        "teachers": list(teachers),
        "sizes": {"train": len(train), "val": len(val), "total": len(everything)},
        "sources": dict(Counter(example.source for example in everything)),
        "text_chars": {
            "min": min(chars, default=0),
            "p50": round(percentile(chars, 50)),
            "p95": round(percentile(chars, 95)),
            "max": max(chars, default=0),
        },
        "text_words_p50": round(percentile(words, 50)),
        "distribution": {"train": distribution(train), "val": distribution(val), "all": distribution(everything)},
        "funnel": [step.model_dump() for step in funnel],
        "label_agreement": label_stats or {},
    }


def dataset_card(stats: dict[str, Any], name: str) -> str:
    """Markdown dataset card (committed for real runs so the README numbers are reproducible)."""
    sizes = stats["sizes"]
    lines = [
        f"# Dataset card: {name}",
        "",
        "Synthetic customer-support tickets for Brightloop (a fictional smart-home store), generated and",
        "labeled by teacher LLMs, then filtered by deterministic rules. Task: ticket -> strict JSON triage",
        "(intent, urgency, sentiment, product_area, order_id, summary).",
        "",
        f"- Created: {stats['created']}",
        f"- Teachers: {', '.join(f'`{teacher}`' for teacher in stats['teachers'])}",
        f"- Size: {sizes['total']} examples ({sizes['train']} train / {sizes['val']} validation, stratified by intent)",
        f"- Text length: median {stats['text_chars']['p50']} characters (p95 {stats['text_chars']['p95']}), "
        f"median {stats['text_words_p50']} words",
        "- Format: chat JSONL (`system`, `user` = ticket, `assistant` = compact JSON answer)",
        "- PII: emails, phone and card numbers replaced with `[EMAIL]`, `[PHONE]`, `[CARD]`",
        "- Not included: the hand-labeled gold test set (`data/gold/`), which is never generated by a teacher",
        "",
        "## Filter funnel",
        "",
        "| Step | Rule | Kept | Dropped | Details |",
        "|---|---|---:|---:|---|",
    ]
    for step in stats["funnel"]:
        details = ", ".join(f"{key}: {value}" for key, value in step["reasons"].items())
        if step.get("modified"):
            details = f"{step['modified']} rewritten" + (f" ({details})" if details else "")
        dropped = step["dropped"] or ""
        lines.append(f"| {step['name']} | {step['description']} | {step['kept']} | {dropped} | {details} |")

    agreement = stats.get("label_agreement") or {}
    if agreement.get("fields"):
        lines += [
            "",
            "## Teacher agreement (before filtering)",
            "",
            f"Both teachers returned a valid label for {agreement['both_valid']} of {agreement['tickets']} tickets; "
            f"all structured fields matched on {agreement['all_fields_agreement']:.1%} of those.",
            "",
            "| Field | Agreement | Cohen's kappa |",
            "|---|---:|---:|",
        ]
        for field, values in agreement["fields"].items():
            lines.append(f"| {field} | {values['agreement']:.1%} | {values['kappa']:.2f} |")

    lines += ["", "## Label distribution (train + val)", ""]
    for field, counts in stats["distribution"]["all"].items():
        total = sum(counts.values()) or 1
        parts = ", ".join(f"{label} {count} ({count / total:.0%})" for label, count in counts.items())
        lines.append(f"- **{field}**: {parts}")
    lines.append("")
    return "\n".join(lines)
