"""Stage 1, generate: diverse synthetic tickets from a seed matrix, or import real unlabeled data.

The seed matrix crosses topic (intended intent) x product x tone x length x writing style x
"mentions an order number" x "includes contact details". Seeds are sampled round-robin over topics
so every intent is covered, and several seeds go into one teacher call.
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from distillery.config import PipelineConfig
from distillery.io import iter_jsonl
from distillery.logging_config import get_logger
from distillery.prompts import GENERATION_SYSTEM_PROMPT, generation_json_schema, generation_user_prompt
from distillery.providers.base import ChatRequest
from distillery.providers.client import LLMClient
from distillery.records import RawTicket
from distillery.schema import Intent, ProductArea, batch_items

log = get_logger(__name__)

TONES = ("angry", "frustrated", "neutral", "polite", "happy")
LENGTHS = ("short", "medium", "long")
STYLES = (
    "clean and well written",
    "several typos and missing punctuation",
    "all lowercase, chat style, no greeting",
    "non-native English speaker",
    "formal email with greeting and sign-off",
    "very terse, a few words",
)
# Topics that usually come with an order number.
ORDER_TOPICS = {
    Intent.ORDER_STATUS,
    Intent.REFUND_REQUEST,
    Intent.RETURN_EXCHANGE,
    Intent.CANCELLATION,
    Intent.BILLING_ISSUE,
}
# Products that make sense per topic (keeps the matrix realistic).
TOPIC_PRODUCTS: dict[Intent, tuple[ProductArea, ...]] = {
    Intent.ACCOUNT_ACCESS: (ProductArea.APP_ACCOUNT,),
    Intent.CANCELLATION: (*[area for area in ProductArea if area is not ProductArea.APP_ACCOUNT],),
    Intent.BILLING_ISSUE: (ProductArea.SUBSCRIPTION, ProductArea.NONE, ProductArea.CAMERA, ProductArea.THERMOSTAT),
    Intent.OTHER: (ProductArea.NONE,),
}
DEFAULT_PRODUCTS = tuple(area for area in ProductArea if area is not ProductArea.NONE)


def build_seed_matrix(
    num: int, rng: random.Random, topics: Sequence[Intent] | None = None, prefix: str = "s"
) -> list[dict[str, Any]]:
    """Sample `num` seed specs, cycling through `topics` (default: every intent) so each gets the same share."""
    topics = list(topics or Intent)
    seeds: list[dict[str, Any]] = []
    for index in range(num):
        topic = topics[index % len(topics)]
        products = TOPIC_PRODUCTS.get(topic, DEFAULT_PRODUCTS)
        tone = rng.choice(TONES)
        if topic is Intent.FEEDBACK and tone == "neutral":
            tone = rng.choice(("happy", "frustrated"))
        seeds.append(
            {
                "id": f"{prefix}{index:05d}",
                "topic": topic.value,
                "product": rng.choice(products).value,
                "tone": tone,
                "length": rng.choices(LENGTHS, weights=(4, 4, 2))[0],
                "style": rng.choice(STYLES),
                "order_number": rng.random() < (0.7 if topic in ORDER_TOPICS else 0.12),
                "contact_details": rng.random() < 0.15,
            }
        )
    rng.shuffle(seeds)
    return seeds


def _chunks(items: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


async def generate_synthetic(
    config: PipelineConfig,
    client: LLMClient,
    *,
    num: int | None = None,
    topics: Sequence[Intent] | None = None,
    prefix: str = "s",
) -> list[RawTicket]:
    # The default pass keeps the plain seed so earlier runs (and their cached answers) stay reproducible.
    rng = random.Random(config.seed if prefix == "s" else f"{config.seed}-{prefix}")
    seeds = build_seed_matrix(num if num is not None else config.generate.num_tickets, rng, topics, prefix)
    batches = _chunks(seeds, config.generate.tickets_per_call)
    log.info("generate.start", tickets=len(seeds), calls=len(batches), model=client.label)

    results = await asyncio.gather(*(_generate_batch(batch, client) for batch in batches))
    tickets = [ticket for batch in results for ticket in batch]
    log.info("generate.done", requested=len(seeds), generated=len(tickets))
    return tickets


async def _generate_batch(seeds: list[dict[str, Any]], client: LLMClient) -> list[RawTicket]:
    request = ChatRequest(
        task="generate",
        system=GENERATION_SYSTEM_PROMPT,
        user=generation_user_prompt(seeds),
        json_schema=generation_json_schema(),
        schema_name="generated_tickets",
        payload=seeds,
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # one failed batch must not sink the whole run
        log.warning("generate.batch_failed", seeds=[seed["id"] for seed in seeds], error=str(exc)[:300])
        return []
    items = batch_items(response.text) or []
    by_id = {seed["id"]: seed for seed in seeds}
    tickets: list[RawTicket] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        seed = by_id.get(str(item.get("id")))
        text = item.get("text")
        if seed is None or not isinstance(text, str) or not text.strip():
            continue
        tickets.append(RawTicket(id=seed["id"], text=text.strip(), seed=seed, generator_model=response.model))
    if len(tickets) < len(seeds):
        log.warning("generate.batch_incomplete", requested=len(seeds), returned=len(tickets))
    return tickets


def import_tickets(path: Path) -> list[RawTicket]:
    """Real unlabeled tickets: CSV with a `text` column (optional `id`) or JSONL with a `text` field."""
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = list(iter_jsonl(path))
    else:
        raise ValueError(f"unsupported import format {path.suffix!r}; use .csv or .jsonl")
    tickets: list[RawTicket] = []
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: row {index + 1} has no `text`")
        ticket_id = str(row.get("id") or f"imp{index:05d}")
        tickets.append(RawTicket(id=ticket_id, text=text.strip(), source="imported"))
    return tickets


def seed_summary(tickets: Sequence[RawTicket]) -> dict[str, dict[str, int]]:
    """Counts per seed dimension (for the dataset card)."""
    summary: dict[str, dict[str, int]] = {}
    for ticket in tickets:
        for key, value in (ticket.seed or {}).items():
            if key == "id":
                continue
            bucket = summary.setdefault(key, {})
            label = json.dumps(value) if isinstance(value, bool) else str(value)
            bucket[label] = bucket.get(label, 0) + 1
    return summary
