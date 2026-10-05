"""Prompts for the teacher (generation, labeling, repair) and the student.

The labeling guide is the single source of truth for what each label means. The gold test set in
`data/gold/` was labeled by hand against the same guide.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from distillery.schema import Intent, ProductArea, Sentiment, Urgency

COMPANY = "Brightloop"

LABELING_GUIDE = f"""\
You triage customer-support tickets for {COMPANY}, an online store that sells smart-home devices
(smart bulbs and light strips, thermostats, security cameras, video doorbells), the {COMPANY} mobile
app, and {COMPANY} Protect, a paid subscription for cloud video storage.

Return one JSON object per ticket with these fields:

intent (exactly one):
- order_status: where is my order, tracking, delayed or missing delivery (item not received yet).
- refund_request: the customer wants their money back for a product or service they no longer want
  or that failed (including after a return).
- return_exchange: return an item, swap it for another model/colour, or get a replacement for an item
  that arrived damaged or wrong (wants a product, not money).
- billing_issue: an incorrect or unexpected charge (double charge, wrong amount, unknown charge, failed
  payment, invoice/receipt request), even if they ask for the charge to be reversed.
- cancellation: cancel an order that has not shipped yet, or cancel/stop a subscription.
- technical_issue: something that worked before is broken: device offline, app crash, error,
  disconnects, poor video, battery problems, firmware update failed.
- setup_help: first-time installation, pairing or configuration questions ("how do I...", "cannot
  pair my new...") where nothing is described as having worked before.
- account_access: login, password reset, two-factor codes, locked or hacked account, change account email.
- product_question: pre-sales or general product questions: compatibility, features, stock, pricing,
  differences between models, warranty terms.
- feedback: praise, suggestions, feature requests, or a general complaint with no concrete request.
- other: spam, partnership or sales pitches, job applications, anything unrelated to support.

urgency:
- high: security or safety risk (camera/doorbell down so the home is unmonitored, smoke, sparks,
  overheating), heating or cooling not working, account hacked, a hard deadline within 48 hours,
  threats of chargeback or legal action, or the customer says it is at least their third contact.
- medium: something is broken, missing, late or wrong with money/orders, with none of the high factors.
- low: questions, how-to, feedback, other, routine account or subscription changes, no time pressure.

sentiment:
- negative: frustration, anger, disappointment or sarcasm is expressed.
- positive: clear praise or satisfaction (a polite "thanks in advance" alone is neutral).
- neutral: everything else.

product_area (what the ticket is about):
- lighting (bulbs, light strips), thermostat, camera, doorbell
- app_account: the {COMPANY} mobile app or the customer account/login
- subscription: {COMPANY} Protect plan, cloud recordings, subscription billing
- none: no specific product (e.g. a mixed order, partnership pitch)

order_id: the order number normalised to "BL-" plus 6 digits (e.g. "order #bl 482913" -> "BL-482913",
"order 482913" -> "BL-482913"). null if the ticket has no order number. Never invent one.

summary: one neutral sentence in English, at most 25 words, describing the customer's issue and
request, e.g. "Customer's thermostat stopped heating after a firmware update; asks for a fix today."
"""

STUDENT_SYSTEM_PROMPT = f"""\
You triage {COMPANY} smart-home support tickets. Reply with one JSON object only:
{{"intent": one of {[member.value for member in Intent]},
"urgency": one of {[member.value for member in Urgency]},
"sentiment": one of {[member.value for member in Sentiment]},
"product_area": one of {[member.value for member in ProductArea]},
"order_id": "BL-" + 6 digits or null,
"summary": one sentence, at most 25 words}}"""


def label_system_prompt() -> str:
    return LABELING_GUIDE + "\nReply with JSON only. Do not add commentary."


def zero_shot_system_prompt() -> str:
    """The baseline for the student before fine-tuning: the teacher's full guide plus the exact output format.

    The guide alone made Qwen2.5-0.5B invent its own JSON shapes (0 of 8 schema-valid in a smoke test);
    adding the explicit format is the strongest fair prompt for the untrained student.
    """
    output_format = STUDENT_SYSTEM_PROMPT.split("\n", 1)[1]
    return LABELING_GUIDE + "\nReply with one JSON object only, exactly in this format:\n" + output_format


def label_batch_user_prompt(items: Sequence[tuple[str, str]]) -> str:
    """User message for labeling several tickets in one call. `items` are (id, text) pairs."""
    tickets = "\n\n".join(f'<ticket id="{ticket_id}">\n{text}\n</ticket>' for ticket_id, text in items)
    return (
        f"Triage each of the {len(items)} tickets below. Return "
        '{"items": [{"id": "<ticket id>", "intent": ..., "urgency": ..., "sentiment": ..., '
        '"product_area": ..., "order_id": ..., "summary": ...}, ...]} with exactly one item per ticket, '
        "in the same order.\n\n" + tickets
    )


def label_single_user_prompt(text: str) -> str:
    return f"Triage this ticket.\n\n<ticket>\n{text}\n</ticket>"


def repair_user_prompt(original_prompt: str, bad_output: str, errors: Sequence[str]) -> str:
    """Ask the teacher to fix its own output, quoting the validation errors."""
    error_list = "\n".join(f"- {error}" for error in errors)
    return (
        f"{original_prompt}\n\nYour previous answer was rejected by the validator:\n{error_list}\n\n"
        f"Previous answer:\n{bad_output[:4000]}\n\nReturn the corrected JSON only, for the same tickets."
    )


GENERATION_SYSTEM_PROMPT = f"""\
You write realistic synthetic customer-support emails and chat messages for {COMPANY}, an online
store for smart-home devices (smart bulbs and light strips, thermostats, security cameras, video
doorbells), the {COMPANY} mobile app and the {COMPANY} Protect cloud-video subscription.

Each ticket must read like a real customer wrote it: specific details (device model names like
"Glow A19 bulb", "Aura thermostat", "Sentry 2K camera", "Chime Pro doorbell", app versions, dates,
amounts), varied openings, no templates, no placeholders like [Name]. Follow the requested tone,
length and writing style exactly, including typos when asked. When an order number is requested,
use "BL-" plus 6 random digits (a different number every time) and write it the way customers do:
with or without "#", "BL-" or just the digits after the word "order", sometimes lowercase. When
contact details are requested, include a made-up email address or phone number, as real tickets do.
Do not label the tickets; only write them.

What each topic means (write a message that a support agent would file under it):
- order_status: asks where an order is, tracking, a late or missing delivery.
- refund_request: wants money back for a product or service.
- return_exchange: wants to return an item, swap it, or get a replacement for a damaged/wrong item.
- billing_issue: a wrong, double or unknown charge, failed payment, invoice or receipt request.
- cancellation: cancel an order before it ships, or cancel the subscription.
- technical_issue: something that used to work is broken (offline, crash, error, battery, update).
- setup_help: how to install, pair or configure something new.
- account_access: login, password reset, two-factor codes, locked or hacked account, account email.
- product_question: pre-sales questions (compatibility, features, stock, price, warranty).
- feedback: praise, suggestions, feature requests, or a general complaint with no concrete request.
- other: NOT a customer-support request at all: SEO/marketing or partnership pitches, spam, job
  applications, student or press questions, influencer offers, messages sent to the wrong address."""


def generation_user_prompt(seeds: Sequence[dict[str, Any]]) -> str:
    specs = "\n".join(json.dumps(seed, ensure_ascii=False) for seed in seeds)
    return (
        f"Write {len(seeds)} tickets, one per spec below. Return "
        '{"items": [{"id": "<spec id>", "text": "<ticket text>"}, ...]} in the same order.\n\n'
        "Spec fields: topic = what the customer wants (the intended intent), product = what it is about, "
        "tone, length (short: 1-2 sentences, medium: 3-5 sentences, long: 6-10 sentences), "
        "style = writing quirks, order_number = whether to mention an order number.\n\n" + specs
    )


def generation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
