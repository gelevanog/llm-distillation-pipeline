"""Deterministic keyword triage.

Used as the offline `fake` teacher and the `fake` student, so the whole pipeline, the tests, CI and
the Docker demo run without API keys or model downloads. It is intentionally simple; its job is to
exercise the pipeline honestly, not to compete with a real model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from distillery.schema import Intent, ProductArea, Sentiment, TicketTriage, Urgency, normalize_order_id

_ORDER_ID = re.compile(r"(?:#\s*)?\b(?:BL[\s-]?)?(\d{6})\b", re.IGNORECASE)
_ORDER_CONTEXT = re.compile(r"order|#|\bbl[\s-]?\d{6}", re.IGNORECASE)

# Checked in order; the first intent with a keyword hit wins.
INTENT_KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.OTHER, ("partnership", "collaborat", "seo services", "resume", "job opening", "sponsor", "backlinks")),
    (
        Intent.ACCOUNT_ACCESS,
        (
            "password",
            "log in",
            "login",
            "logged out",
            "sign in",
            "2fa",
            "two-factor",
            "verification code",
            "hacked",
            "locked out",
        ),
    ),
    (
        Intent.BILLING_ISSUE,
        (
            "charged twice",
            "double charge",
            "charged me",
            "invoice",
            "receipt",
            "payment failed",
            "card was declined",
            "unknown charge",
            "wrong amount",
            "billed",
        ),
    ),
    (Intent.CANCELLATION, ("cancel",)),
    (Intent.REFUND_REQUEST, ("refund", "money back", "reimburse")),
    (
        Intent.RETURN_EXCHANGE,
        ("return", "exchange", "replacement", "swap", "arrived damaged", "arrived broken", "wrong item"),
    ),
    (
        Intent.ORDER_STATUS,
        (
            "where is my order",
            "tracking",
            "not arrived",
            "hasn't arrived",
            "has not arrived",
            "delivery",
            "shipped",
            "package",
            "still waiting",
        ),
    ),
    (Intent.SETUP_HELP, ("how do i", "how to", "set up", "setup", "install", "pair", "configure", "new ")),
    (
        Intent.TECHNICAL_ISSUE,
        ("offline", "not working", "stopped", "crash", "error", "disconnect", "won't", "doesn't", "broken", "keeps"),
    ),
    (
        Intent.PRODUCT_QUESTION,
        ("compatible", "does the", "do you", "difference", "in stock", "warranty", "support for", "?"),
    ),
    (Intent.FEEDBACK, ("love", "great", "suggestion", "feature request", "would be nice", "awesome", "thank")),
]

PRODUCT_KEYWORDS: list[tuple[ProductArea, tuple[str, ...]]] = [
    (ProductArea.SUBSCRIPTION, ("protect", "subscription", "cloud", "plan", "recordings")),
    (ProductArea.DOORBELL, ("doorbell", "chime")),
    (ProductArea.CAMERA, ("camera", "sentry", "cam ")),
    (ProductArea.THERMOSTAT, ("thermostat", "aura", "heating", "heat", "cooling")),
    (ProductArea.LIGHTING, ("bulb", "light", "lamp", "glow", "strip")),
    (ProductArea.APP_ACCOUNT, ("app", "account", "password", "login", "log in", "email")),
]

NEGATIVE_WORDS = (
    "angry",
    "terrible",
    "ridiculous",
    "unacceptable",
    "worst",
    "frustrat",
    "disappoint",
    "annoy",
    "!!",
    "useless",
    "awful",
    "fed up",
    "still waiting",
    "again",
    "wtf",
    "joke",
)
POSITIVE_WORDS = ("love", "great", "awesome", "amazing", "fantastic", "happy", "excellent", "perfect")
HIGH_URGENCY_WORDS = (
    "hacked",
    "smoke",
    "spark",
    "burning",
    "overheat",
    "no heat",
    "freezing",
    "not heating",
    "security",
    "break-in",
    "chargeback",
    "lawyer",
    "legal",
    "third time",
    "3rd time",
    "today",
    "asap",
    "urgent",
    "immediately",
)
LOW_URGENCY_INTENTS = {Intent.PRODUCT_QUESTION, Intent.FEEDBACK, Intent.OTHER, Intent.SETUP_HELP}


@dataclass(frozen=True)
class HeuristicResult:
    triage: TicketTriage
    # 0..1: how many independent keyword signals backed the decision (used as the fake student's confidence).
    confidence: float


def _first_hit(text: str, table: list[tuple[Intent, tuple[str, ...]]]) -> tuple[Intent, int]:
    for intent, words in table:
        hits = sum(1 for word in words if word in text)
        if hits:
            return intent, hits
    return Intent.OTHER, 0


def extract_order_id(text: str) -> str | None:
    if not _ORDER_CONTEXT.search(text):
        return None
    match = _ORDER_ID.search(text)
    return normalize_order_id(match.group(1)) if match else None


def _summary(intent: Intent, product: ProductArea, order_id: str | None) -> str:
    topic = intent.value.replace("_", " ")
    about = "" if product is ProductArea.NONE else f" about their {product.value.replace('_', ' ')}"
    order = f" (order {order_id})" if order_id else ""
    return f"Customer has a {topic} request{about}{order}."


def heuristic_triage(text: str, *, strict_urgency: bool = False) -> HeuristicResult:
    """Keyword triage. `strict_urgency` is a slightly different policy, used by the second fake teacher."""
    lowered = f" {text.lower()} "
    intent, intent_hits = _first_hit(lowered, INTENT_KEYWORDS)
    product = next(
        (area for area, words in PRODUCT_KEYWORDS if any(word in lowered for word in words)), ProductArea.NONE
    )
    negative = sum(1 for word in NEGATIVE_WORDS if word in lowered)
    positive = sum(1 for word in POSITIVE_WORDS if word in lowered)
    if negative > positive:
        sentiment = Sentiment.NEGATIVE
    elif positive > negative:
        sentiment = Sentiment.POSITIVE
    else:
        sentiment = Sentiment.NEUTRAL

    high_hits = sum(1 for word in HIGH_URGENCY_WORDS if word in lowered)
    if high_hits >= (2 if strict_urgency else 1):
        urgency = Urgency.HIGH
    elif intent in LOW_URGENCY_INTENTS:
        urgency = Urgency.LOW
    else:
        urgency = Urgency.MEDIUM

    order_id = extract_order_id(text)
    triage = TicketTriage(
        intent=intent,
        urgency=urgency,
        sentiment=sentiment,
        product_area=product,
        order_id=order_id,
        summary=_summary(intent, product, order_id),
    )
    confidence = min(1.0, 0.35 + 0.2 * intent_hits + (0.1 if product is not ProductArea.NONE else 0.0))
    return HeuristicResult(triage=triage, confidence=round(confidence, 3))
