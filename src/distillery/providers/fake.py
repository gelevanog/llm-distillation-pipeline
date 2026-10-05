"""Deterministic offline provider: template-based ticket writer and keyword-based labeler.

Two fake teachers exist so the self-consistency filter has something to do: `fake-teacher-a`
and `fake-teacher-b` use slightly different urgency rules and therefore disagree on some tickets.
`fake-teacher-a` also returns an invalid enum for a deterministic subset of tickets, which
exercises the validation + repair path end to end.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Any

from distillery.config import ModelConfig
from distillery.heuristics import heuristic_triage
from distillery.providers.base import ChatRequest, ChatResponse, ProviderError

PRODUCT_NAMES: dict[str, tuple[str, ...]] = {
    "lighting": ("Glow A19 bulb", "Glow light strip", "Glow GU10 spots"),
    "thermostat": ("Aura thermostat", "Aura Mini thermostat"),
    "camera": ("Sentry 2K camera", "Sentry Outdoor camera"),
    "doorbell": ("Chime Pro doorbell", "Chime doorbell"),
    "app_account": ("Brightloop app", "account"),
    "subscription": ("Protect plan", "Protect subscription"),
    "none": ("order", "stuff"),
}

TEMPLATES: dict[str, tuple[str, ...]] = {
    "order_status": (
        "Where is my order{order}? The {product} was supposed to arrive {day} and tracking has not moved.",
        "My {product} still has not arrived{order}. Can you tell me when it will be delivered?",
        "Tracking for{order} says label created for a week now. Is my {product} even shipped?",
    ),
    "refund_request": (
        "I want a refund for the {product}{order}. It does not do what I need.",
        "Please refund my money for the {product}{order}, I already sent it back {day}.",
        "The {product} stopped working after two weeks. I just want my money back{order}.",
    ),
    "return_exchange": (
        "The {product} arrived damaged{order}. Can you send a replacement?",
        "I would like to exchange the {product}{order} for the white version. How do I return it?",
        "You sent the wrong item{order}, I ordered the {product}. I need a swap.",
    ),
    "billing_issue": (
        "I was charged twice for the {product}{order}. Please fix the double charge.",
        "There is an unknown charge from Brightloop on my card for the {product}. Can you send an invoice?",
        "My payment failed for the {product}{order} but the money left my account.",
    ),
    "cancellation": (
        "Please cancel my order{order} for the {product}, I ordered it by mistake.",
        "I want to cancel the {product} before it ships{order}.",
        "How can I cancel my {product}? I don't use it anymore.",
    ),
    "technical_issue": (
        "My {product} keeps going offline since {day}. I restarted the router twice.",
        "The {product} is not working after the last update, it shows an error.",
        "Since {day} my {product} disconnects every few hours.",
    ),
    "setup_help": (
        "How do I set up the new {product}? The instructions are not clear.",
        "I can't pair my new {product} with the app. What am I doing wrong?",
        "How to install the {product} if I don't have a C wire?",
    ),
    "account_access": (
        "I can't log in to my account, the password reset email never arrives.",
        "My 2FA verification code does not work and I am locked out of the {product}.",
        "I think my account was hacked, someone changed the email on it.",
    ),
    "product_question": (
        "Is the {product} compatible with Google Home?",
        "What is the difference between the {product} and the older model?",
        "Do you have the {product} in stock and does the warranty cover outdoor use?",
    ),
    "feedback": (
        "Just wanted to say I love the {product}, setup took five minutes.",
        "Suggestion: the {product} would be nice with a scheduling feature.",
        "The {product} is great but the app could be faster.",
    ),
    "other": (
        "Hi, we offer SEO services and backlinks for your store. Interested in a partnership?",
        "I am sending my resume for any job opening in your support team.",
        "We would like to sponsor your blog, let's collaborate.",
    ),
}
TONE_OPENERS: dict[str, tuple[str, ...]] = {
    "angry": ("This is unacceptable!! ", "Worst service ever. ", "I am fed up. "),
    "frustrated": ("Again a problem. ", "Honestly frustrating. ", "I'm disappointed. "),
    "neutral": ("", "Hello, ", "Hi team, "),
    "polite": ("Hello, hope you are well. ", "Good morning, ", "Dear support, "),
    "happy": ("Hi! Love your products. ", "Hello, great service so far. ", "Hey, amazing team! "),
}
FILLERS = (
    "I have had Brightloop devices for two years.",
    "My partner noticed it first.",
    "I already checked the help center.",
    "The app is on the latest version.",
    "We live in a small apartment.",
    "I bought it as a gift originally.",
)
DAYS = ("on Monday", "last Friday", "yesterday", "three days ago", "on the 12th")


def _rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _typos(text: str, rng: random.Random) -> str:
    chars = list(text)
    for _ in range(max(1, len(chars) // 40)):
        index = rng.randrange(1, max(2, len(chars) - 1))
        if chars[index].isalpha() and chars[index - 1].isalpha():
            chars[index - 1], chars[index] = chars[index], chars[index - 1]
    return "".join(chars)


def write_fake_ticket(seed: dict[str, Any]) -> str:
    rng = _rng(str(seed.get("id")), "write")
    topic = str(seed.get("topic", "other"))
    product = rng.choice(PRODUCT_NAMES.get(str(seed.get("product", "none")), PRODUCT_NAMES["none"]))
    order = f" #BL-{rng.randrange(100000, 999999)}" if seed.get("order_number") else ""
    body = rng.choice(TEMPLATES.get(topic, TEMPLATES["other"])).format(
        product=product, order=order, day=rng.choice(DAYS)
    )
    text = rng.choice(TONE_OPENERS.get(str(seed.get("tone", "neutral")), ("",))) + body
    extra = {"short": 0, "medium": 1, "long": 3}.get(str(seed.get("length", "short")), 0)
    if extra:
        text += " " + " ".join(rng.sample(FILLERS, extra))
    if seed.get("contact_details"):
        text += (
            f" You can reach me at jane.doe{rng.randrange(10, 99)}@example.com or +1 415 555 01{rng.randrange(10, 99)}."
        )
    style = str(seed.get("style", ""))
    if "typos" in style:
        text = _typos(text, rng)
    if "lowercase" in style:
        text = text.lower().replace(".", "").replace("!", "")
    return text


class FakeProvider:
    """Answers generate/label/repair requests deterministically from `request.payload`."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config

    @property
    def label(self) -> str:
        return f"fake/{self.config.model}"

    @property
    def is_remote(self) -> bool:
        return False

    def _label_item(self, item: dict[str, Any], *, allow_invalid: bool) -> dict[str, Any]:
        text = str(item.get("text", ""))
        strict = self.config.model.endswith("-b")
        triage = heuristic_triage(text, strict_urgency=strict).triage.model_dump(mode="json")
        if allow_invalid and not strict and int(hashlib.sha256(text.encode()).hexdigest(), 16) % 13 == 0:
            triage["intent"] = "refund"  # not a valid enum value -> triggers validation + repair
        return {"id": item.get("id"), **triage}

    async def complete(self, request: ChatRequest) -> ChatResponse:
        started = time.monotonic()
        if request.task == "generate":
            items = [{"id": seed["id"], "text": write_fake_ticket(seed)} for seed in request.payload]
        elif request.task in {"label", "repair"}:
            allow_invalid = request.task == "label"
            items = [self._label_item(item, allow_invalid=allow_invalid) for item in request.payload]
        else:  # pragma: no cover - Task is a closed Literal
            raise ProviderError(f"unknown task {request.task}")
        text = json.dumps({"items": items}, ensure_ascii=False)
        return ChatResponse(
            text=text,
            model=self.config.model,
            latency_s=round(time.monotonic() - started, 4),
            input_tokens=len(request.system.split()) + len(request.user.split()),
            output_tokens=len(text.split()),
        )
