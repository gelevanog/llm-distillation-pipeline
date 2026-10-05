"""The task contract: a support ticket goes in, a strict `TicketTriage` JSON object comes out.

Every stage shares this module: the teacher is asked for exactly this schema, the filter validates
against it, the student is trained to emit it, and the evaluation scores it field by field.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Intent(StrEnum):
    ORDER_STATUS = "order_status"
    REFUND_REQUEST = "refund_request"
    RETURN_EXCHANGE = "return_exchange"
    BILLING_ISSUE = "billing_issue"
    CANCELLATION = "cancellation"
    TECHNICAL_ISSUE = "technical_issue"
    SETUP_HELP = "setup_help"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_QUESTION = "product_question"
    FEEDBACK = "feedback"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class ProductArea(StrEnum):
    LIGHTING = "lighting"
    THERMOSTAT = "thermostat"
    CAMERA = "camera"
    DOORBELL = "doorbell"
    APP_ACCOUNT = "app_account"
    SUBSCRIPTION = "subscription"
    NONE = "none"


ENUM_FIELDS: dict[str, type[StrEnum]] = {
    "intent": Intent,
    "urgency": Urgency,
    "sentiment": Sentiment,
    "product_area": ProductArea,
}
# Fields with an exact ground truth. `summary` is free text and is scored separately.
STRUCTURED_FIELDS: tuple[str, ...] = ("intent", "urgency", "sentiment", "product_area", "order_id")
ALL_FIELDS: tuple[str, ...] = (*STRUCTURED_FIELDS, "summary")

ORDER_ID_PATTERN = re.compile(r"^BL-\d{6}$")
_ORDER_ID_LOOSE = re.compile(r"^#?\s*(?:BL)?[\s-]*(\d{6})$", re.IGNORECASE)
SUMMARY_MAX_WORDS = 30


def normalize_order_id(value: str | None) -> str | None:
    """Canonical `BL-123456` form; accepts `#BL-123456`, `bl123456`, `BL 123456` or `123456`."""
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "" or cleaned.lower() in {"null", "none", "n/a"}:
        return None
    match = _ORDER_ID_LOOSE.match(cleaned)
    if match:
        return f"BL-{match.group(1)}"
    return cleaned.upper()


class TicketTriage(BaseModel):
    """Structured triage of one customer-support ticket (Brightloop smart-home store)."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    intent: Intent
    urgency: Urgency
    sentiment: Sentiment
    product_area: ProductArea
    order_id: str | None = Field(description="Order number as BL-XXXXXX (6 digits), or null")
    summary: str = Field(min_length=8, max_length=240, description="One sentence, at most 30 words")

    @field_validator("order_id", mode="before")
    @classmethod
    def _normalize_order_id(cls, value: Any) -> Any:
        return normalize_order_id(value) if isinstance(value, str) or value is None else value

    @field_validator("order_id")
    @classmethod
    def _check_order_id(cls, value: str | None) -> str | None:
        if value is not None and not ORDER_ID_PATTERN.match(value):
            raise ValueError("order_id must look like BL-123456 (6 digits) or be null")
        return value

    @field_validator("summary")
    @classmethod
    def _check_summary(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value.split()) > SUMMARY_MAX_WORDS:
            raise ValueError(f"summary must be at most {SUMMARY_MAX_WORDS} words")
        return value

    def to_json(self) -> str:
        """Compact JSON in schema field order: the exact string the student learns to produce."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, separators=(", ", ": "))

    def structured(self) -> dict[str, str | None]:
        data = self.model_dump(mode="json")
        return {field: data[field] for field in STRUCTURED_FIELDS}


def triage_json_schema() -> dict[str, Any]:
    """JSON schema in the subset that strict structured-output modes accept (OpenAI, OpenRouter, Anthropic).

    Hand-written instead of `model_json_schema()` because strict modes reject `$defs`, `title`,
    `pattern`, length limits and defaults on some providers; Pydantic enforces those afterwards.
    """
    properties: dict[str, Any] = {
        name: {"type": "string", "enum": [member.value for member in enum]} for name, enum in ENUM_FIELDS.items()
    }
    properties["order_id"] = {"type": ["string", "null"], "description": "BL-XXXXXX or null"}
    properties["summary"] = {"type": "string", "description": "One sentence, at most 30 words"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(ALL_FIELDS),
        "additionalProperties": False,
    }


def batch_json_schema(item_schema: dict[str, Any], id_field: str = "id") -> dict[str, Any]:
    """Wrap an item schema into `{"items": [{id, ...item}]}` for batched calls."""
    item = {
        **item_schema,
        "properties": {id_field: {"type": "string"}, **item_schema["properties"]},
        "required": [id_field, *item_schema["required"]],
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item}},
        "required": ["items"],
        "additionalProperties": False,
    }


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and prose around one JSON value.

    Raises `ValueError` when no JSON value can be recovered.
    """
    stripped = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("no JSON value found in model output")


def salvage_items(text: str) -> list[dict[str, Any]]:
    """Recover the complete `{"id": ...}` objects from a batch answer that was cut off mid-way.

    Models sometimes hit `max_tokens` (long reasoning) and return `{"items": [{...}, {...}, {"id": "x", "te`.
    The complete items before the cut are still good data.
    """
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    index = text.find('{"id"')
    while index != -1:
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find('{"id"', index + 1)
            continue
        if isinstance(value, dict) and "id" in value:
            items.append(value)
        index = text.find('{"id"', end)
    return items


def batch_items(text: str) -> list[dict[str, Any]] | None:
    """Items of a batch answer (`{"items": [...]}` or a bare list), salvaging truncated answers; None if unusable."""
    try:
        data = extract_json(text)
    except ValueError:
        data = None
    raw = data.get("items") if isinstance(data, dict) else data
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    salvaged = salvage_items(text)
    return salvaged or None


class ParseOutcome(BaseModel):
    """Result of turning raw model text into a `TicketTriage`."""

    triage: TicketTriage | None = None
    json_valid: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.triage is not None


def format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def validate_triage(data: Any) -> ParseOutcome:
    """Validate an already-parsed JSON value."""
    try:
        return ParseOutcome(triage=TicketTriage.model_validate(data), json_valid=True)
    except ValidationError as exc:
        return ParseOutcome(json_valid=True, error=format_validation_error(exc))


def parse_triage(text: str) -> ParseOutcome:
    """Parse and validate raw model output (what the student and the router see)."""
    try:
        data = extract_json(text)
    except ValueError as exc:
        return ParseOutcome(json_valid=False, error=str(exc))
    if not isinstance(data, dict):
        return ParseOutcome(json_valid=True, error="expected a JSON object")
    return validate_triage(data)
