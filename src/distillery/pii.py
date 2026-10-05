"""Deterministic PII scrubbing: emails, phone numbers and card numbers become placeholders.

Order numbers (`BL-123456`) are task data, not PII, and are deliberately left alone. Names and
street addresses are not detected (that needs an NER model); see the README roadmap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# 13-19 digits, optionally grouped by spaces/dashes (payment cards).
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# International or national phone numbers: optional +country, then at least 9 digits in groups.
PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]?\d{2,4}){2,4}(?![\w-])")

PLACEHOLDERS = {"email": "[EMAIL]", "card": "[CARD]", "phone": "[PHONE]"}


@dataclass
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return any(self.counts.values())


def _digits(value: str) -> int:
    return sum(char.isdigit() for char in value)


def scrub_pii(text: str) -> ScrubResult:
    counts = {"email": 0, "card": 0, "phone": 0}

    def sub(pattern: re.Pattern[str], kind: str, min_digits: int, source: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            if min_digits and _digits(match.group(0)) < min_digits:
                return match.group(0)
            counts[kind] += 1
            return PLACEHOLDERS[kind]

        return pattern.sub(_replace, source)

    scrubbed = sub(EMAIL, "email", 0, text)
    scrubbed = sub(CARD, "card", 13, scrubbed)
    scrubbed = sub(PHONE, "phone", 9, scrubbed)
    return ScrubResult(text=scrubbed, counts={kind: count for kind, count in counts.items() if count})
