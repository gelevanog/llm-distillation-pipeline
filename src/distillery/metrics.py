"""Metrics: accuracy, macro-F1, Cohen's kappa, ROUGE-L and latency percentiles (pure Python)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def accuracy(gold: Sequence[str | None], predicted: Sequence[str | None]) -> float:
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, predicted, strict=True) if g == p) / len(gold)


def per_class_f1(gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted, strict=True) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def macro_f1(gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Unweighted mean F1 over the classes present in the gold labels (or the given label set).

    Invalid predictions should be passed as a sentinel (e.g. "__invalid__"): they count as misses
    for the gold class and never as a hit.
    """
    classes = list(labels) if labels is not None else sorted(set(gold))
    if not classes:
        return 0.0
    scores = per_class_f1(gold, predicted, classes)
    return sum(scores.values()) / len(classes)


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Agreement between two annotators corrected for chance (1 = perfect, 0 = chance level)."""
    total = len(first)
    if total == 0:
        return 0.0
    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / total
    count_a, count_b = Counter(first), Counter(second)
    expected = sum(count_a[label] * count_b[label] for label in set(count_a) | set(count_b)) / (total * total)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            current.append(previous[index - 1] + 1 if token_a == token_b else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def _tokens(text: str) -> list[str]:
    return "".join(char.lower() if char.isalnum() else " " for char in text).split()


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1 on lowercase word tokens: a rough lexical-overlap signal for the free-text summary."""
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return 0.0
    lcs = _lcs(ref, cand)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
