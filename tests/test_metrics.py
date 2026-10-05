from __future__ import annotations

import pytest

from distillery.metrics import accuracy, cohen_kappa, macro_f1, per_class_f1, percentile, rouge_l


def test_accuracy() -> None:
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == 0.75
    assert accuracy([], []) == 0.0


def test_macro_f1_known_values() -> None:
    gold = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    # a: P=1, R=0.5 -> F1 2/3; b: P=2/3, R=1 -> F1 0.8
    assert per_class_f1(gold, predicted, ["a", "b"]) == pytest.approx({"a": 2 / 3, "b": 0.8})
    assert macro_f1(gold, predicted) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_invalid_predictions_are_misses() -> None:
    gold = ["a", "b"]
    assert macro_f1(gold, ["__invalid__", "b"], labels=["a", "b"]) == pytest.approx(0.5)


def test_macro_f1_over_fixed_label_set_counts_absent_classes_as_zero() -> None:
    assert macro_f1(["a"], ["a"], labels=["a", "b"]) == pytest.approx(0.5)


def test_cohen_kappa() -> None:
    assert cohen_kappa(["x", "y", "x", "y"], ["x", "y", "x", "y"]) == pytest.approx(1.0)
    # Observed 0.5, expected 0.5 -> kappa 0
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)
    assert cohen_kappa(["x", "x"], ["x", "x"]) == 1.0


def test_rouge_l() -> None:
    assert rouge_l("customer wants a refund", "customer wants a refund") == pytest.approx(1.0)
    assert rouge_l("customer wants a refund", "refund please") == pytest.approx(2 * (1 / 2) * (1 / 4) / (1 / 2 + 1 / 4))
    assert rouge_l("", "anything") == 0.0


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) == 0.0
