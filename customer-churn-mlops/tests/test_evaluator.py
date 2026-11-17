"""Tests for metric calculation and model selection."""

import pytest

from src.evaluator import compute_metrics, select_best_model


def test_compute_metrics_perfect_predictions():

    metrics = compute_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_proba=[0.1, 0.2, 0.8, 0.9],
    )

    assert set(metrics) == {
        "accuracy", "precision", "recall", "f1", "roc_auc"
    }
    assert all(value == 1.0 for value in metrics.values())


def test_select_best_model_uses_roc_auc():

    results = {
        "a": {"roc_auc": 0.80, "f1": 0.70},
        "b": {"roc_auc": 0.85, "f1": 0.50},
    }

    assert select_best_model(results) == "b"


def test_select_best_model_uses_f1_as_tie_breaker():

    results = {
        "a": {"roc_auc": 0.85, "f1": 0.55},
        "b": {"roc_auc": 0.85, "f1": 0.60},
    }

    assert select_best_model(results) == "b"


def test_select_best_model_rejects_empty_input():

    with pytest.raises(ValueError):
        select_best_model({})
