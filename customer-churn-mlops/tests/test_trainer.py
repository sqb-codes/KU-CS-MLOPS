"""Tests for candidate training and the end-to-end training run."""

import json

import mlflow

from src.trainer import get_candidate_models


def test_three_candidate_models():

    assert set(get_candidate_models()) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }


def test_training_compares_all_models(training_run):

    comparison = training_run["comparison"]

    assert len(comparison) == 3
    assert comparison["selected"].sum() == 1

    # The selected model must be the validation ROC-AUC leader.
    assert comparison.iloc[0]["selected"]


def test_selected_model_has_test_metrics(training_run):

    summary = training_run["summary"]

    assert summary["selected_model"] in summary["validation_metrics"]

    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= summary["test_metrics"][metric] <= 1.0

    # Sanity floor: churn models on this dataset reach ~0.84 AUC.
    assert summary["test_metrics"]["roc_auc"] > 0.75

    # customerID must never be a feature.
    assert "customerID" not in summary["feature_columns"]


def test_artifacts_are_saved(training_run):

    base = training_run["base"]

    assert (base / "models" / "churn_pipeline.joblib").exists()
    assert (base / "reports" / "model_comparison.csv").exists()

    metadata = json.loads(
        (base / "models" / "model_metadata.json").read_text()
    )
    assert metadata["mlflow_run_id"]


def test_mlflow_run_logged(training_run):

    mlflow.set_tracking_uri(training_run["tracking_uri"])
    run = mlflow.get_run(training_run["summary"]["mlflow_run_id"])

    assert "test_roc_auc" in run.data.metrics
    assert "val_roc_auc" in run.data.metrics
    assert run.data.params["selected_model"] == (
        training_run["summary"]["selected_model"]
    )


def test_training_does_not_register_model(training_run):

    # Registration/promotion is a separate, explicit step.
    assert training_run["registered_after_training"] == []


def test_logged_mlflow_model_predicts_on_raw_features(training_run):

    # Loading through MLflow (not joblib) checks the logged
    # signature, skops trusted types and bundled src/ code together.
    from src.data_loader import load_raw_data
    from src.preprocessing import prepare_features_and_target

    mlflow.set_tracking_uri(training_run["tracking_uri"])
    model = mlflow.pyfunc.load_model(
        training_run["summary"]["mlflow_model_uri"]
    )

    # The MLflow signature expects typed features (TotalCharges as a
    # number, NaN allowed), i.e. the output of
    # prepare_features_and_target().
    X, _ = prepare_features_and_target(load_raw_data())
    sample = X.head(5).copy()
    sample.loc[sample.index[0], "TotalCharges"] = float("nan")

    predictions = model.predict(sample)

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})
