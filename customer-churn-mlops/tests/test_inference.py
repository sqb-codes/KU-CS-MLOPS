"""Tests for loading the saved pipeline and making predictions."""

import pytest

from src.data_loader import load_raw_data
from src.predict import load_pipeline, predict_churn


@pytest.fixture(scope="module")
def pipeline(training_run):
    return load_pipeline(
        training_run["base"] / "models" / "churn_pipeline.joblib"
    )


def test_predict_on_raw_records(pipeline):

    # Raw rows, including customerID and Churn, straight from the CSV.
    records = load_raw_data().head(10)

    predictions = predict_churn(pipeline, records)

    assert len(predictions) == 10
    assert predictions["churn_probability"].between(0, 1).all()
    assert set(predictions["churn_prediction"]).issubset({0, 1})


def test_predict_handles_blank_total_charges(pipeline):

    # New customers have TotalCharges = " " in the raw data.
    record = load_raw_data().iloc[0].to_dict()
    record["TotalCharges"] = " "

    predictions = predict_churn(pipeline, record)

    assert len(predictions) == 1


def test_predict_rejects_missing_features(pipeline):

    with pytest.raises(ValueError, match="missing required features"):
        predict_churn(pipeline, {"tenure": 5})


def test_load_missing_model_raises(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "nope.joblib")
