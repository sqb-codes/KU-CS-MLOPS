"""
Shared pytest fixtures.

The end-to-end training run is expensive, so it executes ONCE per
test session and writes only to pytest's temporary directory and a
temporary MLflow sqlite database - never to the project's real
models/, reports/ or mlflow.db.
"""

import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def training_run(tmp_path_factory):

    from train import run_training

    base = tmp_path_factory.mktemp("training_run")
    tracking_uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"

    summary, comparison = run_training(
        model_dir=base / "models",
        reports_dir=base / "reports",
        tracking_uri=tracking_uri,
        experiment_name="pytest-churn",
        artifact_dir=base / "mlruns",
    )

    # Snapshot the registry immediately after training, before any
    # other test registers models in this shared database.
    from mlflow import MlflowClient
    registered_after_training = MlflowClient(
        tracking_uri=tracking_uri
    ).search_registered_models()

    return {
        "registered_after_training": registered_after_training,
        "base": base,
        "tracking_uri": tracking_uri,
        "summary": summary,
        "comparison": comparison,
    }
