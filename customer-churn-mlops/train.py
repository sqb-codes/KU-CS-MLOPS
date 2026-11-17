"""
train.py

Entry point for local model training.

Workflow:
    1. Load and validate raw data.
    2. Separate features/target (customerID is excluded).
    3. Stratified, reproducible 70/15/15 train/val/test split.
    4. Train 3 candidate pipelines on TRAIN only.
    5. Compare on VALIDATION; select by ROC-AUC (F1 tie-breaker).
    6. Evaluate ONLY the selected model on the untouched TEST set.
    7. Log everything to MLflow and save the selected pipeline.

What this script deliberately does NOT do:
    It does not register the model or change the "champion".
    Promotion is a separate, human-approved step
    (see src/model_registry.py).

Usage:
    python train.py
    python train.py --help
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Silence MLflow's optional console hint; must be set before import.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
import skops.io
from mlflow.models import infer_signature

from src import config
from src.data_loader import load_raw_data
from src.evaluator import evaluate_pipeline, select_best_model
from src.preprocessing import (
    RANDOM_STATE,
    prepare_features_and_target,
    split_dataset,
)
from src.trainer import train_and_evaluate_candidates


def configure_mlflow(tracking_uri, experiment_name, artifact_dir):
    """
    Point MLflow at the (local) tracking store and experiment.

    The artifact location is set explicitly so model files land in
    this project's mlruns/ folder, regardless of the terminal's
    current working directory.
    """

    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            experiment_name,
            artifact_location=Path(artifact_dir).resolve().as_uri(),
        )

    mlflow.set_experiment(experiment_name)


def prefixed(metrics, prefix):
    """Add a prefix such as 'val_' or 'test_' to metric names."""
    return {f"{prefix}{name}": value for name, value in metrics.items()}


def run_training(
    data_path=None,
    model_dir=config.MODELS_DIR,
    reports_dir=config.REPORTS_DIR,
    tracking_uri=config.DEFAULT_TRACKING_URI,
    experiment_name=config.EXPERIMENT_NAME,
    artifact_dir=config.MLFLOW_ARTIFACT_DIR,
):
    """
    Run the full training workflow and return a summary dict.

    Parameters are injectable so tests can run against temporary
    folders and a temporary MLflow database.
    """

    model_dir = Path(model_dir)
    reports_dir = Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1-3. Load, validate, split ------------------------------
    raw_df = load_raw_data(data_path)
    X, y = prepare_features_and_target(raw_df)

    # Split BEFORE any fitting so validation/test rows never
    # influence imputation medians, scaling or encoding.
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    configure_mlflow(tracking_uri, experiment_name, artifact_dir)

    with mlflow.start_run(run_name="model-comparison") as parent_run:

        # Record how the data was split so the run is reproducible.
        mlflow.log_params({
            "random_state": RANDOM_STATE,
            "split": "70/15/15 stratified",
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features_raw": X.shape[1],
            "selection_metric": "val_roc_auc (tie-breaker: val_f1)",
        })
        mlflow.log_metrics({
            "train_churn_rate": float(y_train.mean()),
            "val_churn_rate": float(y_val.mean()),
            "test_churn_rate": float(y_test.mean()),
        })

        # ---- 4. Train candidates on TRAIN, score on VALIDATION ---
        results = train_and_evaluate_candidates(
            X_train, y_train, X_val, y_val
        )

        # One nested MLflow run per candidate keeps the comparison
        # easy to browse in the MLflow UI.
        for name, result in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                estimator = result["pipeline"].named_steps["model"]
                mlflow.set_tag("model_name", name)
                mlflow.log_params(estimator.get_params())
                mlflow.log_metrics(prefixed(result["metrics"], "val_"))
                mlflow.log_dict(
                    {"labels": [0, 1],
                     "matrix": result["confusion_matrix"]},
                    "val_confusion_matrix.json",
                )

        # ---- 5. Select on validation metrics only ----------------
        validation_metrics = {
            name: result["metrics"] for name, result in results.items()
        }
        best_name = select_best_model(validation_metrics)
        best_pipeline = results[best_name]["pipeline"]

        # ---- 6. Single, final evaluation on the TEST set ---------
        # This happens after selection, so the test score is an
        # unbiased estimate of performance on unseen customers.
        test_metrics, test_matrix = evaluate_pipeline(
            best_pipeline, X_test, y_test
        )

        # ---- 7. Logging and saving -------------------------------
        comparison = pd.DataFrame(validation_metrics).T
        comparison.index.name = "model"
        comparison = comparison.add_prefix("val_").round(4)
        comparison["selected"] = comparison.index == best_name
        comparison = comparison.sort_values(
            ["val_roc_auc", "val_f1"], ascending=False
        )

        comparison_path = reports_dir / "model_comparison.csv"
        comparison.to_csv(comparison_path)

        mlflow.set_tag("selected_model", best_name)
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics(prefixed(results[best_name]["metrics"], "val_"))
        mlflow.log_metrics(prefixed(test_metrics, "test_"))
        mlflow.log_dict(
            {"labels": [0, 1], "matrix": test_matrix},
            "test_confusion_matrix.json",
        )
        mlflow.log_artifact(str(comparison_path))

        # Log the complete preprocessing+model pipeline.
        # NOT registered here - registration is a separate step.
        #
        # MLflow saves sklearn models with skops, which refuses to
        # load any type not on its safe list. We trust exactly the
        # types in the pipeline we just trained ourselves (e.g. our
        # coerce_to_numeric function and sklearn's tree structures)
        # and nothing else.
        trusted_types = skops.io.get_untrusted_types(
            data=skops.io.dumps(best_pipeline)
        )
        input_example = X_train.head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            signature=infer_signature(
                input_example,
                best_pipeline.predict(input_example),
            ),
            input_example=input_example,
            # Ship our src/ package so the pickled pipeline (which
            # references src.preprocessing) can be loaded anywhere.
            code_paths=[str(config.PROJECT_ROOT / "src")],
            skops_trusted_types=trusted_types,
        )

        # Save a local copy for simple loading (tests, API, Docker).
        model_path = model_dir / config.MODEL_FILENAME
        joblib.dump(best_pipeline, model_path)

        summary = {
            "selected_model": best_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "max val_roc_auc, tie-breaker val_f1",
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_confusion_matrix": test_matrix,
            "feature_columns": X.columns.tolist(),
            "split_sizes": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
            "mlflow_run_id": parent_run.info.run_id,
            "mlflow_model_uri": model_info.model_uri,
            "mlflow_experiment": experiment_name,
            "model_path": str(model_path),
        }

        metadata_path = model_dir / config.METADATA_FILENAME
        metadata_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(metadata_path))

    return summary, comparison


def print_report(summary, comparison):
    """Print a readable summary to the console."""

    print("\n=== Validation comparison (selection data) ===")
    print(comparison.to_string())

    print(f"\nSelected model: {summary['selected_model']}")

    print("\n=== Test metrics (selected model only) ===")
    for name, value in summary["test_metrics"].items():
        print(f"  {name:<10} {value:.4f}")

    (tn, fp), (fn, tp) = summary["test_confusion_matrix"]
    print("\nTest confusion matrix:")
    print("               pred_no  pred_yes")
    print(f"  actual_no   {tn:>8} {fp:>9}")
    print(f"  actual_yes  {fn:>8} {tp:>9}")

    print(f"\nSaved pipeline : {summary['model_path']}")
    print(f"MLflow run id  : {summary['mlflow_run_id']}")
    print(f"MLflow model   : {summary['mlflow_model_uri']}")
    print("\nModel NOT registered. To register it (champion is "
          "unchanged):\n  python -m src.model_registry register")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare customer churn models."
    )
    parser.add_argument(
        "--data-path", default=None,
        help="CSV path (default: data/raw/Telco-Customer-Churn.csv)",
    )
    parser.add_argument(
        "--tracking-uri", default=config.DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (default: local sqlite mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name", default=config.EXPERIMENT_NAME,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary, comparison = run_training(
        data_path=args.data_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print_report(summary, comparison)
