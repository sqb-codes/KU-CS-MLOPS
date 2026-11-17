"""
predict.py

Responsibility:
    Load the saved preprocessing-plus-model pipeline and score
    raw customer records.

Because preprocessing is packaged inside the pipeline, callers
send raw fields exactly as they appear in the source data
(e.g. "Contract": "Month-to-month"). No manual encoding needed.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import ID_COLUMN, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


def load_pipeline(model_path=None):
    """Load the fitted pipeline saved by train.py."""

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run 'python train.py' first."
        )

    # Only load model files you produced yourself: joblib/pickle
    # can execute code when loading untrusted files.
    return joblib.load(path)


def predict_churn(pipeline, records, threshold=0.5):
    """
    Predict churn for one or more raw customer records.

    Args:
        pipeline:  Fitted pipeline from load_pipeline().
        records:   dict, list of dicts, or DataFrame.
        threshold: Probability above which a customer is
                   flagged as churn.

    Returns:
        DataFrame with churn_probability and churn_prediction.
    """

    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame(records)

    # The identifier and label must never influence a prediction.
    df = df.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")

    # Fail clearly instead of with a cryptic sklearn error.
    expected = list(pipeline.feature_names_in_)
    missing = set(expected) - set(df.columns)

    if missing:
        raise ValueError(
            f"Input is missing required features: {sorted(missing)}"
        )

    # Keep the column order seen during training.
    df = df[expected]

    probabilities = pipeline.predict_proba(df)[:, 1]

    return pd.DataFrame({
        "churn_probability": probabilities,
        "churn_prediction": (probabilities >= threshold).astype(int),
    })
