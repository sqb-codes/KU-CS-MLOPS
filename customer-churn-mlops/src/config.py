"""
config.py

Central place for paths and MLflow settings shared by
train.py and model_registry.py.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_FILENAME = "churn_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"

# Local MLflow tracking only.
# We pass this URI explicitly to mlflow.set_tracking_uri(), so a
# MLFLOW_TRACKING_URI environment variable pointing at a shared or
# remote server is NOT picked up by accident. Use --tracking-uri to
# opt in to a different server deliberately.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "customer-churn-training"

# Model registry settings (used only by model_registry.py).
REGISTERED_MODEL_NAME = "customer-churn-classifier"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
