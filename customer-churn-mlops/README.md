# Customer Churn MLOps — Local Training Foundation

This project predicts which telecom customers are likely to churn, using the
**IBM Telco Customer Churn** dataset. This stage covers reproducible local
training, model comparison, MLflow experiment tracking, and a separate,
manual model-promotion step.

## Project layout

```
customer-churn-mlops/
├── data/raw/Telco-Customer-Churn.csv   # input data (git-ignored, not committed)
├── src/
│   ├── config.py           # paths + local MLflow settings
│   ├── data_loader.py      # load CSV, check required columns
│   ├── preprocessing.py    # validation, X/y split, 70/15/15 split, preprocessor
│   ├── trainer.py          # 3 candidate Pipelines, train + validate
│   ├── evaluator.py        # metrics, confusion matrix, model selection
│   ├── predict.py          # load saved pipeline, score raw records
│   └── model_registry.py   # manual register / promote (separate from training)
├── tests/                  # pytest suite
├── train.py                # training entry point
├── models/                 # saved pipeline + metadata (generated, git-ignored)
├── reports/                # model_comparison.csv (generated, git-ignored)
├── mlflow.db, mlruns/      # local MLflow store (generated, git-ignored)
└── requirements.txt
```

## How training works

| Step | What happens | Why |
|------|--------------|-----|
| 1 | Load the CSV and check its columns, duplicate IDs and target labels | Bad input data fails early with a clear error |
| 2 | Drop `customerID`, convert `TotalCharges` to a number | An ID carries no predictive signal; blank charges become NaN and are filled in later |
| 3 | Stratified **70/15/15** train/validation/test split (`random_state=42`) | Reproducible, and the churn rate (~26.5%) is preserved in every split |
| 4 | `Pipeline(preprocessor → model)` for Logistic Regression, Random Forest and Gradient Boosting, fitted on **train only** | No leakage: imputation medians, scaling and one-hot categories are learned from training rows only |
| 5 | Compare models on **validation**; choose the highest ROC-AUC, using F1 as the tie-breaker | ROC-AUC measures how well each model ranks customers by churn risk, independent of the 0.5 threshold |
| 6 | Score the **test set once**, for the selected model only | Gives an unbiased estimate, because the test set never influenced any choice |
| 7 | Log to MLflow, save `models/churn_pipeline.joblib` and `model_metadata.json` | The saved file contains preprocessing and model together, so inference only needs raw fields |

Training **never registers a model and never changes the champion**.
Promotion is a separate, explicit step (see below).

MLflow tracking is **local by default** (`sqlite:///mlflow.db` in this folder).
The URI is set explicitly, so an `MLFLOW_TRACKING_URI` environment variable
pointing at a shared or remote server is ignored unless you pass
`--tracking-uri` yourself.

---

## Setup

Requires Python 3.10+.

### macOS (zsh/bash)

```bash
cd customer-churn-mlops

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd customer-churn-mlops

# Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, run this once for the current session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install dependencies
pip install -r requirements.txt
```

> If you already use the shared course environment (`../kiet-env`), activate
> it instead: `source ../kiet-env/bin/activate` (macOS) or
> `..\kiet-env\Scripts\Activate.ps1` (Windows).

Place the dataset at `data/raw/Telco-Customer-Churn.csv`.

---

## Run

The commands are the same on macOS and Windows PowerShell once the
environment is active. Run them from the `customer-churn-mlops` folder.

```bash
# 1. Run the tests (~10 s; uses temporary folders and a temporary MLflow DB)
python -m pytest -q

# 2. Train, compare and save the selected model
python train.py

# Optional: custom data path or experiment name
python train.py --data-path data/raw/Telco-Customer-Churn.csv --experiment-name churn-dev

# 3. Browse experiments in the MLflow UI, then open http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Expected output of `python train.py`

```
=== Validation comparison (selection data) ===
                     val_accuracy  val_precision  val_recall  val_f1  val_roc_auc  selected
gradient_boosting          0.8011         0.6562      0.5250  0.5833       0.8479      True
logistic_regression        0.8059         0.6459      0.5929  0.6182       0.8453     False
random_forest              0.7926         0.6368      0.5071  0.5646       0.8399     False

Selected model: gradient_boosting

=== Test metrics (selected model only) ===
  accuracy   0.7900
  precision  0.6439
  recall     0.4698
  f1         0.5432
  roc_auc    0.8343
...
Model NOT registered. To register it (champion is unchanged):
  python -m src.model_registry register
```

Exact numbers may vary slightly between scikit-learn versions.

Files produced:

- `models/churn_pipeline.joblib`: the full preprocessing-plus-model pipeline
- `models/model_metadata.json`: selected model, validation and test metrics, MLflow run ID and model URI
- `reports/model_comparison.csv`: validation comparison table
- `mlflow.db`, `mlruns/`: the local MLflow store, with one parent run plus a nested run per model

### Making predictions

```python
from src.predict import load_pipeline, predict_churn

pipeline = load_pipeline()  # models/churn_pipeline.joblib
customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85",
}
print(predict_churn(pipeline, customer))
```

Only load `.joblib` files that you produced yourself, because loading a
pickle can execute code.

---

## Model registration and promotion (manual, separate from training)

```bash
# Register the latest trained model as a new version, aliased "challenger"
python -m src.model_registry register

# Show the current champion and challenger
python -m src.model_registry status

# Make version 1 the champion (works only if there is no champion yet)
python -m src.model_registry promote --version 1

# Replace an existing champion: requires an explicit flag after review
python -m src.model_registry promote --version 2 --replace-existing
```

Without `--replace-existing`, `promote` refuses to overwrite an existing champion.

---

## What is intentionally not committed

`.gitignore` excludes raw data, saved models, reports, `mlflow.db`, `mlruns/`
and `.env` files. Recreate them locally by running `python train.py`.
