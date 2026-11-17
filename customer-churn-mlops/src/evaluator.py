"""
evaluator.py

Responsibility:
    Measure model quality and choose the best candidate.

Why several metrics?
    Churn is imbalanced (~26% of customers churn). Accuracy alone
    is misleading: a model that predicts "No churn" for everyone
    scores ~74% accuracy while catching zero churners. Precision,
    recall, F1 and ROC-AUC describe the minority class much better.

Why ROC-AUC for selection?
    ROC-AUC measures how well the model RANKS customers by churn
    risk, independent of the 0.5 decision threshold. The business
    can later choose a threshold; ranking quality is what we compare.
    F1 is used only as a tie-breaker.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Metric used to pick the winning model, and its tie-breaker.
PRIMARY_METRIC = "roc_auc"
TIE_BREAKER_METRIC = "f1"


def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute classification metrics for the positive class (churn = 1).

    Args:
        y_true:  True labels (0/1).
        y_pred:  Predicted labels (0/1).
        y_proba: Predicted probability of churn (class 1).

    Returns:
        dict of metric name -> float.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0 avoids crashes/warnings when a weak model
        # predicts no positives at all.
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC-AUC needs probabilities, not hard labels.
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_pipeline(pipeline, X, y):
    """
    Evaluate a FITTED pipeline on a dataset.

    The pipeline receives RAW features: preprocessing is applied
    internally using parameters learned from the training set only.

    Returns:
        metrics: dict of metric values.
        matrix:  2x2 confusion matrix as nested lists
                 [[TN, FP], [FN, TP]] (JSON-serialisable).
    """

    y_pred = pipeline.predict(X)

    # Column 1 of predict_proba is the probability of churn.
    y_proba = pipeline.predict_proba(X)[:, 1]

    metrics = compute_metrics(y, y_pred, y_proba)

    # Fix label order so the matrix layout is always the same.
    matrix = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()

    return metrics, matrix


def select_best_model(validation_results):
    """
    Choose the best model using VALIDATION metrics only.

    Args:
        validation_results:
            dict of model name -> metrics dict
            (each must contain 'roc_auc' and 'f1').

    Returns:
        Name of the best model.

    Rule:
        Highest validation ROC-AUC wins; if two models tie on
        ROC-AUC, the higher validation F1-score wins.

    Important:
        Never pass test-set metrics here. The test set must stay
        untouched until AFTER selection, otherwise the reported
        test score is optimistically biased.
    """

    if not validation_results:
        raise ValueError("No model results supplied for selection.")

    # Sorting by a tuple compares ROC-AUC first, then F1.
    return max(
        validation_results,
        key=lambda name: (
            validation_results[name][PRIMARY_METRIC],
            validation_results[name][TIE_BREAKER_METRIC],
        ),
    )
