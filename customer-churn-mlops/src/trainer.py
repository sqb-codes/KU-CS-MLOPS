"""
trainer.py

Responsibility:
    Define candidate models, wrap each one with preprocessing in a
    single scikit-learn Pipeline, train on the training split and
    score on the validation split.

Why a full Pipeline (preprocessor + model)?
    1. No leakage: Pipeline.fit() fits the imputers, scaler and
       encoder on the data passed to fit() - the training split only.
    2. Training/serving consistency: the saved artifact contains the
       exact preprocessing used in training, so inference code only
       sends raw customer records.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluator import evaluate_pipeline
from src.preprocessing import RANDOM_STATE, build_preprocessor


def get_candidate_models(random_state=RANDOM_STATE):
    """
    Return the models to compare, with fixed hyperparameters.

    Hyperparameters are deliberately fixed (no tuning) in this
    foundation step. A fixed random_state makes results reproducible.
    """

    return {
        # Linear baseline: fast, interpretable, works well on
        # scaled + one-hot encoded data.
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        # Bagged trees: captures non-linear interactions.
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        # Boosted trees: often strongest on tabular data.
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def build_model_pipeline(X_train, estimator):
    """
    Combine a fresh (unfitted) preprocessor with an estimator.

    A NEW preprocessor is created per model so that fitted state
    is never shared between candidates.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ]
    )


def train_and_evaluate_candidates(X_train, y_train, X_val, y_val):
    """
    Train every candidate on TRAIN data and evaluate on VALIDATION.

    The test split is intentionally NOT an argument here: model
    comparison must never see test data.

    Returns:
        dict of model name -> {
            "pipeline": fitted Pipeline,
            "metrics": validation metrics,
            "confusion_matrix": validation confusion matrix
        }
    """

    results = {}

    for name, estimator in get_candidate_models().items():

        pipeline = build_model_pipeline(X_train, estimator)

        # Preprocessing + model are fitted together on training data.
        pipeline.fit(X_train, y_train)

        metrics, matrix = evaluate_pipeline(pipeline, X_val, y_val)

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "confusion_matrix": matrix,
        }

    return results
