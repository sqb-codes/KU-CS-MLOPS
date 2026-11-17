
"""
test_preprocessing.py

Validate the preprocessing workflow.

We check:
    - Input and target separation.
    - Correct target encoding.
    - Train/validation/test splitting.
    - Missing value handling.
    - Transformation output.
"""

from src.data_loader import load_raw_data

from src.preprocessing import (
    prepare_features_and_target,
    preprocess_dataset
)


def test_feature_target_separation():

    df = load_raw_data()

    X, y = prepare_features_and_target(df)

    # Target must not be present in input features.
    assert "Churn" not in X.columns

    # Customer ID must not be used as a model feature.
    assert "customerID" not in X.columns

    # Target must contain only 0 and 1.
    assert set(y.unique()) == {0, 1}

    assert len(X) == len(y)


def test_preprocessing_pipeline():

    df = load_raw_data()

    result = preprocess_dataset(df)

    # Verify that all three feature datasets exist.
    assert result["X_train"] is not None
    assert result["X_val"] is not None
    assert result["X_test"] is not None

    # Verify that the split sizes add up to the original
    # number of usable records.
    total_records = (
        result["X_train"].shape[0]
        + result["X_val"].shape[0]
        + result["X_test"].shape[0]
    )

    assert total_records == len(df)

    # Verify that preprocessing produces numeric matrices.
    assert result["X_train"].shape[1] > 0

    # Check that there are no NaN values after transformation.
    assert not __import__("numpy").isnan(
        result["X_train"].toarray()
        if hasattr(result["X_train"], "toarray")
        else result["X_train"]
    ).any()


def test_split_is_70_15_15_stratified_and_disjoint():

    from src.preprocessing import split_dataset

    df = load_raw_data()
    X, y = prepare_features_and_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    n = len(X)
    assert abs(len(X_train) / n - 0.70) < 0.01
    assert abs(len(X_val) / n - 0.15) < 0.01
    assert abs(len(X_test) / n - 0.15) < 0.01

    # Stratification: churn rate is (almost) identical in each split.
    for part in (y_train, y_val, y_test):
        assert abs(part.mean() - y.mean()) < 0.01

    # No customer appears in more than one split.
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)


def test_split_is_reproducible():

    from src.preprocessing import split_dataset

    X, y = prepare_features_and_target(load_raw_data())

    first = split_dataset(X, y)
    second = split_dataset(X, y)

    assert first[0].index.equals(second[0].index)
    assert first[2].index.equals(second[2].index)


def test_blank_total_charges_become_nan():

    # The raw IBM file stores TotalCharges as text with blanks
    # for brand-new customers.
    X, _ = prepare_features_and_target(load_raw_data())

    assert X["TotalCharges"].dtype.kind == "f"
    assert X["TotalCharges"].isna().sum() == 11


def test_preprocessor_fitted_on_training_data_only():

    df = load_raw_data()
    result = preprocess_dataset(df)

    X, y = prepare_features_and_target(df)
    from src.preprocessing import split_dataset
    X_train = split_dataset(X, y)[0]

    scaler = (
        result["preprocessor"]
        .named_transformers_["numerical"]
        .named_steps["scaler"]
    )
    tenure_position = list(
        result["preprocessor"].transformers_[0][2]
    ).index("tenure")

    # The scaler's mean must equal the TRAINING mean (no leakage
    # from validation/test rows).
    assert abs(
        scaler.mean_[tenure_position] - X_train["tenure"].mean()
    ) < 1e-9
