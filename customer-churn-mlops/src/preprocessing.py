
"""
preprocessing.py

Responsibility:
    Clean and transform customer data.

    Separate input features and target.
    Handle missing values.
    Encode categorical features.
    Scale numerical features.
    Split data into training, validation and testing sets.

Important:
    All preprocessing transformations are fitted ONLY
    on the training dataset.

    This prevents data leakage from validation and test data.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


# Target column we want to predict.
TARGET_COLUMN = "Churn"

# Identifier column.
# This is retained for reference but excluded from model features.
ID_COLUMN = "customerID"

# Reproducibility: the same random splits are generated
# whenever we run the pipeline with the same dataset.
RANDOM_STATE = 42


def prepare_features_and_target(df):
    """
    Validate the raw dataset and separate X and y.

    X = customer input features
    y = churn target labels
    """

    # Work on a copy so that the original raw dataset
    # remains unchanged.
    df = df.copy()

    # Verify required columns.
    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Remove duplicate customer identifiers.
    # In this dataset, each customer should appear only once.
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs found in the dataset."
        )

    # Remove rows where the target is missing.
    df = df.dropna(subset=[TARGET_COLUMN])

    # Validate the target labels before encoding.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    invalid_labels = set(df[TARGET_COLUMN].unique()) - {"Yes", "No"}

    if invalid_labels:
        raise ValueError(
            f"Unexpected churn labels: {invalid_labels}"
        )

    # Convert the TotalCharges column from text to numeric.
    # Invalid or blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Encode target labels.
    # No = 0, Yes = 1.
    y = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # Remove target and customer identifier from features.
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    # Return the raw features and target.
    # Imputation and encoding will happen later.
    return X, y


def coerce_to_numeric(X):
    """
    Convert every column to numeric, turning invalid values into NaN.

    Why this lives INSIDE the pipeline:
        prepare_features_and_target() already converts TotalCharges
        for training, but at inference time the saved pipeline may
        receive raw records (e.g. TotalCharges = " " for brand-new
        customers). Putting the conversion in the pipeline guarantees
        training and serving apply exactly the same logic.

    Defined at module level (not as a lambda) so the fitted pipeline
    can be pickled by joblib / MLflow.
    """
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")


def build_preprocessor(X):
    """
    Create a reusable preprocessing pipeline.

    Numerical features:
        Median imputation + standard scaling.

    Categorical features:
        Most frequent imputation + one-hot encoding.
    """

    # Automatically identify numerical columns.
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Automatically identify categorical columns.
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing pipeline.
    numerical_pipeline = Pipeline(
        steps=[
            (
                # Text -> number (blank TotalCharges -> NaN -> imputed).
                "to_numeric",
                FunctionTransformer(
                    coerce_to_numeric,
                    feature_names_out="one-to-one"
                )
            ),
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing pipeline.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Apply different transformations to different columns.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return preprocessor


def split_dataset(X, y):
    """
    Split data into training, validation and testing sets.

    70% = Training
    15% = Validation
    15% = Testing

    Stratification preserves the approximate ratio
    of churn and non-churn records in each dataset.
    """

    # First separate 70% training and 30% temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Divide the temporary 30% equally:
    # 15% validation and 15% testing.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def preprocess_dataset(df):
    """
    Execute the complete preprocessing workflow.

    Returns:
        preprocessor:
            Fitted preprocessing pipeline.

        Transformed datasets:
            Training, validation and test features.

        Labels:
            Training, validation and test target values.
    """

    # Step 1: Separate features and target.
    X, y = prepare_features_and_target(df)

    # Step 2: Split before fitting preprocessing.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    # Step 3: Build preprocessing transformations
    # using the training feature columns.
    preprocessor = build_preprocessor(X_train)

    # Step 4: Fit transformations only on training data.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Step 5: Transform validation and test data
    # without fitting again.
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_columns": X.columns.tolist()
    }
