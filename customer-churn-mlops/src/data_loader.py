
"""
data_loader.py

Responsibility:
    Load raw customer churn data from the configured source.

Current source:
    Local CSV file downloaded from IBM's public dataset.

Future sources:
    AWS S3, PostgreSQL/RDS or other data platforms.

Design principle:
    Other modules should not need to know where the
    training dataset comes from.
"""

from pathlib import Path

import pandas as pd


# Resolve the project root independently of the current
# terminal working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the original, unmodified dataset.
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
)


def load_raw_data(file_path=None):
    """
    Load the raw customer churn dataset.

    Args:
        file_path:
            Optional custom CSV path.
            If omitted, use the default IBM dataset location.

    Returns:
        df:
            Pandas DataFrame containing the raw customer data.
    """

    # Use the default location unless a custom path is supplied.
    data_path = Path(file_path) if file_path else DEFAULT_DATA_PATH

    # Fail early with a clear message if the dataset is missing.
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Please download the IBM Customer Churn CSV first."
        )

    # Read the CSV file into memory.
    df = pd.read_csv(data_path)

    # Check whether the file contains records.
    if df.empty:
        raise ValueError("The input dataset is empty.")

    # These columns are essential for our churn project.
    required_columns = {
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    # Find required columns missing from the input file.
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Return the original DataFrame.
    # Do not clean or transform data here.
    # That responsibility belongs to preprocessing.py.
    return df


if __name__ == "__main__":

    # Execute the loader independently for testing.
    customer_df = load_raw_data()

    print("Dataset loaded successfully!")
    print("Shape:", customer_df.shape)

    print("\nColumns:")
    print(customer_df.columns.tolist())

    print("\nFirst five records:")
    print(customer_df.head())
