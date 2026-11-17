
"""
test_data_loader.py

Tests for the raw customer data loader.
"""

import pytest

from src.data_loader import load_raw_data


def test_load_raw_data():

    # Load the actual IBM dataset.
    df = load_raw_data()

    # Verify that records were loaded.
    assert not df.empty

    # Verify the original dataset dimensions.
    assert df.shape == (7043, 21)

    # Verify essential columns exist.
    assert "customerID" in df.columns
    assert "tenure" in df.columns
    assert "MonthlyCharges" in df.columns
    assert "Churn" in df.columns


def test_missing_file_raises_error():

    # An invalid file path should raise a clear exception.
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/non_existing.csv")


def test_missing_required_columns_raises_error(tmp_path):

    # Create a temporary CSV containing incorrect columns.
    invalid_file = tmp_path / "invalid.csv"

    invalid_file.write_text("name,age\nJohn,25\n")

    # The loader must reject this dataset.
    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_data(invalid_file)
