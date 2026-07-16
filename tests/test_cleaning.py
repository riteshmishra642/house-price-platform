"""Unit tests for src/preprocessing/cleaning.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import DataCleaner


def test_cleaner_removes_all_missing_values(raw_dataset):
    cleaner = DataCleaner(target_column="SalePrice")
    cleaned = cleaner.fit_transform(raw_dataset)
    assert cleaned.isna().sum().sum() == 0


def test_cleaner_preserves_row_count_when_no_duplicates(raw_dataset):
    cleaner = DataCleaner(target_column="SalePrice")
    deduped = raw_dataset.drop_duplicates(subset=[c for c in raw_dataset.columns if c != "Id"])
    cleaned = cleaner.fit_transform(raw_dataset)
    assert len(cleaned) == len(deduped)


def test_domain_specific_missingness_becomes_none_not_mode():
    """A missing BsmtQual means 'no basement', not 'unknown' -- imputing
    with the modal quality value (e.g. 'TA') would fabricate a basement
    that doesn't exist. This must resolve to the literal string 'None'."""
    df = pd.DataFrame(
        {
            "SalePrice": [200000, 210000, 190000],
            "BsmtQual": ["Gd", "TA", np.nan],
            "GarageCars": [1, 2, 0],
            "GarageArea": [300.0, 400.0, 0.0],
        }
    )
    cleaner = DataCleaner(target_column="SalePrice")
    cleaned = cleaner.fit_transform(df)
    assert cleaned.loc[2, "BsmtQual"] == "None"


def test_outlier_capping_reduces_extreme_values():
    rng = np.random.default_rng(0)
    values = rng.normal(1000, 50, size=200).tolist() + [50000]  # one extreme outlier
    df = pd.DataFrame({"SalePrice": rng.normal(200000, 10000, size=201), "LotArea": values})
    cleaner = DataCleaner(target_column="SalePrice", outlier_iqr_multiplier=1.5)
    cleaned = cleaner.fit_transform(df, cap_outliers=True)
    assert cleaned["LotArea"].max() < 50000


def test_whitespace_is_stripped():
    df = pd.DataFrame({"SalePrice": [200000], "MSZoning": ["  RL  "]})
    cleaner = DataCleaner(target_column="SalePrice")
    cleaned = cleaner.fit_transform(df)
    assert cleaned.loc[0, "MSZoning"] == "RL"


def test_transform_before_fit_raises():
    df = pd.DataFrame({"SalePrice": [200000]})
    cleaner = DataCleaner(target_column="SalePrice")
    with pytest.raises(RuntimeError):
        cleaner.transform(df)


def test_new_category_at_inference_does_not_crash():
    """A category never seen during fit() should not crash transform() --
    it should fall through the safety-net imputation instead."""
    train_df = pd.DataFrame({"SalePrice": [200000, 210000], "MSZoning": ["RL", "RM"]})
    cleaner = DataCleaner(target_column="SalePrice").fit(train_df)

    new_df = pd.DataFrame({"SalePrice": [np.nan], "MSZoning": [np.nan]})
    cleaned = cleaner.transform(new_df, cap_outliers=False)
    assert cleaned["MSZoning"].isna().sum() == 0
