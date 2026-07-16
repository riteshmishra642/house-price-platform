"""Unit tests for src/preprocessing/pipeline.py"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import build_full_preprocessor, split_feature_types


def test_split_feature_types_excludes_id_and_target():
    df = pd.DataFrame({"Id": [1, 2], "SalePrice": [100, 200], "LotArea": [1000, 2000], "MSZoning": ["RL", "RM"]})
    numeric_cols, categorical_cols = split_feature_types(df)
    assert "Id" not in numeric_cols and "SalePrice" not in numeric_cols
    assert "LotArea" in numeric_cols
    assert "MSZoning" in categorical_cols


def test_preprocessor_produces_no_missing_values(engineered_dataset):
    X = engineered_dataset.drop(columns=["SalePrice", "Id"])
    preprocessor = build_full_preprocessor(engineered_dataset)
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed.to_numpy()).any()


def test_preprocessor_handles_unseen_category_gracefully(engineered_dataset):
    X = engineered_dataset.drop(columns=["SalePrice", "Id"])
    preprocessor = build_full_preprocessor(engineered_dataset)
    preprocessor.fit(X)

    X_new = X.iloc[[0]].copy()
    if "Neighborhood" in X_new.columns:
        X_new["Neighborhood"] = "NeverSeenBefore"
    # Should not raise, thanks to handle_unknown="ignore".
    transformed = preprocessor.transform(X_new)
    assert transformed.shape[0] == 1
