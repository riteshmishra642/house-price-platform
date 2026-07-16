"""
Preprocessing pipeline: encoding + scaling, built on scikit-learn's
ColumnTransformer so it can be persisted with joblib and reused identically
across training, batch prediction, and the live API — the single biggest
source of "worked in the notebook, broke in production" bugs is preprocessing
logic that isn't literally the same object in both places.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_AND_TARGET_COLUMNS = {"Id", "SalePrice"}


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric vs categorical, excluding Id/target."""
    usable = df.drop(columns=[c for c in ID_AND_TARGET_COLUMNS if c in df.columns])
    numeric_cols = usable.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = usable.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaling_method: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer applying scaling to numeric columns and
    one-hot encoding to categorical columns.

    handle_unknown="ignore" on the encoder is critical for production safety:
    a live API request may contain a categorical value (e.g. a new
    Neighborhood code) never seen during training. Without this, the pipeline
    would raise instead of gracefully encoding it as all-zeros.
    """
    numeric_transformer = StandardScaler() if scaling_method == "standard" else "passthrough"

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_full_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Convenience: infer column types from a DataFrame and build the transformer."""
    config = load_config()
    numeric_cols, categorical_cols = split_feature_types(df)
    logger.info(
        "Building preprocessor: %d numeric columns, %d categorical columns.",
        len(numeric_cols), len(categorical_cols),
    )
    return build_preprocessing_pipeline(
        numeric_cols, categorical_cols, scaling_method=config.preprocessing.scaling_method
    )


if __name__ == "__main__":
    from src.feature_engineering.features import engineer_features
    from src.pipelines.data_ingestion import load_or_create_raw_dataset
    from src.preprocessing.cleaning import clean_dataset

    raw_df = load_or_create_raw_dataset()
    cleaned_df, _ = clean_dataset(raw_df)
    engineered_df = engineer_features(cleaned_df)

    X = engineered_df.drop(columns=["SalePrice", "Id"])
    y = engineered_df["SalePrice"]

    preprocessor = build_full_preprocessor(engineered_df)
    X_transformed = preprocessor.fit_transform(X)
    print("Transformed shape:", X_transformed.shape)
    print(X_transformed.head())
