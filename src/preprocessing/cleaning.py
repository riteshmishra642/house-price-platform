"""
Data cleaning module.

Handles missing values, incorrect dtypes, duplicates, whitespace, and
outliers in a single, testable, reusable class (DataCleaner) so the exact
same logic runs in training, batch scoring, and the live API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns where a missing value has a real-world meaning of "does not have
# this feature" rather than "unknown" (per the Ames Housing data dictionary).
NONE_MEANS_ABSENT_COLUMNS = [
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "FireplaceQu",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "Alley",
    "MasVnrType",
]

# Numeric columns where missing plausibly means zero (e.g. no garage -> 0 cars).
ZERO_MEANS_ABSENT_COLUMNS = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]


@dataclass
class CleaningReport:
    """Structured summary of every cleaning action taken, for transparency."""

    initial_shape: tuple
    final_shape: tuple = field(default=None)
    duplicates_removed: int = 0
    missing_values_before: Dict[str, int] = field(default_factory=dict)
    missing_values_after: Dict[str, int] = field(default_factory=dict)
    outliers_capped: Dict[str, int] = field(default_factory=dict)
    dtype_corrections: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "initial_shape": self.initial_shape,
            "final_shape": self.final_shape,
            "duplicates_removed": self.duplicates_removed,
            "missing_values_before": self.missing_values_before,
            "missing_values_after": self.missing_values_after,
            "outliers_capped": self.outliers_capped,
            "dtype_corrections": self.dtype_corrections,
        }


class DataCleaner:
    """
    Encapsulates every data cleaning step.

    Why a class instead of standalone functions: cleaning must apply
    identically to training data, held-out test data, and a single incoming
    API prediction request. A stateful, fit/transform-style object (mirroring
    scikit-learn's API) lets us reuse learned statistics (like median values)
    consistently instead of recomputing them differently in each context.
    """

    def __init__(self, target_column: str = "SalePrice", outlier_iqr_multiplier: float = 3.0):
        self.target_column = target_column
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self._numeric_medians: Dict[str, float] = {}
        self._categorical_modes: Dict[str, str] = {}
        self._is_fitted = False
        self.report: CleaningReport | None = None

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn imputation statistics (medians/modes) from training data only."""
        working = df.copy()
        working = self._standardize_whitespace(working)

        numeric_cols = working.select_dtypes(include=[np.number]).columns
        categorical_cols = working.select_dtypes(exclude=[np.number]).columns

        for col in numeric_cols:
            if col == self.target_column:
                continue
            self._numeric_medians[col] = working[col].median()

        for col in categorical_cols:
            mode_series = working[col].mode(dropna=True)
            self._categorical_modes[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

        self._is_fitted = True
        logger.info(
            "DataCleaner fitted on %d rows: learned %d numeric medians, %d categorical modes.",
            len(working),
            len(self._numeric_medians),
            len(self._categorical_modes),
        )
        return self

    def transform(self, df: pd.DataFrame, cap_outliers: bool = True) -> pd.DataFrame:
        """Apply cleaning steps using statistics learned during fit()."""
        if not self._is_fitted:
            raise RuntimeError("DataCleaner.transform() called before fit(). Call fit() first.")

        initial_shape = df.shape
        missing_before = df.isna().sum()
        missing_before = missing_before[missing_before > 0].to_dict()

        working = df.copy()
        working = self._standardize_whitespace(working)
        working = self._drop_duplicates(working)
        working = self._fix_dtypes(working)
        working = self._impute_domain_specific_missing(working)
        working = self._impute_remaining_missing(working)

        outlier_counts: Dict[str, int] = {}
        if cap_outliers:
            working, outlier_counts = self._cap_outliers(working)

        missing_after = working.isna().sum()
        missing_after = missing_after[missing_after > 0].to_dict()

        self.report = CleaningReport(
            initial_shape=initial_shape,
            final_shape=working.shape,
            duplicates_removed=initial_shape[0] - len(working) if "duplicated" in working.attrs else 0,
            missing_values_before=missing_before,
            missing_values_after=missing_after,
            outliers_capped=outlier_counts,
        )
        logger.info("Cleaning complete: %s -> %s", initial_shape, working.shape)
        return working

    def fit_transform(self, df: pd.DataFrame, cap_outliers: bool = True) -> pd.DataFrame:
        return self.fit(df).transform(df, cap_outliers=cap_outliers)

    # ------------------------------------------------------------------ #
    # Individual cleaning steps (each isolated + testable)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _standardize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from string columns and column names."""
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        subset = [c for c in df.columns if c != "Id"]
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate rows.", removed)
        return df

    @staticmethod
    def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns that are semantically categorical but stored as numbers."""
        df = df.copy()
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)
        for year_col in ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]:
            if year_col in df.columns:
                df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        return df

    @staticmethod
    def _impute_domain_specific_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Ames data-dictionary semantics: for many columns, NaN doesn't
        mean "unknown" — it means "this house doesn't have this feature".
        Imputing these with the mode/median (as a naive pipeline would) would
        fabricate features that don't exist. This step must run before
        generic imputation.
        """
        df = df.copy()
        for col in NONE_MEANS_ABSENT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in ZERO_MEANS_ABSENT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def _impute_remaining_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute any remaining missing values with statistics learned in fit()."""
        df = df.copy()
        for col, median_value in self._numeric_medians.items():
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(median_value)
        for col, mode_value in self._categorical_modes.items():
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(mode_value)

        # Catch-all safety net for any column not seen during fit (e.g. a
        # brand-new column at inference time) so the API never crashes on NaN.
        remaining_numeric = df.select_dtypes(include=[np.number]).columns
        for col in remaining_numeric:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        remaining_categorical = df.select_dtypes(exclude=[np.number]).columns
        for col in remaining_categorical:
            if df[col].isna().any():
                df[col] = df[col].fillna("Unknown")
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
        """
        Cap (winsorize) extreme numeric outliers using the IQR method rather
        than deleting rows — deletion loses information and can bias a
        housing model against legitimately large/expensive properties.
        """
        df = df.copy()
        outlier_counts: Dict[str, int] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = {"Id", self.target_column, "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold"}

        for col in numeric_cols:
            if col in exclude:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - self.outlier_iqr_multiplier * iqr
            upper = q3 + self.outlier_iqr_multiplier * iqr
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_outliers:
                outlier_counts[col] = n_outliers
                df[col] = df[col].clip(lower=lower, upper=upper)

        if outlier_counts:
            logger.info("Capped outliers via IQR method in %d columns: %s", len(outlier_counts), outlier_counts)
        return df, outlier_counts


def clean_dataset(df: pd.DataFrame, target_column: str = "SalePrice") -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience function: fit and transform in one call, returning the report too."""
    config = load_config()
    cleaner = DataCleaner(
        target_column=target_column,
        outlier_iqr_multiplier=config.preprocessing.outlier_iqr_multiplier,
    )
    cleaned = cleaner.fit_transform(df)
    return cleaned, cleaner.report


if __name__ == "__main__":
    from src.pipelines.data_ingestion import load_or_create_raw_dataset

    raw_df = load_or_create_raw_dataset()
    cleaned_df, report = clean_dataset(raw_df)
    print("Initial shape:", report.initial_shape)
    print("Final shape:", report.final_shape)
    print("Remaining missing values:", report.missing_values_after)
