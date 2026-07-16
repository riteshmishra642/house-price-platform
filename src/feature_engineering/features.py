"""
Feature engineering module.

Creates domain-driven derived features that consistently improve tree-based
and linear model performance on housing data, beyond the raw columns.
Every feature is documented with WHY it helps, not just what it computes.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0}


class FeatureEngineer:
    """
    Stateless (no fit needed) feature engineering transformer.

    Kept fit/transform-shaped anyway so it composes into an sklearn Pipeline
    alongside stateful steps (cleaning, scaling, encoding).
    """

    def __init__(self, current_year: int | None = None):
        self.current_year = current_year

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        reference_year = self.current_year or int(df["YrSold"].max()) if "YrSold" in df.columns else 2010

        df = self._add_age_features(df, reference_year)
        df = self._add_area_ratio_features(df)
        df = self._add_bathroom_room_features(df)
        df = self._add_quality_index_features(df)
        df = self._add_luxury_and_size_category_features(df)
        df = self._add_price_context_features(df)

        logger.info("Feature engineering complete: %d columns.", df.shape[1])
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_age_features(df: pd.DataFrame, reference_year: int) -> pd.DataFrame:
        """
        HouseAge and RemodAge capture depreciation and renovation recency —
        two of the strongest real-world price drivers that raw YearBuilt /
        YearRemodAdd only express indirectly (a linear model can't learn
        "age" from an absolute year the way it can from a relative age).
        """
        df = df.copy()
        sale_year = df["YrSold"] if "YrSold" in df.columns else reference_year
        if "YearBuilt" in df.columns:
            df["HouseAge"] = (sale_year - df["YearBuilt"]).clip(lower=0)
        if "YearRemodAdd" in df.columns:
            df["RemodAge"] = (sale_year - df["YearRemodAdd"]).clip(lower=0)
        if "GarageYrBlt" in df.columns:
            garage_age = sale_year - df["GarageYrBlt"]
            df["GarageAge"] = garage_age.clip(lower=0).fillna(0)
        if "HouseAge" in df.columns:
            df["IsRemodeled"] = (df.get("RemodAge", df["HouseAge"]) < df["HouseAge"]).astype(int)
        if "HouseAge" in df.columns:
            df["IsNewHouse"] = (df["HouseAge"] <= 5).astype(int)
        return df

    @staticmethod
    def _add_area_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ratios normalize raw square footage against lot/garage/basement size,
        which helps models separate "big house on a big lot" (typical) from
        "big house crammed onto a small lot" (atypical, price-relevant).
        """
        df = df.copy()
        if {"GarageArea", "LotArea"}.issubset(df.columns):
            df["GarageRatio"] = (df["GarageArea"] / df["LotArea"].replace(0, np.nan)).fillna(0)
        if {"TotalBsmtSF", "1stFlrSF"}.issubset(df.columns):
            df["BasementRatio"] = (df["TotalBsmtSF"] / df["1stFlrSF"].replace(0, np.nan)).fillna(0).clip(upper=3)
        if {"GrLivArea", "LotArea"}.issubset(df.columns):
            df["LivingAreaRatio"] = (df["GrLivArea"] / df["LotArea"].replace(0, np.nan)).fillna(0)
        if {"1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
            df["TotalPorchSF"] = df.get("OpenPorchSF", 0) + df.get("WoodDeckSF", 0)
        if {"1stFlrSF", "2ndFlrSF", "TotalBsmtSF"}.issubset(df.columns):
            df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
        return df

    @staticmethod
    def _add_bathroom_room_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Buyers reason in "total bathrooms", not separate full/half/basement
        counts. Half baths count as 0.5 full-bath-equivalents, matching how
        real estate listings advertise bathroom counts.
        """
        df = df.copy()
        full = df.get("FullBath", 0)
        half = df.get("HalfBath", 0)
        bsmt_full = df.get("BsmtFullBath", 0)
        bsmt_half = df.get("BsmtHalfBath", 0)
        df["TotalBathrooms"] = full + 0.5 * half + bsmt_full + 0.5 * bsmt_half
        if {"TotRmsAbvGrd", "BedroomAbvGr"}.issubset(df.columns):
            df["TotalRooms"] = df["TotRmsAbvGrd"] + df["TotalBathrooms"]
            df["RoomsPerBedroom"] = (df["TotRmsAbvGrd"] / df["BedroomAbvGr"].replace(0, np.nan)).fillna(
                df["TotRmsAbvGrd"]
            )
        return df

    @staticmethod
    def _add_quality_index_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts ordinal quality strings (Po/Fa/TA/Gd/Ex) into a single
        numeric Construction Quality Index. Tree models can split on strings
        via encoding, but a composite numeric index captures *combined*
        quality (exterior + kitchen + basement + heating together) in one
        feature, which is easier for linear models to use meaningfully and
        reduces dimensionality versus one-hot encoding every quality column.
        """
        df = df.copy()
        quality_col_names = ["ExterQual", "KitchenQual", "HeatingQC", "BsmtQual", "GarageQual"]
        quality_cols = [c for c in quality_col_names if c in df.columns]
        if quality_cols:
            scores = pd.DataFrame({c: df[c].map(QUALITY_MAP).fillna(0) for c in quality_cols})
            df["ConstructionQualityIndex"] = scores.mean(axis=1)
        if {"OverallQual", "OverallCond"}.issubset(df.columns):
            df["OverallQualCondScore"] = df["OverallQual"] * 0.7 + df["OverallCond"] * 0.3
        return df

    @staticmethod
    def _add_luxury_and_size_category_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        LuxuryScore aggregates discretionary amenities (pool, fireplace,
        high garage capacity, large porch/deck) that correlate with premium
        pricing beyond core size/quality — useful signal for high-end
        properties that basic size/quality features under-predict.
        PropertySizeCategory buckets GrLivArea into human-interpretable tiers
        useful for dashboard filtering/segmentation, not just modeling.
        """
        df = df.copy()
        luxury_components = []
        if "PoolArea" in df.columns:
            luxury_components.append((df["PoolArea"] > 0).astype(int) * 2)
        if "Fireplaces" in df.columns:
            luxury_components.append(df["Fireplaces"].clip(upper=2))
        if "GarageCars" in df.columns:
            luxury_components.append((df["GarageCars"] >= 3).astype(int) * 2)
        if "TotalPorchSF" in df.columns:
            luxury_components.append((df["TotalPorchSF"] > df["TotalPorchSF"].median()).astype(int))
        if luxury_components:
            df["LuxuryScore"] = sum(luxury_components)

        if "GrLivArea" in df.columns:
            df["PropertySizeCategory"] = pd.cut(
                df["GrLivArea"],
                bins=[-np.inf, 900, 1400, 2000, 2800, np.inf],
                labels=["Compact", "Small", "Medium", "Large", "Estate"],
            ).astype(str)
        return df

    @staticmethod
    def _add_price_context_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        NeighborhoodQualityScore encodes each neighborhood's *typical*
        OverallQual (not price directly, to avoid leaking the target into
        features) — this is a common, leakage-safe way to give models a
        location-quality signal without one-hot-encoding 20+ neighborhoods.
        """
        df = df.copy()
        if {"Neighborhood", "OverallQual"}.issubset(df.columns):
            neighborhood_quality = df.groupby("Neighborhood")["OverallQual"].transform("mean")
            df["NeighborhoodQualityScore"] = neighborhood_quality
        if {"GrLivArea", "TotalBsmtSF"}.issubset(df.columns):
            total_area = df["GrLivArea"] + df.get("TotalBsmtSF", 0)
            df["PropertyValueIndex"] = (df.get("OverallQual", 5) * total_area.replace(0, np.nan)).fillna(0) / 1000
        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function wrapping FeatureEngineer for scripts/notebooks."""
    return FeatureEngineer().fit_transform(df)


if __name__ == "__main__":
    from src.pipelines.data_ingestion import load_or_create_raw_dataset
    from src.preprocessing.cleaning import clean_dataset

    raw_df = load_or_create_raw_dataset()
    cleaned_df, _ = clean_dataset(raw_df)
    engineered_df = engineer_features(cleaned_df)
    new_cols = set(engineered_df.columns) - set(cleaned_df.columns)
    print(f"Added {len(new_cols)} new features:")
    for col in sorted(new_cols):
        print(" -", col)
    print("Final shape:", engineered_df.shape)
