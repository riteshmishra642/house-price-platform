"""Unit tests for src/feature_engineering/features.py"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.features import FeatureEngineer, engineer_features


def test_house_age_is_non_negative(engineered_dataset):
    assert (engineered_dataset["HouseAge"] >= 0).all()


def test_total_bathrooms_computed_correctly():
    df = pd.DataFrame(
        {
            "FullBath": [2], "HalfBath": [1], "BsmtFullBath": [1], "BsmtHalfBath": [0],
            "TotRmsAbvGrd": [8], "BedroomAbvGr": [3],
            "YrSold": [2008], "YearBuilt": [2000], "YearRemodAdd": [2000],
        }
    )
    result = FeatureEngineer().transform(df)
    # 2 full + 0.5*1 half + 1 basement full + 0.5*0 basement half = 3.5
    assert result.loc[0, "TotalBathrooms"] == 3.5


def test_construction_quality_index_in_expected_range(engineered_dataset):
    assert engineered_dataset["ConstructionQualityIndex"].between(0, 5).all()


def test_property_size_category_has_no_nulls(engineered_dataset):
    assert engineered_dataset["PropertySizeCategory"].isna().sum() == 0


def test_neighborhood_quality_score_matches_group_mean():
    df = pd.DataFrame(
        {
            "Neighborhood": ["A", "A", "B"],
            "OverallQual": [4, 6, 8],
            "YrSold": [2008, 2008, 2008],
            "YearBuilt": [2000, 2000, 2000],
            "YearRemodAdd": [2000, 2000, 2000],
        }
    )
    result = FeatureEngineer().transform(df)
    assert result.loc[0, "NeighborhoodQualityScore"] == 5.0  # mean(4, 6)
    assert result.loc[2, "NeighborhoodQualityScore"] == 8.0


def test_engineer_features_adds_columns(cleaned_dataset):
    engineered = engineer_features(cleaned_dataset)
    assert engineered.shape[1] > cleaned_dataset.shape[1]
