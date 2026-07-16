"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.features import FeatureEngineer
from src.pipelines.data_ingestion import load_or_create_raw_dataset
from src.preprocessing.cleaning import DataCleaner


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    """Loads (or generates via synthetic fallback) the raw dataset once per
    test session -- expensive to regenerate per-test, and tests should not
    mutate it (each test that needs a mutable copy should call .copy())."""
    return load_or_create_raw_dataset()


@pytest.fixture(scope="session")
def cleaned_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner(target_column="SalePrice")
    return cleaner.fit_transform(raw_dataset)


@pytest.fixture(scope="session")
def engineered_dataset(cleaned_dataset: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().fit_transform(cleaned_dataset)


@pytest.fixture
def sample_property() -> dict:
    """A single, valid, realistic property payload used across
    prediction/API tests."""
    return {
        "MSSubClass": "60", "MSZoning": "RL", "LotFrontage": 80.0, "LotArea": 9600,
        "Street": "Pave", "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub",
        "Neighborhood": "CollgCr", "BldgType": "1Fam", "HouseStyle": "2Story",
        "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003,
        "RoofStyle": "Gable", "Exterior1st": "VinylSd", "MasVnrArea": 196.0,
        "ExterQual": "Gd", "Foundation": "PConc", "BsmtQual": "Gd", "BsmtCond": "TA",
        "TotalBsmtSF": 856, "HeatingQC": "Ex", "CentralAir": "Y",
        "1stFlrSF": 856, "2ndFlrSF": 854, "GrLivArea": 1710,
        "BsmtFullBath": 1, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3,
        "KitchenQual": "Gd", "TotRmsAbvGrd": 8, "Fireplaces": 1,
        "GarageType": "Attchd", "GarageYrBlt": 2003, "GarageCars": 2, "GarageArea": 548,
        "GarageQual": "TA", "WoodDeckSF": 0, "OpenPorchSF": 61, "PoolArea": 0,
        "Fence": "None", "MoSold": 2, "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal",
    }
