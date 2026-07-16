"""
Pydantic request/response schemas for the House Price Prediction API.

Defining these explicitly (rather than accepting a raw dict) gives us
automatic validation, OpenAPI/Swagger documentation, and clear error
messages when a client sends a malformed request -- all for free from
FastAPI + Pydantic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PropertyFeatures(BaseModel):
    """
    Raw property attributes accepted by the /predict endpoint. Field names
    intentionally mirror the Ames Housing Data Dictionary column names so
    the API contract is self-documenting for anyone familiar with the
    dataset.
    """

    MSSubClass: str = Field(..., description="Building class code, e.g. '60'")
    MSZoning: str = Field(..., description="General zoning classification, e.g. 'RL'")
    LotFrontage: float = Field(..., ge=0, le=500, description="Linear feet of street connected to property")
    LotArea: int = Field(..., ge=500, le=250000, description="Lot size in square feet")
    Street: str = Field(default="Pave")
    LotShape: str = Field(default="Reg")
    LandContour: str = Field(default="Lvl")
    Utilities: str = Field(default="AllPub")
    Neighborhood: str = Field(..., description="Physical location within Ames city limits")
    BldgType: str = Field(default="1Fam")
    HouseStyle: str = Field(..., description="e.g. '1Story', '2Story'")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material/finish quality, 1 (worst) - 10 (best)")
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition rating, 1 (worst) - 10 (best)")
    YearBuilt: int = Field(..., ge=1800, le=2026)
    YearRemodAdd: int = Field(..., ge=1800, le=2026)
    RoofStyle: str = Field(default="Gable")
    Exterior1st: str = Field(default="VinylSd")
    MasVnrArea: float = Field(default=0.0, ge=0)
    ExterQual: str = Field(..., description="Exterior quality: Po/Fa/TA/Gd/Ex")
    Foundation: str = Field(default="PConc")
    BsmtQual: str = Field(default="TA", description="Basement quality: Po/Fa/TA/Gd/Ex/None")
    BsmtCond: str = Field(default="TA")
    TotalBsmtSF: float = Field(..., ge=0)
    HeatingQC: str = Field(default="TA")
    CentralAir: str = Field(default="Y")
    FirstFlrSF: float = Field(..., ge=0, alias="1stFlrSF")
    SecondFlrSF: float = Field(default=0.0, ge=0, alias="2ndFlrSF")
    GrLivArea: float = Field(..., ge=200, description="Above-grade living area, square feet")
    BsmtFullBath: int = Field(default=0, ge=0, le=4)
    FullBath: int = Field(..., ge=0, le=6)
    HalfBath: int = Field(default=0, ge=0, le=4)
    BedroomAbvGr: int = Field(..., ge=0, le=12)
    KitchenQual: str = Field(..., description="Kitchen quality: Po/Fa/TA/Gd/Ex")
    TotRmsAbvGrd: int = Field(..., ge=1, le=20)
    Fireplaces: int = Field(default=0, ge=0, le=5)
    GarageType: str = Field(default="None")
    GarageYrBlt: Optional[float] = Field(default=None)
    GarageCars: int = Field(default=0, ge=0, le=6)
    GarageArea: float = Field(default=0.0, ge=0)
    GarageQual: str = Field(default="None")
    WoodDeckSF: float = Field(default=0.0, ge=0)
    OpenPorchSF: float = Field(default=0.0, ge=0)
    PoolArea: float = Field(default=0.0, ge=0)
    Fence: str = Field(default="None")
    MoSold: int = Field(..., ge=1, le=12)
    YrSold: int = Field(..., ge=2000, le=2026)
    SaleType: str = Field(default="WD")
    SaleCondition: str = Field(default="Normal")

    model_config = {"populate_by_name": True}

    @field_validator("YearRemodAdd")
    @classmethod
    def remod_not_before_built(cls, value: int, info) -> int:
        year_built = info.data.get("YearBuilt")
        if year_built is not None and value < year_built:
            raise ValueError("YearRemodAdd cannot be earlier than YearBuilt.")
        return value


class PredictionRequest(BaseModel):
    property: PropertyFeatures
    explain: bool = Field(default=False, description="If true, include a SHAP-based explanation.")


class BatchPredictionRequest(BaseModel):
    properties: List[PropertyFeatures]
    explain: bool = Field(default=False)


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval_low: float
    confidence_interval_high: float
    model_name: str
    explanation: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    failed_indices: List[int] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    test_rmse: float
    test_r2: float
    test_mape: float
    feature_count: int


class MetricsResponse(BaseModel):
    leaderboard: List[Dict]
