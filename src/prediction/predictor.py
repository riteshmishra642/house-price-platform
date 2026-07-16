"""
Inference module: the single entry point both the FastAPI service and the
Streamlit dashboard call to turn raw property attributes into a prediction.

Centralizing inference here (rather than duplicating "load model, build a
DataFrame, predict" logic in both api/main.py and dashboard/app.py) is what
guarantees the API and dashboard can never silently drift apart in how they
compute a price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering.features import FeatureEngineer
from src.preprocessing.cleaning import DataCleaner
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from src.explainability.shap_explainer import _HAS_SHAP, explain_single_prediction
except ImportError:  # pragma: no cover
    _HAS_SHAP = False


@dataclass
class PredictionResult:
    """Structured prediction output returned by both the API and dashboard."""

    predicted_price: float
    confidence_interval_low: float
    confidence_interval_high: float
    model_name: str
    explanation: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "predicted_price": round(self.predicted_price, 2),
            "confidence_interval_low": round(self.confidence_interval_low, 2),
            "confidence_interval_high": round(self.confidence_interval_high, 2),
            "model_name": self.model_name,
            "explanation": self.explanation,
        }


class HousePricePredictor:
    """
    Loads the trained pipeline once (expensive: deserializing a fitted
    sklearn Pipeline) and reuses it across many predict() calls -- critical
    for API latency, since reloading from disk on every request would add
    tens to hundreds of milliseconds per call.
    """

    def __init__(self):
        config = load_config()
        self.config = config
        self.pipeline = joblib.load(resolve_path(config.paths.best_model_file))

        with open(resolve_path(config.paths.feature_names_file), "r", encoding="utf-8") as f:
            import json

            self.feature_names: List[str] = json.load(f)

        import json as _json

        with open(resolve_path(config.paths.model_registry_file), "r", encoding="utf-8") as f:
            registry = _json.load(f)
        self.model_name = registry["best_model"]

        # Approximate residual std (in log space) used to build a simple
        # symmetric confidence interval around each point prediction. Loaded
        # from the leaderboard produced by trainer.py's best model entry.
        best_entry = next((r for r in registry["leaderboard"] if r["name"] == self.model_name), None)
        self._rmse = best_entry["rmse"] if best_entry else 0.0

        self._cleaner = DataCleaner(target_column=config.data.target_column)
        self._feature_engineer = FeatureEngineer()

        # Fit the cleaner's imputation statistics on the processed training
        # data so a single incoming API row can be cleaned consistently
        # with how training data was cleaned.
        self._fit_cleaner_reference()

        logger.info("HousePricePredictor initialized with model '%s'.", self.model_name)

    def _fit_cleaner_reference(self) -> None:
        from src.pipelines.data_ingestion import load_or_create_raw_dataset

        raw_df = load_or_create_raw_dataset()
        self._cleaner.fit(raw_df)

    def _build_feature_row(self, property_input: Dict[str, Any]) -> pd.DataFrame:
        """Convert a single property's raw attributes into a one-row
        DataFrame matching the schema the pipeline was trained on."""
        row = pd.DataFrame([property_input])
        cleaned = self._cleaner.transform(row, cap_outliers=False)
        engineered = self._feature_engineer.transform(cleaned)
        engineered = engineered.reindex(columns=self.feature_names, fill_value=0)
        return engineered

    def predict(self, property_input: Dict[str, Any], explain: bool = False) -> PredictionResult:
        """Predict a single property's price, optionally with a SHAP explanation."""
        X = self._build_feature_row(property_input)
        log_pred = self.pipeline.predict(X)[0]
        price = float(np.expm1(log_pred))
        price = max(price, 0.0)

        # Symmetric interval derived from the model's historical test RMSE --
        # a simple, transparent uncertainty estimate (not a full predictive
        # distribution, which would require quantile regression or a
        # Bayesian model, but sufficient to signal "how much to trust this").
        low = max(price - 1.5 * self._rmse, 0.0)
        high = price + 1.5 * self._rmse

        explanation = None
        if explain and _HAS_SHAP:
            try:
                from src.pipelines.data_ingestion import load_or_create_raw_dataset
                from src.preprocessing.cleaning import clean_dataset

                background_raw = load_or_create_raw_dataset().sample(
                    min(100, len(load_or_create_raw_dataset())), random_state=42
                )
                background_clean, _ = clean_dataset(background_raw)
                background_engineered = self._feature_engineer.transform(background_clean)
                background_engineered = background_engineered.reindex(columns=self.feature_names, fill_value=0)
                explanation = explain_single_prediction(self.pipeline, background_engineered, X)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Explanation generation failed: %s", exc)
                explanation = {"error": "Explanation unavailable for this request."}
        elif explain and not _HAS_SHAP:
            explanation = {"error": "shap is not installed in this environment."}

        return PredictionResult(
            predicted_price=price,
            confidence_interval_low=low,
            confidence_interval_high=high,
            model_name=self.model_name,
            explanation=explanation,
        )

    def predict_batch(self, property_inputs: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Predict prices for multiple properties. Kept as a loop over
        predict() (rather than a single batched call) so a single malformed
        row cannot fail the entire batch -- see the error handling in the
        API's /batch_predict endpoint, which catches per-row failures."""
        return [self.predict(item) for item in property_inputs]


@lru_cache(maxsize=1)
def get_predictor() -> HousePricePredictor:
    """
    Module-level singleton accessor. Both the FastAPI app and Streamlit
    dashboard call this instead of instantiating HousePricePredictor
    directly, so the (expensive) model load happens exactly once per
    process regardless of how many requests/reruns follow.
    """
    return HousePricePredictor()


if __name__ == "__main__":
    predictor = get_predictor()
    sample_property = {
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
    result = predictor.predict(sample_property, explain=False)
    print(result.to_dict())
