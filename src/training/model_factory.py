"""
Model factory: single source of truth for every candidate regression model.

Centralizing model construction here (rather than scattering
`XGBRegressor(...)` calls across scripts) means hyperparameter defaults,
random seeds, and n_jobs settings are consistent everywhere the model is
built — training, tuning, and any future retraining job.
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Gradient-boosting libraries are treated as optional at import time. In
# restricted/offline environments (locked-down CI runners, air-gapped
# sandboxes) these packages may not be installable; the rest of the platform
# (cleaning, feature engineering, sklearn models, API, dashboard) must still
# run correctly. When unavailable, get_all_models() simply omits them and
# logs a warning instead of crashing the whole pipeline.
try:
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False
    logger.warning("xgboost not installed - xgboost model will be skipped.")

try:
    from lightgbm import LGBMRegressor

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    _HAS_LIGHTGBM = False
    logger.warning("lightgbm not installed - lightgbm model will be skipped.")

try:
    from catboost import CatBoostRegressor

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    _HAS_CATBOOST = False
    logger.warning("catboost not installed - catboost model will be skipped.")


def get_base_models(random_seed: int = 42) -> Dict[str, object]:
    """
    Return every base candidate model, keyed by the names used throughout
    config.yaml (models.candidates) and the model comparison reports.
    """
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10.0, random_state=random_seed),
        "lasso": Lasso(alpha=0.001, random_state=random_seed, max_iter=10000),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_seed, max_iter=10000),
        "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=random_seed),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=random_seed),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=random_seed),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_seed
        ),
        "svr": SVR(kernel="rbf", C=10.0, epsilon=0.01),
        "knn": KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
    }

    if _HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            n_jobs=-1,
            verbosity=0,
        )
    if _HAS_LIGHTGBM:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    if _HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            random_state=random_seed,
            verbose=False,
        )
    return models


def get_ensemble_models(random_seed: int = 42) -> Dict[str, object]:
    """
    Ensemble models (stacking/voting) built on top of a curated subset of
    strong base learners. Kept separate from get_base_models because they
    depend on other fitted estimators and are more expensive to train.
    """
    base = get_base_models(random_seed)
    preferred_order = ["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "gradient_boosting"]
    available = [name for name in preferred_order if name in base]
    chosen = available[:3] if len(available) >= 3 else available
    estimators = [(name, base[name]) for name in chosen]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=random_seed),
        n_jobs=-1,
        cv=3,
    )
    voting = VotingRegressor(estimators=estimators, n_jobs=-1)

    return {"stacking": stacking, "voting": voting}


def get_all_models(random_seed: int = 42) -> Dict[str, object]:
    """Return the full candidate model zoo: base models + ensembles."""
    models = get_base_models(random_seed)
    models.update(get_ensemble_models(random_seed))
    return models


def get_model(name: str, random_seed: int = 42):
    """Fetch a single model by its config-key name."""
    models = get_all_models(random_seed)
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models.keys())}")
    return models[name]
