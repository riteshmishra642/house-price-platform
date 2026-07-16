"""
Hyperparameter tuning module.

Provides two tuning strategies:
1. RandomizedSearchCV - fast, broad exploration of a hyperparameter grid.
2. Optuna - sequential, Bayesian-informed search that converges faster than
   random search once a promising region is found (used here for the
   final refinement pass on the model selected by trainer.py).

Optuna is treated as an optional dependency (same pattern as the boosting
libraries in model_factory.py) so the rest of the platform still works if
it isn't installed.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_full_preprocessor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler

    _HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    _HAS_OPTUNA = False
    logger.warning("optuna not installed - Optuna tuning will be unavailable (RandomizedSearchCV still works).")


# Search spaces per model, keyed by the model_factory.py candidate names.
# Kept intentionally small/sane so tuning finishes in reasonable time on a
# laptop, while still covering the parameters that matter most in practice.
PARAM_DISTRIBUTIONS: Dict[str, Dict[str, list]] = {
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]},
    "lasso": {"model__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]},
    "elasticnet": {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [6, 8, 10, 12, 16, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "extra_trees": {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [6, 8, 10, 12, 16, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
    },
    "xgboost": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [3, 4, 6, 8, -1],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
    },
    "svr": {
        "model__C": [0.1, 1.0, 10.0, 50.0, 100.0],
        "model__epsilon": [0.001, 0.01, 0.05, 0.1],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 10, 15, 20],
        "model__weights": ["uniform", "distance"],
    },
}


def tune_with_randomized_search(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv_folds: int = 3,
    random_seed: int = 42,
) -> RandomizedSearchCV:
    """
    Broad, fast exploration of the hyperparameter space.

    Why RandomizedSearchCV over GridSearchCV as the default: grid search's
    cost grows multiplicatively with every added hyperparameter, while
    random search samples a fixed budget (n_iter) regardless of grid size —
    for the 3-4 hyperparameter grids here, random search finds comparably
    good configurations in a fraction of the runtime.
    """
    if model_name not in PARAM_DISTRIBUTIONS:
        raise ValueError(f"No hyperparameter grid defined for '{model_name}'.")

    preprocessor = build_full_preprocessor(pd.concat([X_train, y_train.rename("SalePrice")], axis=1))
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS[model_name],
        n_iter=min(n_iter, _grid_size(PARAM_DISTRIBUTIONS[model_name])),
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed),
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        n_jobs=-1,
        verbose=0,
    )
    logger.info("Starting RandomizedSearchCV for '%s' (n_iter=%d, cv=%d)...", model_name, n_iter, cv_folds)
    search.fit(X_train, np.log1p(y_train))
    logger.info(
        "RandomizedSearchCV complete for '%s': best CV RMSE(log) = %.5f, best params = %s",
        model_name,
        -search.best_score_,
        search.best_params_,
    )
    return search


def _grid_size(param_distributions: Dict[str, list]) -> int:
    size = 1
    for values in param_distributions.values():
        size *= len(values)
    return size


def tune_with_optuna(
    model_name: str,
    model_builder,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 3,
    random_seed: int = 42,
):
    """
    Bayesian hyperparameter search via Optuna's Tree-structured Parzen
    Estimator sampler. Unlike RandomizedSearchCV, each trial's result informs
    the next trial's sampling distribution, so Optuna typically reaches a
    better score in fewer total trials once the search space is non-trivial.

    model_builder: callable(trial) -> unfitted sklearn-compatible estimator.
    """
    if not _HAS_OPTUNA:
        raise ImportError(
            "optuna is not installed. Install it with `pip install optuna` "
            "or use tune_with_randomized_search() instead."
        )

    preprocessor = build_full_preprocessor(pd.concat([X_train, y_train.rename("SalePrice")], axis=1))
    y_log = np.log1p(y_train)

    def objective(trial: "optuna.Trial") -> float:
        model = model_builder(trial)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(
            pipeline,
            X_train,
            y_log,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed),
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna tuning complete for '%s': best RMSE(log) = %.5f, best params = %s",
        model_name,
        study.best_value,
        study.best_params,
    )
    return study


def default_xgboost_optuna_builder(trial: "optuna.Trial"):
    """Example Optuna search space builder for XGBoost, used when available."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 150, 700),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


if __name__ == "__main__":
    from src.training.model_factory import get_model
    from src.training.trainer import prepare_train_test_data

    config = load_config()
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data(random_seed=config.project.random_seed)

    best_name = "ridge"
    search_result = tune_with_randomized_search(
        best_name,
        get_model(best_name),
        X_train,
        y_train,
        n_iter=config.models.hyperparameter_tuning.n_iter,
        cv_folds=3,
        random_seed=config.project.random_seed,
    )

    y_pred = np.expm1(search_result.predict(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"Tuned '{best_name}' test RMSE: {rmse:.2f}")
    print("Best params:", search_result.best_params_)
