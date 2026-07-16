"""
Model training and comparison module.

Trains every candidate model on the same train/test split and preprocessing
pipeline, evaluates each with a consistent metric set, and selects the best
model by cross-validated RMSE. This is the module that answers "which model
should we ship" with evidence, not intuition.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering.features import FeatureEngineer
from src.preprocessing.cleaning import DataCleaner
from src.preprocessing.pipeline import build_full_preprocessor
from src.training.model_factory import get_all_models
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Evaluation results for a single trained model."""

    name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: float
    mape: float
    cv_rmse_mean: float
    cv_rmse_std: float
    train_time_seconds: float

    def to_dict(self) -> dict:
        return self.__dict__


def _adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Adjusted R^2 penalizes adding features that don't improve fit,
    which plain R^2 rewards by default -- important when comparing models
    with very different effective feature counts (e.g. linear + one-hot vs
    tree models using raw columns)."""
    if n_samples - n_features - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error -- the metric a business stakeholder
    intuitively understands ('predictions are off by X% on average')."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def prepare_train_test_data(random_seed: int = 42):
    """
    Run the full data pipeline (ingest -> clean -> engineer) and split into
    train/test. Cleaning statistics (medians/modes) are fit on the training
    split ONLY to prevent test-set information leaking into imputation.
    """
    from src.pipelines.data_ingestion import load_or_create_raw_dataset

    config = load_config()
    target_col = config.data.target_column

    raw_df = load_or_create_raw_dataset()
    train_raw, test_raw = train_test_split(raw_df, test_size=config.data.test_size, random_state=random_seed)

    cleaner = DataCleaner(
        target_column=target_col,
        outlier_iqr_multiplier=config.preprocessing.outlier_iqr_multiplier,
    )
    train_clean = cleaner.fit_transform(train_raw)
    test_clean = cleaner.transform(test_raw, cap_outliers=False)

    engineer = FeatureEngineer()
    train_engineered = engineer.fit_transform(train_clean)
    test_engineered = engineer.fit_transform(test_clean)

    y_train = train_engineered[target_col].copy()
    y_test = test_engineered[target_col].copy()
    X_train = train_engineered.drop(columns=[target_col, "Id"], errors="ignore")
    X_test = test_engineered.drop(columns=[target_col, "Id"], errors="ignore")

    # Align columns in case a rare category appears only in one split.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test, y_train, y_test, cleaner


def train_and_evaluate_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    log_target: bool = True,
) -> tuple[Dict[str, Pipeline], List[ModelResult]]:
    """
    Train every candidate model (wrapped with the shared preprocessing
    pipeline) and evaluate on the held-out test set plus cross-validation.

    log_target=True trains on log1p(SalePrice): housing prices are strongly
    right-skewed, and modeling log-price makes errors proportionally
    consistent across cheap and expensive homes instead of the model being
    dominated by absolute-dollar errors on expensive outliers.
    """
    config = load_config()
    preprocessor = build_full_preprocessor(pd.concat([X_train, y_train.rename("SalePrice")], axis=1))
    models = get_all_models(random_seed=config.project.random_seed)

    y_train_model = np.log1p(y_train) if log_target else y_train

    fitted_pipelines: Dict[str, Pipeline] = {}
    results: List[ModelResult] = []

    for name, estimator in models.items():
        logger.info("Training model: %s", name)
        start = time.time()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        try:
            pipeline.fit(X_train, y_train_model)
        except Exception as exc:  # noqa: BLE001
            logger.error("Model '%s' failed to train: %s. Skipping.", name, exc)
            continue
        train_time = time.time() - start

        y_pred_model = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_model) if log_target else y_pred_model
        y_pred = np.clip(y_pred, a_min=0, a_max=None)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        adj_r2 = _adjusted_r2(r2, n_samples=len(y_test), n_features=X_test.shape[1])
        mape = _mape(y_test.to_numpy(), y_pred)

        # Expensive meta-estimators (stacking/voting) already perform internal
        # cross-validation during fit; wrapping them in another full outer CV
        # multiplies training cost with little added insight, so we use a
        # cheaper 2-fold check for those and the configured fold count for
        # everything else.
        outer_cv_folds = 2 if name in ("stacking", "voting") else config.models.cv_folds
        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train_model,
                cv=KFold(n_splits=outer_cv_folds, shuffle=True, random_state=config.project.random_seed),
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            cv_rmse_mean = float(-cv_scores.mean())
            cv_rmse_std = float(cv_scores.std())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-validation failed for '%s': %s", name, exc)
            cv_rmse_mean, cv_rmse_std = float("nan"), float("nan")

        result = ModelResult(
            name=name,
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            adjusted_r2=adj_r2,
            mape=mape,
            cv_rmse_mean=cv_rmse_mean,
            cv_rmse_std=cv_rmse_std,
            train_time_seconds=train_time,
        )
        results.append(result)
        fitted_pipelines[name] = pipeline
        logger.info(
            "%s -> RMSE=%.2f | R2=%.4f | MAPE=%.2f%% | CV_RMSE=%.2f (+/- %.2f) | %.1fs",
            name,
            rmse,
            r2,
            mape,
            cv_rmse_mean,
            cv_rmse_std,
            train_time,
        )

    results.sort(key=lambda r: r.rmse)
    return fitted_pipelines, results


def select_best_model(results: List[ModelResult]) -> str:
    """Best model = lowest test-set RMSE (primary business-relevant error metric)."""
    if not results:
        raise RuntimeError("No models were trained successfully.")
    return results[0].name


def save_artifacts(
    fitted_pipelines: Dict[str, Pipeline],
    results: List[ModelResult],
    best_model_name: str,
    feature_names: List[str],
) -> None:
    """Persist the best model, full leaderboard, and feature names to disk."""
    config = load_config()
    model_dir = resolve_path(config.paths.model_dir)
    reports_dir = resolve_path(config.paths.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(best_pipeline, resolve_path(config.paths.best_model_file))

    registry = {
        "best_model": best_model_name,
        "trained_at": pd.Timestamp.now().isoformat(),
        "leaderboard": [r.to_dict() for r in results],
    }
    with open(resolve_path(config.paths.model_registry_file), "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)

    with open(resolve_path(config.paths.feature_names_file), "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f, indent=2)

    with open(resolve_path(config.paths.metrics_file), "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)

    logger.info("Saved best model ('%s') and artifacts to %s", best_model_name, model_dir)


def run_training_pipeline() -> tuple[str, List[ModelResult]]:
    """End-to-end entry point: prepare data, train all models, save the winner."""
    config = load_config()
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data(random_seed=config.project.random_seed)
    fitted_pipelines, results = train_and_evaluate_all_models(X_train, X_test, y_train, y_test)
    best_model_name = select_best_model(results)
    save_artifacts(fitted_pipelines, results, best_model_name, list(X_train.columns))

    try:
        from src.training.mlflow_tracking import log_all_results

        log_all_results([r.to_dict() for r in results], fitted_pipelines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging step failed (non-fatal): %s", exc)

    return best_model_name, results


if __name__ == "__main__":
    best_name, all_results = run_training_pipeline()
    print(f"\nBest model: {best_name}\n")
    print(f"{'Model':<20}{'RMSE':>12}{'MAE':>12}{'R2':>10}{'MAPE%':>10}{'CV_RMSE':>12}")
    for r in all_results:
        print(f"{r.name:<20}{r.rmse:>12.1f}{r.mae:>12.1f}{r.r2:>10.4f}{r.mape:>10.2f}{r.cv_rmse_mean:>12.1f}")
