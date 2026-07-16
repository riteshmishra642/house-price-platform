"""
Model evaluation and diagnostic visualization module.

Generates the plots a Principal Data Scientist would actually check before
signing off on a model: residual behavior, prediction error, learning
curves (bias/variance), and validation curves — saved to reports/figures
so they can be embedded in documentation or a portfolio README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, learning_curve, validation_curve
from sklearn.pipeline import Pipeline

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

plt.rcParams.update({"figure.autolayout": True, "font.size": 10})


def _figures_dir() -> Path:
    config = load_config()
    figures_dir = resolve_path(config.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Path:
    """
    Residuals (actual - predicted) should scatter randomly around zero with
    no pattern. A funnel shape means heteroscedasticity (the model is less
    reliable for expensive homes); a curve means a missing nonlinear term.
    """
    residuals = y_true.to_numpy() - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="#2563eb")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Sale Price")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title(f"Residual Plot — {model_name}")

    axes[1].hist(residuals, bins=40, color="#2563eb", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    path = _figures_dir() / f"residuals_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved residual plot to %s", path)
    return path


def plot_prediction_error(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Path:
    """
    Predicted vs. actual scatter with a perfect-prediction reference line.
    The tighter the cloud hugs the diagonal, the better — this is the single
    plot most stakeholders intuitively grasp fastest.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#16a34a")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect Prediction")
    ax.set_xlabel("Actual Sale Price")
    ax.set_ylabel("Predicted Sale Price")
    ax.set_title(f"Prediction Error Plot — {model_name}")
    ax.legend()

    path = _figures_dir() / f"prediction_error_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved prediction error plot to %s", path)
    return path


def plot_learning_curve(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, cv_folds: int = 3) -> Path:
    """
    Training-set-size vs. error for both train and validation splits.
    - Train and validation error both high & converged -> high bias (underfitting).
    - Large gap between train (low) and validation (high) error -> high
      variance (overfitting) -> more data or regularization would help.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        np.log1p(y),
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_rmse, "o-", label="Training RMSE (log scale)", color="#2563eb")
    ax.plot(train_sizes, val_rmse, "o-", label="Validation RMSE (log scale)", color="#dc2626")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("RMSE (log target)")
    ax.set_title(f"Learning Curve — {model_name}")
    ax.legend()

    path = _figures_dir() / f"learning_curve_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved learning curve to %s", path)
    return path


def plot_validation_curve(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    param_name: str,
    param_range: list,
    model_name: str,
    cv_folds: int = 3,
) -> Path:
    """
    Sweeps a single hyperparameter to show the bias/variance tradeoff it
    controls -- e.g. for Ridge, sweeping `model__alpha` shows underfitting
    (both curves worsen) at high alpha and overfitting (train << validation
    error) at very low alpha.
    """
    train_scores, val_scores = validation_curve(
        pipeline,
        X,
        np.log1p(y),
        param_name=param_name,
        param_range=param_range,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_axis = range(len(param_range))
    ax.plot(x_axis, train_rmse, "o-", label="Training RMSE", color="#2563eb")
    ax.plot(x_axis, val_rmse, "o-", label="Validation RMSE", color="#dc2626")
    ax.set_xticks(list(x_axis))
    ax.set_xticklabels([str(p) for p in param_range], rotation=45)
    ax.set_xlabel(param_name)
    ax.set_ylabel("RMSE (log target)")
    ax.set_title(f"Validation Curve — {model_name} ({param_name})")
    ax.legend()

    path = _figures_dir() / f"validation_curve_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved validation curve to %s", path)
    return path


def plot_model_comparison(results: List[dict]) -> Path:
    """Bar chart comparing every trained model's test RMSE, for the leaderboard."""
    df = pd.DataFrame(results).sort_values("rmse")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#16a34a" if i == 0 else "#2563eb" for i in range(len(df))]
    ax.barh(df["name"], df["rmse"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE ($)")
    ax.set_title("Model Comparison — Test RMSE (lower is better)")

    path = _figures_dir() / "model_comparison_rmse.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved model comparison chart to %s", path)
    return path


def generate_full_evaluation_report(
    fitted_pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    results: List[dict],
    best_model_name: str,
) -> List[Path]:
    """Generate the complete Step-9 evaluation figure set for the best model."""
    saved_paths = [plot_model_comparison(results)]

    best_pipeline = fitted_pipelines[best_model_name]
    y_pred = np.expm1(best_pipeline.predict(X_test))
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    saved_paths.append(plot_residuals(y_test, y_pred, best_model_name))
    saved_paths.append(plot_prediction_error(y_test, y_pred, best_model_name))
    saved_paths.append(plot_learning_curve(best_pipeline, X_train, y_train, best_model_name))

    if best_model_name == "ridge":
        saved_paths.append(
            plot_validation_curve(
                best_pipeline,
                X_train,
                y_train,
                param_name="model__alpha",
                param_range=[0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                model_name=best_model_name,
            )
        )

    return saved_paths


if __name__ == "__main__":
    from src.training.trainer import prepare_train_test_data, train_and_evaluate_all_models

    X_train, X_test, y_train, y_test, _ = prepare_train_test_data()
    fitted_pipelines, results = train_and_evaluate_all_models(X_train, X_test, y_train, y_test)
    best_name = results[0].name
    paths = generate_full_evaluation_report(
        fitted_pipelines,
        X_train,
        X_test,
        y_train,
        y_test,
        [r.to_dict() for r in results],
        best_name,
    )
    print("Generated figures:")
    for p in paths:
        print(" -", p)
