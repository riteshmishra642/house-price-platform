"""
Explainable AI module, built on SHAP (SHapley Additive exPlanations).

Why SHAP over simpler alternatives (e.g. raw coefficients or impurity
importance): SHAP values are additive and consistent -- for any single
prediction, the SHAP values of every feature sum exactly to
(prediction - baseline), so we can honestly say "this feature added
exactly $X to this specific house's predicted price," which raw feature
importances cannot do (they describe the model globally, not one
prediction). This is the mechanism that fulfills the platform's core
promise from docs/01_business_understanding.md: every prediction must be
individually justifiable, not just accurate on average.

shap is treated as an optional dependency (same pattern used for the
boosting libraries and optuna) so the rest of the platform still runs if
it isn't installed; only this module's functionality is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import shap

    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False
    logger.warning("shap not installed - explainability plots will be unavailable.")


def _require_shap() -> None:
    if not _HAS_SHAP:
        raise ImportError(
            "shap is not installed. Install it with `pip install shap` to use " "the explainability module."
        )


def _figures_dir() -> Path:
    config = load_config()
    figures_dir = resolve_path(config.paths.figures_dir) / "shap"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def build_explainer(pipeline: Pipeline, X_background: pd.DataFrame):
    """
    Build a SHAP explainer for a fitted (preprocessor + model) sklearn
    Pipeline. `shap.Explainer` auto-selects the fastest exact/approximate
    algorithm for the given model type (TreeExplainer for tree models,
    LinearExplainer for linear models, KernelExplainer as a general
    fallback), so this single call works regardless of which model won
    Step 7's comparison.
    """
    _require_shap()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_background_transformed = preprocessor.transform(X_background)
    background_sample = shap.sample(X_background_transformed, min(100, len(X_background_transformed)))

    explainer = shap.Explainer(model, background_sample)
    return explainer, preprocessor


def compute_shap_values(explainer, pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 300):
    """Transform raw features through the pipeline's preprocessor, then
    compute SHAP values on the model-ready representation."""
    _require_shap()
    preprocessor = pipeline.named_steps["preprocessor"]
    X_sample = X.sample(min(max_samples, len(X)), random_state=42) if len(X) > max_samples else X
    X_transformed = preprocessor.transform(X_sample)
    shap_values = explainer(X_transformed)
    return shap_values, X_transformed


def plot_global_summary(shap_values, save_name: str = "shap_summary.png") -> Path:
    """
    Global Explanation: ranks every feature by its average impact magnitude
    across all predictions, and shows the direction of effect (color) --
    e.g. confirms whether high OverallQual consistently pushes price up.
    """
    _require_shap()
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, show=False)
    path = _figures_dir() / save_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP global summary plot to %s", path)
    return path


def plot_waterfall(shap_values, index: int, save_name: Optional[str] = None) -> Path:
    """
    Local Explanation for a single prediction: shows exactly which features
    pushed this one house's price up or down from the baseline, and by how
    much -- this is what powers the "Why this price?" section of the
    Streamlit dashboard (Step 13) and the /predict API response.
    """
    _require_shap()
    save_name = save_name or f"shap_waterfall_{index}.png"
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    path = _figures_dir() / save_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP waterfall plot to %s", path)
    return path


def plot_dependence(
    shap_values, feature_name: str, X_transformed: pd.DataFrame, save_name: Optional[str] = None
) -> Path:
    """
    Dependence Plot: shows how a single feature's SHAP contribution changes
    across its value range, and (via color) whether that relationship
    interacts with another feature -- e.g. does the price boost from
    GrLivArea depend on OverallQual.
    """
    _require_shap()
    save_name = save_name or f"shap_dependence_{feature_name}.png"
    fig = plt.figure(figsize=(8, 6))
    shap.plots.scatter(shap_values[:, feature_name], show=False)
    path = _figures_dir() / save_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP dependence plot for '%s' to %s", feature_name, path)
    return path


def plot_force(shap_values, index: int, save_name: Optional[str] = None) -> Path:
    """
    Force Plot: a compact, additive visual (push/pull forces) of a single
    prediction -- functionally similar information to the waterfall plot but
    in the format most commonly embedded in reports/dashboards.
    """
    _require_shap()
    save_name = save_name or f"shap_force_{index}.png"
    fig = plt.figure(figsize=(14, 3))
    shap.plots.force(shap_values[index], matplotlib=True, show=False)
    path = _figures_dir() / save_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP force plot to %s", path)
    return path


def plot_decision(shap_values, indices: list[int], save_name: str = "shap_decision.png") -> Path:
    """
    Decision Plot: traces multiple predictions' cumulative feature
    contributions on one chart -- useful for comparing a handful of
    properties (e.g. "why did house A price higher than house B") in a
    single view rather than one waterfall each.
    """
    _require_shap()
    fig = plt.figure(figsize=(9, 8))
    shap.plots.decision(
        shap_values.base_values[0],
        shap_values.values[indices],
        feature_names=list(shap_values.feature_names),
        show=False,
    )
    path = _figures_dir() / save_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP decision plot to %s", path)
    return path


def explain_single_prediction(pipeline: Pipeline, X_background: pd.DataFrame, X_instance: pd.DataFrame) -> dict:
    """
    Production-facing helper: returns a JSON-serializable per-feature SHAP
    breakdown for a single incoming prediction request. This is what the
    FastAPI /predict endpoint and Streamlit dashboard call directly (they
    do not need the plotting functions above, only these numbers).
    """
    _require_shap()
    explainer, preprocessor = build_explainer(pipeline, X_background)
    X_instance_transformed = preprocessor.transform(X_instance)
    shap_values = explainer(X_instance_transformed)

    contributions = {
        str(feature): float(value) for feature, value in zip(shap_values.feature_names, shap_values.values[0])
    }
    sorted_contributions = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
    return {
        "base_value": float(np.ravel(shap_values.base_values[0])[0]),
        "feature_contributions": sorted_contributions,
    }


def generate_full_explainability_report(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[Path]:
    """Generate the full Step-10 SHAP figure set (global + local explanations)."""
    _require_shap()
    explainer, _ = build_explainer(pipeline, X_train)
    shap_values, X_transformed = compute_shap_values(explainer, pipeline, X_test, max_samples=300)

    paths = [plot_global_summary(shap_values)]
    paths.append(plot_waterfall(shap_values, index=0))
    paths.append(plot_force(shap_values, index=0))

    top_feature = X_transformed.var().sort_values(ascending=False).index[0]
    paths.append(plot_dependence(shap_values, top_feature, X_transformed))
    paths.append(plot_decision(shap_values, indices=list(range(min(10, len(X_transformed))))))
    return paths


if __name__ == "__main__":
    import joblib

    from src.training.trainer import prepare_train_test_data

    config = load_config()
    pipeline = joblib.load(resolve_path(config.paths.best_model_file))
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data()

    if _HAS_SHAP:
        report_paths = generate_full_explainability_report(pipeline, X_train, X_test)
        print("Generated SHAP figures:")
        for p in report_paths:
            print(" -", p)
    else:
        print("shap is not installed in this environment; skipping explainability report generation.")
