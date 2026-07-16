"""
MLflow experiment tracking integration.

Wraps the training pipeline so every model's parameters, metrics, and the
serialized pipeline artifact are logged to a local MLflow tracking store
(reports/mlruns) -- giving a full experiment history you can browse with:

    mlflow ui --backend-store-uri reports/mlruns

mlflow is treated as an optional dependency (same pattern as elsewhere in
the platform): if it isn't installed, training still runs and produces
models/model_registry.json as the source of truth; MLflow just adds a
browsable UI/history on top.
"""

from __future__ import annotations

from typing import Dict, List

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAS_MLFLOW = False
    logger.warning("mlflow not installed - experiment tracking will be skipped.")


def _configure_mlflow() -> None:
    config = load_config()
    tracking_uri = str(resolve_path(config.mlflow.tracking_uri))
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    mlflow.set_experiment(config.mlflow.experiment_name)


def log_training_run(
    model_name: str,
    params: Dict,
    metrics: Dict,
    model_pipeline=None,
) -> None:
    """Log a single model's training run. No-op (with a warning already
    logged at import time) if mlflow isn't installed."""
    if not _HAS_MLFLOW:
        return

    _configure_mlflow()
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if model_pipeline is not None:
            try:
                mlflow.sklearn.log_model(model_pipeline, artifact_path="model")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not log model artifact to MLflow for '%s': %s", model_name, exc)
        logger.info("Logged MLflow run for '%s'.", model_name)


def log_all_results(results: List[Dict], fitted_pipelines: Dict) -> None:
    """Log every model's leaderboard entry as its own MLflow run."""
    if not _HAS_MLFLOW:
        logger.info("Skipping MLflow logging for all results (mlflow not installed).")
        return

    for result in results:
        name = result["name"]
        pipeline = fitted_pipelines.get(name)
        model_params = {}
        if pipeline is not None:
            try:
                model_params = {f"model__{k}": v for k, v in pipeline.named_steps["model"].get_params().items()}
            except Exception:  # noqa: BLE001
                model_params = {}
        log_training_run(
            model_name=name,
            params=model_params,
            metrics={
                "rmse": result["rmse"],
                "mae": result["mae"],
                "r2": result["r2"],
                "mape": result["mape"],
                "cv_rmse_mean": result["cv_rmse_mean"],
            },
            model_pipeline=pipeline,
        )


if __name__ == "__main__":
    if _HAS_MLFLOW:
        print("mlflow is installed and configured. Tracking URI:", mlflow.get_tracking_uri())
    else:
        print("mlflow is not installed in this environment.")
