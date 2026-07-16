"""
FastAPI application: production-facing REST API for the House Price
Prediction Platform.

Run locally with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:      http://localhost:8000/docs
ReDoc:           http://localhost:8000/redoc
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.prediction.predictor import get_predictor
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

_predictor_ref = {"instance": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once at startup (not per-request) -- the standard
    production pattern for serving ML models behind an API. If loading
    fails (e.g. the model hasn't been trained yet), the API still starts
    so /health can report the problem instead of crashing on boot.
    """
    try:
        _predictor_ref["instance"] = get_predictor()
        logger.info("Model loaded successfully at API startup.")
    except Exception as exc:  # noqa: BLE001
        _predictor_ref["error"] = str(exc)
        logger.error("Failed to load model at startup: %s", exc)
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds an X-Process-Time header to every response -- useful for the
    API-response-time KPI defined in docs/01_business_understanding.md."""
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"
    return response


def _get_predictor_or_503():
    if _predictor_ref["instance"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {_predictor_ref['error'] or 'unknown error'}. "
            "Train a model first with `python -m src.training.trainer`.",
        )
    return _predictor_ref["instance"]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all handler so unexpected errors return clean JSON instead of
    a raw traceback -- never expose internals to API consumers."""
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred while processing the request."},
    )


@app.get("/", tags=["General"])
async def root():
    """Landing endpoint with basic API metadata and links to documentation."""
    return {
        "message": "House Price Prediction API",
        "version": config.api.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Liveness/readiness probe -- what a load balancer or orchestrator
    (Kubernetes, Docker Compose healthcheck) should poll."""
    predictor = _predictor_ref["instance"]
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict the sale price for a single property.

    Set `explain: true` to also receive a SHAP-based per-feature
    contribution breakdown (requires shap to be installed on the server).
    """
    predictor = _get_predictor_or_503()
    property_dict = request.property.model_dump(by_alias=True)
    try:
        result = predictor.predict(property_dict, explain=request.explain)
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result.to_dict())


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict sale prices for multiple properties in one call.

    Individual row failures do not fail the whole batch: failed rows are
    reported by index in `failed_indices` so a client can retry just those.
    """
    predictor = _get_predictor_or_503()
    predictions = []
    failed_indices = []

    for i, prop in enumerate(request.properties):
        try:
            result = predictor.predict(prop.model_dump(by_alias=True), explain=request.explain)
            predictions.append(PredictionResponse(**result.to_dict()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch item %d failed: %s", i, exc)
            failed_indices.append(i)

    return BatchPredictionResponse(predictions=predictions, failed_indices=failed_indices)


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Metadata about the currently deployed model -- what a monitoring
    dashboard or an engineer debugging a regression would check first."""
    import json

    predictor = _get_predictor_or_503()
    with open(resolve_path(config.paths.model_registry_file), "r", encoding="utf-8") as f:
        registry = json.load(f)

    best_entry = next((r for r in registry["leaderboard"] if r["name"] == registry["best_model"]), {})
    return ModelInfoResponse(
        model_name=registry["best_model"],
        trained_at=registry["trained_at"],
        test_rmse=best_entry.get("rmse", 0.0),
        test_r2=best_entry.get("r2", 0.0),
        test_mape=best_entry.get("mape", 0.0),
        feature_count=len(predictor.feature_names),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Model"])
async def metrics():
    """Full model leaderboard (every candidate model's metrics) -- what
    Step 9's evaluation table looks like as a machine-readable endpoint."""
    import json

    with open(resolve_path(config.paths.metrics_file), "r", encoding="utf-8") as f:
        leaderboard = json.load(f)
    return MetricsResponse(leaderboard=leaderboard)
