"""
Unit tests for api/main.py, using FastAPI's TestClient (httpx-based).

Skipped automatically if no trained model exists yet, for the same reason
as test_predictor.py.
"""

from __future__ import annotations

import pytest

from src.utils.config import load_config, resolve_path

config = load_config()
_MODEL_EXISTS = resolve_path(config.paths.best_model_file).exists()

pytestmark = pytest.mark.skipif(
    not _MODEL_EXISTS, reason="No trained model found. Run `python -m src.training.trainer` first."
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True


def test_predict_endpoint_returns_valid_price(client, sample_property):
    payload = {"property": sample_property, "explain": False}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price"] > 0


def test_predict_endpoint_rejects_invalid_quality_range(client, sample_property):
    invalid_property = dict(sample_property, OverallQual=99)  # out of 1-10 range
    payload = {"property": invalid_property, "explain": False}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_batch_predict_endpoint(client, sample_property):
    payload = {"properties": [sample_property, sample_property], "explain": False}
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["failed_indices"] == []


def test_model_info_endpoint(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model_name" in response.json()


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "leaderboard" in response.json()
