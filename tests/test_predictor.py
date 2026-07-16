"""
Unit tests for src/prediction/predictor.py

These tests require a trained model to already exist at
models/best_model.joblib (produced by `python -m src.training.trainer`).
If it hasn't been trained yet, the tests are skipped rather than failed,
since "no model trained yet" is an expected state in a fresh checkout,
not a bug.
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
def predictor():
    from src.prediction.predictor import HousePricePredictor

    return HousePricePredictor()


def test_predict_returns_positive_price(predictor, sample_property):
    result = predictor.predict(sample_property)
    assert result.predicted_price > 0


def test_predict_confidence_interval_contains_point_estimate(predictor, sample_property):
    result = predictor.predict(sample_property)
    assert result.confidence_interval_low <= result.predicted_price <= result.confidence_interval_high


def test_predict_batch_returns_same_length(predictor, sample_property):
    results = predictor.predict_batch([sample_property, sample_property, sample_property])
    assert len(results) == 3


def test_predict_is_deterministic(predictor, sample_property):
    result_a = predictor.predict(sample_property)
    result_b = predictor.predict(sample_property)
    assert result_a.predicted_price == result_b.predicted_price


def test_predict_higher_quality_yields_higher_price(predictor, sample_property):
    low_quality = dict(sample_property, OverallQual=3, KitchenQual="Fa", ExterQual="Fa")
    high_quality = dict(sample_property, OverallQual=9, KitchenQual="Ex", ExterQual="Ex")
    low_result = predictor.predict(low_quality)
    high_result = predictor.predict(high_quality)
    assert high_result.predicted_price > low_result.predicted_price
