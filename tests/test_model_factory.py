"""Unit tests for src/training/model_factory.py"""

from __future__ import annotations

from src.training.model_factory import get_all_models, get_base_models, get_model


def test_get_base_models_returns_non_empty_dict():
    models = get_base_models()
    assert len(models) >= 8  # at minimum, all sklearn-only models


def test_all_models_are_fit_predict_compatible():
    models = get_all_models()
    for name, model in models.items():
        assert hasattr(model, "fit"), f"{name} missing fit()"
        assert hasattr(model, "predict"), f"{name} missing predict()"


def test_get_model_returns_same_type_as_dict():
    models = get_all_models()
    for name in models:
        assert type(get_model(name)) is type(models[name])


def test_get_model_raises_on_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        get_model("not_a_real_model")


def test_random_seed_is_respected_for_reproducibility():
    from src.training.model_factory import get_model

    model_a = get_model("random_forest", random_seed=7)
    model_b = get_model("random_forest", random_seed=7)
    assert model_a.get_params()["random_state"] == model_b.get_params()["random_state"] == 7
