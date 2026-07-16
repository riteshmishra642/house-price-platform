"""Unit tests for src/utils/config.py and src/utils/logger.py"""

from __future__ import annotations

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger


def test_config_loads_and_caches():
    config_a = load_config()
    config_b = load_config()
    assert config_a is config_b  # lru_cache returns the same object


def test_config_attribute_access():
    config = load_config()
    assert config.project.name == "house-price-platform"
    assert config.data.target_column == "SalePrice"


def test_config_nested_attribute_access_returns_node():
    config = load_config()
    assert config.preprocessing.missing_value_strategy.numeric == "median"


def test_resolve_path_is_absolute():
    path = resolve_path("data/raw")
    assert path.is_absolute()


def test_get_logger_returns_usable_logger():
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    logger.info("Test log message from test_utils.py")  # should not raise
