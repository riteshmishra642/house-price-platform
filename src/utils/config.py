"""
Configuration loader for the House Price Prediction Platform.

Loads config/config.yaml once and exposes it as a cached, dict-like,
attribute-accessible object so every module reads settings from a single
source of truth instead of hardcoding values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigNode(dict):
    """A dict that also allows attribute-style access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key '{item}'. Check config/config.yaml."
            ) from exc
        if isinstance(value, dict):
            return ConfigNode(value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> ConfigNode:
    """
    Load and cache the YAML configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML config file. Defaults to config/config.yaml at the
        project root, or the HOUSE_PRICE_CONFIG environment variable if set.

    Returns
    -------
    ConfigNode
        Nested, attribute-accessible configuration object.
    """
    path = Path(config_path or os.getenv("HOUSE_PRICE_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config: Dict[str, Any] = yaml.safe_load(f)

    return ConfigNode(raw_config)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return PROJECT_ROOT / relative_path
