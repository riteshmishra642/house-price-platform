"""
Centralized logging setup for the House Price Prediction Platform.

Every module calls get_logger(__name__) instead of configuring its own
handlers, so log format/level/destination is controlled in one place.
"""

from __future__ import annotations

import logging
import sys

from src.utils.config import load_config, resolve_path

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    config = load_config()
    log_cfg = config.logging
    logs_dir = resolve_path(config.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=log_cfg.format,
        datefmt=log_cfg.date_format,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls (e.g. under pytest/reload).
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
    """
    _configure_root_logger()
    return logging.getLogger(name)
