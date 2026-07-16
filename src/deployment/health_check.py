"""
Deployment utilities.

Small, dependency-light helpers used by deployment tooling (CI, a release
script, or an orchestrator like Kubernetes/Docker Compose) to verify a
running instance of the API is healthy and serving the expected model
before routing production traffic to it.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import requests

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def wait_for_api_ready(
    base_url: str = "http://localhost:8000",
    timeout_seconds: int = 60,
    poll_interval_seconds: float = 2.0,
) -> bool:
    """
    Poll the API's /health endpoint until it reports a loaded model or the
    timeout elapses. Intended for use in a deployment script: don't mark a
    rollout as successful, or start sending it traffic, until this returns
    True.
    """
    deadline = time.time() + timeout_seconds
    last_error: Optional[Exception] = None

    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200 and response.json().get("model_loaded"):
                logger.info("API at %s is ready: %s", base_url, response.json())
                return True
        except requests.RequestException as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(poll_interval_seconds)

    logger.error(
        "API at %s did not become ready within %ds (last error: %s)",
        base_url,
        timeout_seconds,
        last_error,
    )
    return False


def verify_model_registry_matches_deployed(base_url: str = "http://localhost:8000") -> bool:
    """
    Sanity check for a deployment pipeline: confirms the model the *running*
    API reports via /model_info matches what's currently in
    models/model_registry.json on disk -- catches the class of bug where a
    container was built from a stale image and is serving an old model.
    """
    config = load_config()
    import json

    from src.utils.config import resolve_path

    try:
        with open(resolve_path(config.paths.model_registry_file), "r", encoding="utf-8") as f:
            registry = json.load(f)
    except FileNotFoundError:
        logger.error("No local model_registry.json found to compare against.")
        return False

    try:
        response = requests.get(f"{base_url}/model_info", timeout=10)
        response.raise_for_status()
        deployed_info = response.json()
    except requests.RequestException as exc:  # noqa: BLE001
        logger.error("Could not reach %s/model_info: %s", base_url, exc)
        return False

    matches = deployed_info.get("model_name") == registry.get("best_model")
    if matches:
        logger.info("Deployed model ('%s') matches local registry.", deployed_info.get("model_name"))
    else:
        logger.warning(
            "Deployed model ('%s') does NOT match local registry ('%s').",
            deployed_info.get("model_name"),
            registry.get("best_model"),
        )
    return matches


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    ready = wait_for_api_ready(base_url=url)
    if not ready:
        sys.exit(1)
    verify_model_registry_matches_deployed(base_url=url)
