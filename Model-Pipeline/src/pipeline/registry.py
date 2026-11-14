"""Shared helpers for register_model.py and rollback_model.py: config, MLflow client, Slack.

The tracking server comes from ``MLFLOW_TRACKING_URI`` (the team VM; login from
``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD``), or ``--tracking-uri``. The
model registry lives on the same server.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from src.teacher.config import PIPELINE_ROOT, get_logger

logger = get_logger(__name__)
DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "registry.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def client(tracking_uri: str | None = None):
    """An ``MlflowClient`` for ``tracking_uri``, else ``MLFLOW_TRACKING_URI``."""
    import mlflow
    from mlflow import MlflowClient

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("No MLflow server: set MLFLOW_TRACKING_URI (and the username and "
                           "password variables) or pass --tracking-uri")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def version_by_alias(mlflow_client, name: str, alias: str):
    """The model version holding ``alias``, or None when no version holds it."""
    from mlflow.exceptions import MlflowException

    try:
        return mlflow_client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def notify(cfg: dict[str, Any], level: str, title: str, details: dict[str, Any]) -> bool:
    """Post to Slack when ``SLACK_WEBHOOK_URL`` is set; always log. Never raises.

    ``level`` is ``ok``, ``warn`` or ``fail``. Returns True when Slack accepted the message.
    """
    log = logger.error if level == "fail" else logger.info
    log("%s: %s %s", level.upper(), title, json.dumps(details, default=str))
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url or not cfg.get("notify", {}).get("enabled", True):
        return False
    icon = {"ok": ":white_check_mark:", "warn": ":warning:", "fail": ":x:"}.get(level, "")
    lines = [f"{icon} *{title}*"] + [f"• {k}: {v}" for k, v in details.items()]
    body = json.dumps({"text": "\n".join(lines)}).encode()
    request = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # a broken webhook must never fail a registry operation
        logger.warning("Slack post failed: %s", exc)
        return False
