"""Config loading and the few names every teacher module shares."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "teacher.yaml"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

# Fixed order. Every file and every loop uses this order.
LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the same format as the data pipeline's."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("DISTILLATE_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load teacher.yaml and check that its label list matches ``LABELS``."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if tuple(cfg.get("labels", ())) != LABELS:
        raise ValueError(f"config labels {cfg.get('labels')} must equal {list(LABELS)}")
    return cfg


def model_version(cfg: dict[str, Any]) -> str:
    """The ``model_version`` string written into every shard: ``<model_id>@<revision>``."""
    model = cfg["model"]
    if not model.get("revision"):
        raise ValueError("model.revision must be pinned to a commit hash")
    return f"{model['model_id']}@{model['revision']}"
