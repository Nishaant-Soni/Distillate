"""Shared helpers for every pipeline stage.

Contract (see CONTRACTS.md):
- log only through ``get_logger``
- write outputs only through ``atomic_write`` / ``atomic_path``
- every stage returns a summary dict and saves it with ``write_summary``
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PIPELINE_ROOT / "config" / "pipeline.yaml"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the pipeline's standard format.

    Airflow captures stdout/stderr into task logs, so a stream handler is enough.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("DISTILLATE_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline.yaml and apply environment overrides.

    Relative paths under ``paths`` are resolved against the Data-Pipeline root,
    so scripts behave the same from the CLI, from pytest, and inside Airflow.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else PIPELINE_ROOT / p)

    dvc_push = os.environ.get("DVC_PUSH")
    if dvc_push is not None:
        cfg.setdefault("dvc", {})["push"] = dvc_push.strip().lower() in {"1", "true", "yes"}
    return cfg


@contextmanager
def atomic_path(target: str | Path) -> Iterator[Path]:
    """Yield a temporary path; rename it onto ``target`` only if the block succeeds.

    Use this for writers that need a file path, e.g.
    ``with atomic_path(out) as tmp: df.to_parquet(tmp)``.
    A crash leaves the previous target untouched and no half-written file.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write(target: str | Path, data: str | bytes) -> Path:
    """Atomically write text or bytes to ``target``."""
    with atomic_path(target) as tmp:
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(data, encoding="utf-8")
    return Path(target)


def write_summary(task_id: str, summary: dict[str, Any], cfg: dict[str, Any]) -> Path:
    """Save one task's JSON summary. Each task writes its own file, never a shared one."""
    out = Path(cfg["paths"]["summaries_dir"]) / f"{task_id}.json"
    return atomic_write(out, json.dumps(summary, indent=2, default=str, sort_keys=True))


@contextmanager
def timed(summary: dict[str, Any]) -> Iterator[None]:
    """Record ``duration_s`` into ``summary`` for the enclosed block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        summary["duration_s"] = round(time.perf_counter() - start, 3)
