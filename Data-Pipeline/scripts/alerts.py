"""Alerting for the data pipeline.

Contract: every lane raises alerts only through ``send_alert(level, title, details)``.
If ``SLACK_WEBHOOK_URL`` is not set (local dev, CI, graders), alerts are only logged,
so nothing ever fails because Slack isn't configured.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from scripts.utils import get_logger

logger = get_logger(__name__)
LEVELS = {"warn": ":warning:", "fail": ":rotating_light:", "ok": ":white_check_mark:"}


def send_alert(level: str, title: str, details: dict[str, Any] | None = None) -> bool:
    """Log an alert and post it to Slack if a webhook is configured.

    Returns True if the alert was posted to Slack, False if it was only logged.
    Never raises: a broken webhook must not fail the pipeline task that called it.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
    details = details or {}
    log_fn = logger.error if level == "fail" else logger.warning if level == "warn" else logger.info
    log_fn("ALERT [%s] %s %s", level, title, json.dumps(details, default=str))

    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        return False

    lines = [f"{LEVELS[level]} *{title}*"] + [f"• *{k}*: {v}" for k, v in details.items()]
    payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
    request = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # noqa: BLE001 - alerting must never crash a task
        logger.error("Slack alert failed: %s", exc)
        return False


def task_failure_callback(context: dict[str, Any]) -> None:
    """Airflow ``on_failure_callback``: posts DAG id, task id, error, and log link."""
    ti = context.get("task_instance") or context.get("ti")
    send_alert(
        "fail",
        "Distillate pipeline task failed",
        {
            "dag": getattr(ti, "dag_id", "unknown"),
            "task": getattr(ti, "task_id", "unknown"),
            "try": getattr(ti, "try_number", "unknown"),
            "error": str(context.get("exception", "unknown"))[:500],
            "log": getattr(ti, "log_url", "n/a"),
        },
    )
