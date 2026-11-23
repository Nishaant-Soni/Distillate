"""The demo app's settings, all from environment variables.

The same image runs locally, in tests and on Cloud Run; only the environment changes. Anything
not set falls back to a safe default: no ``DISTILLATE_API_URL`` means the built-in mock, and a
missing registry, deployment log or retraining record shows as "not yet" on the Status page.
On Cloud Run the MLflow password comes from Secret Manager as an environment variable
(``--set-secrets``), so it is never in the image or the repository.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_url: str | None              # the serving API (contract section 4); None: the mock
    api_timeout_s: float
    mlflow_uri: str | None           # the team MLflow server; None: no registry on Status
    mlflow_username: str | None
    mlflow_password: str | None
    registered_model: str
    production_alias: str
    staging_alias: str
    grafana_url: str | None
    deployments_log: str | None      # deploy.sh's deployments.jsonl (a local path for now)
    retrain_records: str | None      # a folder of retrain_dag's <run_id>.json records
    session_per_minute: int          # requests one visitor may score per minute
    instance_per_minute: int         # requests all visitors on one instance may score per minute
    max_chars: int                   # longest text the Try-it page sends


def _text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name, "").strip()
    return int(value) if value else default


def load(env: Mapping[str, str] | None = None) -> Settings:
    """Settings from ``env`` (default: the process environment)."""
    env = os.environ if env is None else env
    return Settings(
        api_url=_text(env, "DISTILLATE_API_URL"),
        api_timeout_s=float(env.get("DISTILLATE_API_TIMEOUT_S", "").strip() or 10),
        mlflow_uri=_text(env, "MLFLOW_TRACKING_URI"),
        mlflow_username=_text(env, "MLFLOW_TRACKING_USERNAME"),
        mlflow_password=_text(env, "MLFLOW_TRACKING_PASSWORD"),
        registered_model=_text(env, "DISTILLATE_REGISTERED_MODEL") or "distillate-student",
        production_alias=_text(env, "DISTILLATE_PRODUCTION_ALIAS") or "production",
        staging_alias=_text(env, "DISTILLATE_STAGING_ALIAS") or "staging",
        grafana_url=_text(env, "GRAFANA_URL"),
        deployments_log=_text(env, "DEPLOYMENTS_LOG"),
        retrain_records=_text(env, "RETRAIN_RECORDS"),
        session_per_minute=_int(env, "DISTILLATE_SESSION_PER_MINUTE", 10),
        instance_per_minute=_int(env, "DISTILLATE_INSTANCE_PER_MINUTE", 60),
        max_chars=_int(env, "DISTILLATE_MAX_CHARS", 2000),
    )
