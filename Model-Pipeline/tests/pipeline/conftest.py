"""Shared fixtures: the registry config and a fresh SQLite MLflow store per test."""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from src.pipeline import register_model as reg  # noqa: E402
from src.pipeline import registry  # noqa: E402


@pytest.fixture
def cfg():
    return registry.load_config()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """(client, artifact root) on a fresh SQLite store; no Slack."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = registry.client(uri)
    artifacts = (tmp_path / "artifacts").as_uri()
    client.create_experiment("distillate-student", artifact_location=artifacts)
    client.create_experiment(reg.REGISTRY_EXPERIMENT, artifact_location=artifacts)
    return client
