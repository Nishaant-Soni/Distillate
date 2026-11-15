"""Shared by every test: no test talks to an MLflow server unless it sets one itself."""

import pytest

# A server (and credentials) exported in the shell, as for the MLflow loggers, would otherwise
# reach every test that logs: the training smoke test would write runs to the team server, or
# fail with 401 without the password. MLflow also writes the URI and experiment id into the
# environment itself, so one test's local store would otherwise leak into the next.
MLFLOW_ENV = ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI", "MLFLOW_EXPERIMENT_ID",
              "MLFLOW_EXPERIMENT_NAME", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


@pytest.fixture(autouse=True)
def no_mlflow_server(monkeypatch):
    for name in MLFLOW_ENV:
        monkeypatch.delenv(name, raising=False)
