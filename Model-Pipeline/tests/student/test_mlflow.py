"""MLflow setup: local runs keep their artifacts under paths.mlflow_artifacts, and one run's
SQLite store does not leak into the next run in the same process."""

import os
from pathlib import Path

import pytest

from src.student.train import _mlflow_start
from tests.student.conftest import smoke_cfg

mlflow = pytest.importorskip("mlflow")


def _configured(cfg) -> str:
    return Path(cfg["paths"]["mlflow_artifacts"]).resolve().as_uri()


def _artifact_location(cfg) -> str:
    with _mlflow_start(cfg, "probe"):
        pass
    return mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"]).artifact_location


def test_local_runs_use_the_configured_artifact_store(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    first, second = smoke_cfg(tmp_path / "a"), smoke_cfg(tmp_path / "b")

    assert _artifact_location(first) == _configured(first)
    # set_tracking_uri writes MLFLOW_TRACKING_URI; a local run must not leave it behind.
    assert "MLFLOW_TRACKING_URI" not in os.environ

    assert _artifact_location(second) == _configured(second)
    assert mlflow.get_tracking_uri() == f"sqlite:///{second['paths']['mlflow_db']}"
    assert (tmp_path / "b" / "mlruns" / "mlflow.db").exists()


def test_a_tracking_server_from_the_environment_is_kept(tmp_path, monkeypatch):
    server = f"sqlite:///{tmp_path / 'server.db'}"   # stands in for the VM's server
    monkeypatch.setenv("MLFLOW_TRACKING_URI", server)
    cfg = smoke_cfg(tmp_path)

    location = _artifact_location(cfg)
    assert mlflow.get_tracking_uri() == server
    assert os.environ["MLFLOW_TRACKING_URI"] == server
    assert location != _configured(cfg)
    assert not Path(cfg["paths"]["mlflow_db"]).exists()
