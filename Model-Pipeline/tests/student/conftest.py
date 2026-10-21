"""Synthetic inputs and an offline smoke config, all under pytest's tmp_path."""

from __future__ import annotations

import pytest

from src.student.config import PIPELINE_ROOT, load_config
from src.student.synthetic import build


def smoke_cfg(root, *overrides):
    """configs/smoke.yaml with every output path under ``root``."""
    return load_config(PIPELINE_ROOT / "configs" / "smoke.yaml", [
        f"paths.raw_train={root}/wiki_train_raw.parquet",
        f"soft_labels.local_root={root}/store",
        f"paths.out_dir={root}/runs",
        f"paths.cache_dir={root}/cache",
        f"paths.mlflow_db={root}/mlruns/mlflow.db",
        f"paths.mlflow_artifacts={root}/mlruns/artifacts",
        f"paths.reports_dir={root}/reports",
        *overrides,
    ])


@pytest.fixture
def synthetic_root(tmp_path):
    build(tmp_path, n=600, seed=42)
    return tmp_path
