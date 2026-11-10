"""End-to-end: the tiny offline model on synthetic data, gold-only and mixed loss."""

import json
import math

import pandas as pd
import pytest

from src.student.predict import check_prediction_frame
from src.student.synthetic import build
from src.student.train import run
from tests.student.conftest import smoke_cfg


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_training_runs_end_to_end(synthetic_root, alpha):
    summary = run(smoke_cfg(synthetic_root, f"loss.alpha={alpha}"))
    run_dir = synthetic_root / "runs" / summary["run_name"]

    for path in ("model/meta.json", "model/config.json", "predictions/val.parquet",
                 "splits/train_ids.txt", "splits/val_ids.txt", "metrics.json", "summary.json"):
        assert (run_dir / path).exists(), path
    assert (synthetic_root / "reports" / "train.json").exists()

    pred = pd.read_parquet(run_dir / "predictions" / "val.parquet")
    check_prediction_frame(pred)
    val_ids = (run_dir / "splits" / "val_ids.txt").read_text().split()
    assert sorted(pred["id"]) == sorted(val_ids)

    meta = json.loads((run_dir / "model" / "meta.json").read_text())
    assert meta["alpha"] == alpha
    if alpha < 1:
        assert summary["soft_labels"]["full_validation"]
        assert meta["teacher_version"] == "stack-v1" and len(meta["stack_sha256"]) == 64
    else:
        assert summary["soft_labels"] is None and meta["stack_sha256"] is None
    assert summary["best_val_mean_auc"] > 0.6          # the markers are learnable
    assert summary["mlflow_run_id"]


def test_failed_soft_labels_still_train(tmp_path):
    build(tmp_path, n=600, failed=4)
    summary = run(smoke_cfg(tmp_path, "loss.alpha=0.0", "train.max_steps=10",
                            "train.epochs=1", "mlflow.enabled=false"))
    assert summary["soft_labels"]["failed_rows"] == 4
    assert math.isfinite(summary["history"][0]["train_loss"])


def test_same_seed_gives_same_result(synthetic_root):
    overrides = ("train.max_steps=15", "train.epochs=1", "mlflow.enabled=false")
    a = run(smoke_cfg(synthetic_root, *overrides))
    b = run(smoke_cfg(synthetic_root, *overrides))
    assert a["best_val_mean_auc"] == pytest.approx(b["best_val_mean_auc"], abs=1e-6)
