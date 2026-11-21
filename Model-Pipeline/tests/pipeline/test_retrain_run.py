"""A retraining run end to end (docs/retraining.md): train on drift batches 0-1 plus obfuscated
rows, hold out batch 2, score it whole, and hand promote_check what it needs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.student.synthetic import build, write_drift
from tests.pipeline.test_run import FAST, write_eval_tables
from tests.student.conftest import smoke_cfg

pytest.importorskip("onnxruntime")
pytest.importorskip("matplotlib")

CIVIL = ["toxicity", "obscene", "threat", "insult", "identity_attack"]


@pytest.fixture(scope="module")
def retrain_run(tmp_path_factory):
    from src.pipeline.run import run

    root = tmp_path_factory.mktemp("retrain")
    build(root, n=400)
    data = write_eval_tables(root)
    ids = write_drift(data / "civil" / "drift", root / "store", batches=3, n=60)
    cfg = smoke_cfg(root, *FAST, f"pipeline.data_dir={data}", "loss.alpha=0.5",
                    "run_name=retrain", "retrain.drift_batches=[0,1]",
                    "retrain.holdout_batch=2", "retrain.obfuscation_augment.rows=30")
    return root, data, ids, run(cfg)


def test_every_step_runs(retrain_run):
    _, _, _, record = retrain_run
    errors = {k: v for k, v in record["errors"].items() if not k.endswith("_traceback")}
    assert errors == {}, errors
    assert record["data"]["holdout_batch"] == 2 and record["data"]["rows"]["drift_holdout"] == 60


def test_training_used_the_drift_and_augmented_rows_but_not_the_held_out_batch(retrain_run):
    _, _, ids, record = retrain_run
    run_dir = Path(record["run_dir"])
    trained = set((run_dir / "splits" / "train_ids.txt").read_text().split())
    assert set(ids[0]) | set(ids[1]) <= trained
    assert not set(ids[2]) & trained
    assert any(i.endswith("_aug") for i in trained)
    meta = json.loads((run_dir / "model" / "meta.json").read_text())
    assert meta["retrain"]["drift_batches"] == [0, 1] and meta["retrain"]["holdout_batch"] == 2
    assert meta["retrain"]["drift_rows"] == 120 and meta["retrain"]["obfuscation_augment"]["rows"]
    package = json.loads((run_dir / "package" / "meta.json").read_text())
    assert package["retrain"] == meta["retrain"]          # promote_check --trained-on reads it


def test_the_held_out_batch_is_scored_whole_with_civil_metrics(retrain_run):
    _, _, ids, record = retrain_run
    run_dir = Path(record["run_dir"])
    pred = pd.read_parquet(run_dir / "predictions" / "drift_holdout.parquet")
    assert sorted(pred["id"]) == sorted(ids[2])
    gold = pd.read_parquet(run_dir / "gold" / "drift_holdout.parquet")
    assert "text" not in gold.columns and sorted(gold["id"]) == sorted(ids[2])
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    assert metrics["drift_holdout"]["labels"] == CIVIL


def test_promote_check_runs_on_the_job_output(retrain_run, tmp_path):
    from src.pipeline.promote_check import main

    _, data, _, record = retrain_run
    run_dir = Path(record["run_dir"])
    cand = pd.read_parquet(run_dir / "predictions" / "drift_holdout.parquet")
    # "Production" as drift_ingest stores it: score_<label>, here an uninformative model.
    rng = np.random.default_rng(0)
    live = pd.DataFrame({"id": cand["id"], "model_version": "prod-v1"})
    for label in ["toxicity", "severe_toxicity", *CIVIL[1:]]:
        live[f"score_{label}"] = rng.random(len(live))
    live.to_parquet(tmp_path / "batch_02.parquet", index=False)
    trained = json.loads((run_dir / "package" / "meta.json").read_text())["retrain"]
    code = main(["--batch", "2", "--candidate", str(run_dir / "predictions" /
                                                    "drift_holdout.parquet"),
                 "--production", str(tmp_path / "batch_02.parquet"),
                 "--gold", str(data / "civil" / "drift" / "batch_02.parquet"),
                 "--production-model-version", "prod-v1",
                 "--trained-on", *map(str, trained["drift_batches"]), "--out", str(tmp_path)])
    report = json.loads((tmp_path / "promote_check.json").read_text())
    assert report["rows"] == 60 and report["trained_on"] == [0, 1]
    assert code == (0 if report["passed"] else 1)
